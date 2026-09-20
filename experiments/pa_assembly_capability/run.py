# -*- coding: utf-8 -*-
"""PA 部分装配无矩阵算子数据点的自包含调度与测量入口.

本脚本同时承担「调度器」与「独立子进程测量器」两个角色, 结构与
``experiments/ea_assembly_capability/run.py`` 一致, 共用代码在 ``experiments/_common/``.

1. 调度模式 (默认入口, 每个数据点独占一个子进程):
   - python run.py --list                                   # 列出已注册数据点
   - python run.py --all --check-only                       # 只打印将执行的子进程命令
   - python run.py --case element-cache --grid 32 --monitor
   - python run.py --case pa-matvec --grid 32 --repeats 20 --monitor
   - python run.py --case pa-continuous --grid 32 --monitor
   - python run.py --case pa-cg-solve --grid 32 --monitor
   - python run.py --case cpu-baseline --monitor            # 单核硬件基线 (memcpy 带宽 + dgemm 算力)
   - python run.py --verify-ea --n 8                        # 逐位核对 PA 与 EA 算子乘代数等价性

2. Worker 模式 (由调度器在独立进程中调用, 保证内存高水位严格隔离):
   - python run.py --worker --cache  --method fast --n 32 --output outputs/cache_fast_n32.json
   - python run.py --worker --matvec --method fast --n 32 --repeats 20 --output outputs/matvec_fast_n32.json
   - python run.py --worker --continuous --method fast --n 32 --repeats 20 --output outputs/cache_matvec_continuous_fast_n32.json
   - python run.py --worker --solve  --method fast --n 32 --maxiter 5000 --tol 1e-6 \
         --output outputs/solve_fast_n32.json
   - python run.py --worker --baseline --output outputs/baseline_cpu.json

被测对象是仓库核心代码 ``soptx.fem.levels.partial.PartialAssembly`` 及其底层:
``LagrangeFEMAnalyzer.assemble_stiff_matrix('pa')`` 构造并缓存积分点几何量 (jacobi_inverse, weighted_measure,
grad_ref) 与 cell2dof, 不组装也不常驻任何单元刚度矩阵; ``@`` 走 ``PartialAssembly.__matmul__``
(gather -> dof_to_quad -> qfunction -> quad_to_dof -> scatter_add) 外包 ``ConstrainedOperator``;
Jacobi-PCG 用 ``soptx.solvers.cg`` 与 ``DiagonalPreconditioner``, 对角由 ``PartialAssembly.diagonal()`` 闭式给出.
本脚本不含任何算子或求解器的自有实现.

问题、网格、空间、材料的构建路径与 fa / ea 完全相同 (``_common.fe_problem``).
阶段 1 的 PA 与 EA 算子乘在相同位移向量下是否代数等价由 ``--verify-ea`` 在同一进程内核对.

内存口径 (CPU): 每个阶段先记 before = 当前 VmRSS, 再向 /proc/self/clear_refs 写 5 重置 VmHWM,
阶段结束读 VmHWM 作为该阶段的绝对峰值 peak, net = peak - before. 全程峰值 (process_max_rss)
= 各阶段峰值的最大值.

阶段划分:
  cache  面板: mesh (网格 + 空间 + 材料 + 分析器) -> cache (assemble_stiff_matrix: 积分点几何 + cell2dof)
  matvec 面板: mesh -> assemble (facade.assemble(): 几何数据 + 体力右端 + Dirichlet 投影) -> warmup
               -> matvec (刚度算子乘 K x, 重复 repeats 次)
  solve  面板: mesh -> assemble -> setup_solve (对角 + 预条件子) -> solve (cg, 每步调用系统算子)
  baseline 面板: 与网格无关, 单线程 memcpy 带宽与 dgemm 算力, 供阶段 2 换算占比

产物命名: <kind>_<method>_n<N>.json; 后处理与对比表见 compare.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
for _p in (_THIS_DIR, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from _common import scheduler  # noqa: E402
from _common.fe_problem import (  # noqa: E402
    MESH_TYPE,
    METHOD_NAMES,
    PROBLEM_NAME,
    build_problem_space,
    import_fe_stack_cpu,
    mesh_facts,
)
from _common.baseline import measure_baseline, print_baseline  # noqa: E402
from _common.metrology import StageMeter, cur_rss_kib, peak_rss_kib, reset_peak_rss  # noqa: E402

# -----------------------------------------------------------------------------
# 1. 核心算子的构建与测量 (Worker 核心)
# -----------------------------------------------------------------------------

LOCAL_DOFS = 12  # tet4 向量 P1: 4 节点 x 3 分量
DEGREE = 1
DEVICE = "cpu"


class ElasticityPAOperator:
    """线弹性 Matrix-Free PA 刚度算子与边界条件的懒装配缓存门面."""

    def __init__(
        self,
        space: Any,
        pde: Any,
        material: Any,
        degree: int = 1,
        assembly_method: str = "fast",
    ) -> None:
        from soptx.fem.analyzers.builders import build_serial_analyzer

        self.space = space
        self.pde = pde
        self.material = material
        self.degree = degree
        self.assembly_method = assembly_method
        self.analyzer = build_serial_analyzer(
            space, pde, material, degree=degree, operator_level="pa", assembly_method=assembly_method
        )
        self._system_operator: Any = None
        self._load_vector: Any = None
        self._prescribed: Any = None
        self._boundary_dofs: Any = None

    def assemble(self) -> tuple[Any, np.ndarray]:
        """执行 PA 积分点几何缓存与体力向量装配, 并施加 Dirichlet 边界条件对角投影."""
        stiff_matrix = self.analyzer.assemble_stiff_matrix()
        body_force = self.analyzer.assemble_body_force_vector()
        system_operator, load_vector = self.analyzer.apply_bc(stiff_matrix, body_force)
        self._system_operator = system_operator
        self._load_vector = np.asarray(load_vector)
        self._prescribed = np.asarray(self.analyzer.prescribed_solution)
        threshold_uh = self.pde.is_dirichlet_boundary()
        self._boundary_dofs = np.asarray(self.space.is_boundary_dof(threshold=threshold_uh, method="interp"), dtype=bool)
        return self._system_operator, self._load_vector

    @property
    def system_operator(self) -> Any:
        return self._system_operator

    @property
    def load_vector(self) -> np.ndarray:
        return self._load_vector

    @property
    def prescribed_solution(self) -> np.ndarray:
        return self._prescribed

    @property
    def boundary_dofs(self) -> np.ndarray:
        return self._boundary_dofs


def _partial_data(operator: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 PartialAssembly 算子提取核心常驻数组: cell2dof, jacobi_inverse, weighted_measure."""
    cell2dof = np.asarray(operator.restriction.cell2dof)
    jacobi_inverse = np.asarray(operator.dof_to_quad._jacobi_inverse)
    weighted_measure = np.asarray(operator.qfunction._weighted_measure)
    return cell2dof, jacobi_inverse, weighted_measure


def _relerr(y: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_ref)) / max(float(np.max(np.abs(y_ref))), 1e-300))


def _build_facade(method: str, n: int) -> tuple[Dict[str, Any], StageMeter]:
    """构建网格 / 空间 / 材料与 ``ElasticityPAOperator`` 门面 (mesh 阶段), 不触发装配."""
    import_fe_stack_cpu()

    meter = StageMeter()
    with meter.stage("mesh"):
        problem, mesh, vs, material = build_problem_space(n)
        facade = ElasticityPAOperator(vs, problem, material, degree=DEGREE, assembly_method=method)

    ctx: Dict[str, Any] = {
        "mesh": mesh,
        "vs": vs,
        "problem": problem,
        "material": material,
        "facade": facade,
        "facts": mesh_facts(mesh, vs),
    }
    return ctx, meter


def _assemble_system(ctx: Dict[str, Any], meter: StageMeter) -> None:
    """assemble 阶段: ``facade.assemble()`` 一次给出 PA 几何缓存、体力右端与 Dirichlet 投影算子."""
    facade = ctx["facade"]
    with meter.stage("assemble"):
        operator, load = facade.assemble()
    pa_op = facade.analyzer._K
    cell2dof, jacobi_inverse, weighted_measure = _partial_data(pa_op)
    ctx.update(
        {
            "operator": operator,  # ConstrainedOperator
            "pa_op": pa_op,        # PartialAssembly 裸算子
            "load": load,
            "cell2dof": cell2dof,
            "jacobi_inverse": jacobi_inverse,
            "weighted_measure": weighted_measure,
            "is_bd": facade.boundary_dofs,
        }
    )


def _finish(panel: str, ctx: Dict[str, Any], method: str, n: int, meter: StageMeter) -> Dict[str, Any]:
    """在全部阶段结束后组装公共字段 (网格事实、PA 理论常驻、各阶段峰值 / 净增与单价)."""
    facts = ctx["facts"]
    Ndof = facts["Ndof"]
    pa_op = ctx.get("pa_op")
    c2d = ctx["cell2dof"]
    j_inv = ctx["jacobi_inverse"]
    w_meas = ctx["weighted_measure"]

    persistent = int(pa_op.persistent_bytes()) if pa_op is not None else int(c2d.nbytes + j_inv.nbytes + w_meas.nbytes)
    nq = int(j_inv.shape[1])

    def kb_per_dof(nbytes: float) -> float:
        return round(nbytes / Ndof / 1000, 2)

    out: Dict[str, Any] = {
        "panel": panel,
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": meter.memory_kind,
        "problem": PROBLEM_NAME,
        "mesh_type": MESH_TYPE,
        "operator_impl": "soptx.fem.levels.partial.PartialAssembly",
        "method": method,
        "n": n,
        **facts,
        "quadrature_points": nq,
        "jacobi_inverse_shape": [int(s) for s in j_inv.shape],
        "jacobi_inverse_MiB": round(int(j_inv.nbytes) / 2**20, 1),
        "weighted_measure_MiB": round(int(w_meas.nbytes) / 2**20, 1),
        "cell2dof_MiB": round(int(c2d.nbytes) / 2**20, 1),
        "operator_persistent_MiB": round(persistent / 2**20, 1),
        "operator_persistent_KB_per_dof": kb_per_dof(persistent),
        **meter.fields(),
        "process_max_rss_KB_per_dof": kb_per_dof(meter.max_peak_kib() * 1024),
    }
    if "cache" in meter.records:
        out.update(
            {
                "cache_KB_per_dof": kb_per_dof(meter.net_bytes("cache")),
                "cache_peak_KB_per_dof": kb_per_dof(meter.peak_bytes("cache")),
            }
        )
    return out


def _malloc_trim() -> bool:
    """把 glibc 持有但未归还内核的空闲页归还; 非 glibc 平台返回 False."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_size_t(0))
        return True
    except (OSError, AttributeError):
        return False


def measure_cache(method: str, n: int) -> dict:
    """阶段 1 (panel cache): 测 ``assemble_stiff_matrix('pa')`` 缓存积分点几何量与限制算子."""
    import gc

    ctx, meter = _build_facade(method, n)
    facade = ctx["facade"]
    gc.collect()
    _malloc_trim()
    with meter.stage("cache"):
        pa_op = facade.analyzer.assemble_stiff_matrix()
    cell2dof, jacobi_inverse, weighted_measure = _partial_data(pa_op)
    ctx.update(
        {
            "pa_op": pa_op,
            "cell2dof": cell2dof,
            "jacobi_inverse": jacobi_inverse,
            "weighted_measure": weighted_measure,
        }
    )

    gc.collect()
    _malloc_trim()

    out = _finish("cache", ctx, method, n, meter)
    out.update(
        {
            "cache_seconds": meter.seconds("cache"),
            "quad_points_per_cell": int(jacobi_inverse.shape[1]),
        }
    )
    return out


def _timed(fn, repeats: int) -> tuple[list[float], Any]:
    times: list[float] = []
    res = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        times.append(time.perf_counter() - t0)
    return times, res


def measure_matvec(method: str, n: int, repeats: int = 20, seed: int = 0) -> dict:
    """阶段 2 (panel matvec): 核心 PA 刚度算子乘 K x = G^T B^T D B G x 计时."""
    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    pa_op = ctx["pa_op"]
    facts = ctx["facts"]
    Ndof = facts["Ndof"]
    NC = facts["NC"]
    j_inv = ctx["jacobi_inverse"]
    w_meas = ctx["weighted_measure"]
    c2d = ctx["cell2dof"]
    nq = int(j_inv.shape[1])

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(Ndof)

    # warmup
    with meter.stage("warmup"):
        _ = pa_op @ x

    with meter.stage("matvec"):
        times, y = _timed(lambda: pa_op @ x, repeats)

    y = np.asarray(y)
    t_med = statistics.median(times)
    # 最小搬运量: jacobi_inverse + weighted_measure + cell2dof + 读 x (8B) + 写 y (8B)
    bytes_moved_min = j_inv.nbytes + w_meas.nbytes + c2d.nbytes + 16 * Ndof
    # 理论 FLOPs: 两次梯化 (72 NQ) + 本构 (72 NQ) + 梯化转置 (72 NQ) = 216 NQ FLOPs/cell
    flops_per_cell = 216 * nq

    out = _finish("matvec", ctx, method, n, meter)
    out.update(
        {
            "repeats": repeats,
            "seed": seed,
            "matvec_impl": "soptx.fem.levels.partial.PartialAssembly.__matmul__: gather -> dof_to_quad -> qfunction -> quad_to_dof -> scatter_add",
            "matvec_seconds_median": round(t_med, 6),
            "matvec_seconds_min": round(min(times), 6),
            "matvec_seconds_all": [round(t, 6) for t in times],
            "bytes_moved_min_per_matvec": int(bytes_moved_min),
            "effective_gbps_lower_bound": round(bytes_moved_min / t_med / 1e9, 2),
            "gflops": round(flops_per_cell * NC / t_med / 1e9, 2),
            "y_norm": float(np.linalg.norm(y)),
        }
    )
    return out


def measure_continuous(method: str, n: int, repeats: int = 20) -> dict:
    """连续测量面板: 同一进程内测量 cache -> input -> first_matvec -> repeat_matvec."""
    import gc

    ctx, mesh_meter = _build_facade(method, n)
    stages = {}
    rng = np.random.default_rng(0)
    times = [0.0] * repeats

    @contextlib.contextmanager
    def stage(name: str):
        before = cur_rss_kib()
        reset = reset_peak_rss()
        start = time.perf_counter()
        yield
        seconds = time.perf_counter() - start
        after = cur_rss_kib()
        peak = max(before, after, peak_rss_kib())
        stages[name] = {
            "before_kib": before,
            "peak_kib": peak,
            "after_kib": after,
            "net_kib": peak - before,
            "t_s": seconds,
            "reset_supported": reset,
        }

    gc.collect()
    _malloc_trim()
    with stage("cache"):
        pa_op = ctx["facade"].analyzer.assemble_stiff_matrix()
    with stage("input"):
        x = rng.standard_normal(ctx["facts"]["Ndof"])
    with stage("first_matvec"):
        y = pa_op @ x
    with stage("repeat_matvec"):
        for i in range(repeats):
            start = time.perf_counter()
            y = pa_op @ x
            times[i] = time.perf_counter() - start

    med = statistics.median(times)
    peak = max(mesh_meter.max_peak_kib(), *(r["peak_kib"] for r in stages.values()))
    facts = ctx["facts"]
    persistent_bytes = int(pa_op.persistent_bytes())

    return {
        "panel": "continuous",
        "device": "CPU",
        "device_type": "cpu",
        "problem": PROBLEM_NAME,
        "mesh_type": MESH_TYPE,
        "operator_impl": "soptx.fem.levels.partial.PartialAssembly",
        "method": method,
        "n": n,
        **facts,
        "stages": stages,
        "continuous_process_peak_MiB": round(peak / 1024, 1),
        "operator_persistent_MiB": round(persistent_bytes / 2**20, 1),
        "workspace_first_net_MiB": round(stages["first_matvec"]["net_kib"] / 1024, 1),
        "repeat_matvec_steady_net_MiB": round(
            max(0, stages["repeat_matvec"]["peak_kib"] - stages["first_matvec"]["after_kib"]) / 1024, 1
        ),
        "matvec_seconds_median": round(med, 6),
        "matvec_seconds_all": [round(t, 6) for t in times],
        "y_norm": float(np.linalg.norm(np.asarray(y))),
    }


def measure_solve(method: str, n: int, maxiter: int = 5000, tol: float = 1e-6) -> dict:
    """阶段 3 (panel solve): 核心 Jacobi-PCG 求解制造解问题."""
    from soptx.solvers import DiagonalPreconditioner, cg

    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    facade = ctx["facade"]
    operator = ctx["operator"]
    load = ctx["load"]

    with meter.stage("setup_solve"):
        diag = facade.analyzer.assemble_operator_diagonal(operator)
        precond = DiagonalPreconditioner(diag)
        x0 = facade.prescribed_solution

    with meter.stage("solve"):
        x, info = cg(
            operator, load, x0, precond,
            atol=0.0, rtol=tol, maxit=maxiter, returninfo=True, print_level=0,
        )

    it_count = int(info["niter"])
    residual = float(info["residual"])
    reference = float(info["reference_norm"])
    true_res = float(np.linalg.norm(np.asarray(operator @ x) - load))
    load_norm = float(np.linalg.norm(load))
    solve_s = meter.seconds("solve")
    reason = info.get("reason")

    out = _finish("solve", ctx, method, n, meter)
    out.update(
        {
            "solver_impl": "soptx.solvers.cg + DiagonalPreconditioner (Jacobi)",
            "preconditioner": "jacobi",
            "boundary": "pde-dirichlet",
            "rhs": "pde-body-force",
            "tolerance": tol,
            "maxiter": maxiter,
            "iterations": it_count,
            "converged": bool(info["converged"]),
            "reason": str(getattr(reason, "name", reason)),
            "final_relres": residual / reference if reference > 0 else float("nan"),
            "true_relres": true_res / load_norm if load_norm > 0 else float("nan"),
            "iterations_per_n": round(it_count / n, 3),
            "solve_seconds": round(solve_s, 3),
            "seconds_per_iteration": round(solve_s / it_count, 6) if it_count else 0.0,
            "n_boundary_dofs": int(ctx["is_bd"].sum()),
        }
    )
    return out


def verify_ea(n: int = 8) -> bool:
    """逐位核对 PA 与 EA 算子乘在相同位移向量下的数值等价性 (浮点误差内)."""
    import_fe_stack_cpu()
    from soptx.fem.analyzers.builders import build_serial_analyzer

    print(f"[verify-ea] 构建 n={n} 问题网格并比较 PA vs EA 算子乘...")
    problem, mesh, vs, material = build_problem_space(n)

    analyzer_ea = build_serial_analyzer(vs, problem, material, degree=1, operator_level="ea", assembly_method="fast")
    K_ea = analyzer_ea.assemble_stiff_matrix()

    analyzer_pa = build_serial_analyzer(vs, problem, material, degree=1, operator_level="pa", assembly_method="fast")
    K_pa = analyzer_pa.assemble_stiff_matrix()

    rng = np.random.default_rng(42)
    x = rng.standard_normal(vs.number_of_global_dofs())

    y_ea = np.asarray(K_ea @ x)
    y_pa = np.asarray(K_pa @ x)

    rel_diff = float(np.linalg.norm(y_ea - y_pa) / np.linalg.norm(y_ea))
    print(f"[verify-ea] Ndof={vs.number_of_global_dofs():,} 相对误差: {rel_diff:.3e}")
    if rel_diff < 1e-12:
        print("[verify-ea] 校验通过: PA 与 EA 算子乘严格等价!")
        return True
    else:
        print(f"[verify-ea] 校验失败: 相对误差 {rel_diff:.3e} 超出阈值 1e-12")
        return False


# -----------------------------------------------------------------------------
# 2. CLI 调度器与主入口
# -----------------------------------------------------------------------------

def _build_worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PA 评测独立 Worker")
    p.add_argument("--worker", action="store_true", required=True)
    p.add_argument("--cache", action="store_true", help="运行阶段 1 积分点几何缓存测量")
    p.add_argument("--matvec", action="store_true", help="运行阶段 2 算子乘计时测量")
    p.add_argument("--continuous", action="store_true", help="运行连续内存观测")
    p.add_argument("--solve", action="store_true", help="运行阶段 3 Jacobi-PCG 求解测量")
    p.add_argument("--baseline", action="store_true", help="运行单核硬件基线测试")
    p.add_argument("--method", default="fast", choices=METHOD_NAMES)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--output", type=str, default="")
    return p


def _dispatch_worker(args: argparse.Namespace) -> int:
    if args.baseline:
        payload = measure_baseline()
    elif args.cache:
        payload = measure_cache(args.method, args.n)
    elif args.matvec:
        payload = measure_matvec(args.method, args.n, repeats=args.repeats, seed=args.seed)
    elif args.continuous:
        payload = measure_continuous(args.method, args.n, repeats=args.repeats)
    elif args.solve:
        payload = measure_solve(args.method, args.n, maxiter=args.maxiter, tol=args.tol)
    else:
        print("未指定 worker 测量任务 (--cache / --matvec / --continuous / --solve / --baseline)", file=sys.stderr)
        return 2

    if args.output:
        out_p = Path(args.output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[worker] 产物已写入 {out_p}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _build_scheduler_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PA 部分装配评测调度器")
    p.add_argument("--list", action="store_true", help="打印已注册工况列表")
    p.add_argument("--all", action="store_true", help="执行全部工况")
    p.add_argument("--case", choices=list(config.PANELS) + ["element-cache", "pa-matvec", "pa-continuous", "pa-cg-solve", "cpu-baseline"], help="指定要跑的 panel 或 case-id")
    p.add_argument("--method", default="", help="覆盖单刚方法")
    p.add_argument("--grid", type=int, default=0, help="覆盖网格 n")
    p.add_argument("--repeats", type=int, default=0, help="覆盖 matvec 重复次数")
    p.add_argument("--check-only", action="store_true", help="只打印命令不执行")
    p.add_argument("--monitor", action="store_true", help="调度子进程时显示进度条")
    p.add_argument("--verify-ea", action="store_true", help="校验 PA 与 EA 算子乘等价性")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        wp = _build_worker_parser()
        wargs = wp.parse_args(argv)
        return _dispatch_worker(wargs)

    sp = _build_scheduler_parser()
    sargs = sp.parse_args(argv)

    if sargs.verify_ea:
        ok = verify_ea(n=sargs.grid or 8)
        return 0 if ok else 1

    figure, cases = config.load_cases()

    if sargs.list:
        print(f"评测项目: {figure.get('title')} ({len(cases)} cases)")
        for c in cases:
            print(f"  [{c.panel:<10}] {c.id:<18} n={c.extra.get('n', '-')} -> {c.artifact_path.name}")
        return 0

    target_cases = []
    if sargs.all:
        target_cases = list(cases)
    elif sargs.case:
        # 兼容 panel 匹配与 id 匹配
        for c in cases:
            if c.id == sargs.case or c.panel == sargs.case:
                target_cases.append(c)
        if not target_cases:
            print(f"未找到匹配工况: {sargs.case}", file=sys.stderr)
            return 1
    else:
        sp.print_help()
        return 0

    print(f"即将调度 {len(target_cases)} 个工况 (单数据点独占子进程)...")
    for c in target_cases:
        cmd = [sys.executable, str(c.script_path)] + list(c.args)
        if sargs.method and "--method" in cmd:
            idx = cmd.index("--method")
            cmd[idx + 1] = sargs.method
        if sargs.grid and "--n" in cmd:
            idx = cmd.index("--n")
            cmd[idx + 1] = str(sargs.grid)
        if sargs.repeats and "--repeats" in cmd:
            idx = cmd.index("--repeats")
            cmd[idx + 1] = str(sargs.repeats)
        n_val = sargs.grid if sargs.grid else c.extra.get("n", 32)
        m_val = sargs.method if sargs.method else c.extra.get("method", "fast")
        if c.panel == "baseline":
            out_p = config.OUTPUT_DIR / c.artifact
        elif c.panel == "cache":
            out_p = config.OUTPUT_DIR / f"cache_{m_val}_n{n_val}.json"
        elif c.panel == "matvec":
            out_p = config.OUTPUT_DIR / f"matvec_{m_val}_n{n_val}.json"
        elif c.panel == "continuous":
            out_p = config.OUTPUT_DIR / f"cache_matvec_continuous_{m_val}_n{n_val}.json"
        elif c.panel == "solve":
            out_p = config.OUTPUT_DIR / f"solve_{m_val}_n{n_val}.json"
        else:
            out_p = c.artifact_path
        cmd.extend(["--output", str(out_p)])

        if sargs.check_only:
            print("  [DRY-RUN]", " ".join(cmd))
            continue

        print(f"==> 启动工况 {c.id} ({c.summary})...")
        t0 = time.perf_counter()
        import subprocess
        res = subprocess.run(cmd)
        ret = res.returncode
        elapsed = time.perf_counter() - t0
        if ret != 0:
            print(f"  [ERROR] 工况 {c.id} 失败, 退出码 {ret} (耗时 {elapsed:.1f}s)")
            return ret
        print(f"  [SUCCESS] 工况 {c.id} 完成 (耗时 {elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
