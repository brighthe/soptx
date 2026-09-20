# -*- coding: utf-8 -*-
"""EA 单元级无矩阵算子三个数据点的自包含调度与测量入口.

本脚本同时承担「调度器」与「独立子进程测量器」两个角色, 结构与
``experiments/fa_assembly_capability/run.py`` 一致, 共用代码在 ``experiments/_common/``.

1. 调度模式 (默认入口, 每个数据点独占一个子进程):
   - python run.py --list                                   # 列出已注册数据点
   - python run.py --all --check-only                       # 只打印将执行的子进程命令
   - python run.py --case element-cache --grid 32 --monitor
   - python run.py --case ea-matvec --grid 32 --repeats 20 --monitor
   - python run.py --case ea-continuous --grid 32 --monitor
   - python run.py --case ea-cg-solve --grid 32 --monitor
   - python run.py --case cpu-baseline --monitor                # 单核硬件基线 (memcpy 带宽 + dgemm 算力)
   - python run.py --verify-fa --n 8                        # 逐位核对 K_e / cell2dof 与 fa 构建路径一致

2. Worker 模式 (由调度器在独立进程中调用, 保证内存高水位严格隔离):
   - python run.py --worker --cache  --method fast --n 32 --output outputs/cache_fast_n32.json
   - python run.py --worker --matvec --method fast --n 32 --repeats 20 --output outputs/matvec_fast_n32.json
   - python run.py --worker --continuous --method fast --n 32 --repeats 20 --output outputs/cache_matvec_continuous_fast_n32.json
   - python run.py --worker --solve  --method fast --n 32 --maxiter 5000 --tol 1e-6 \
         --output outputs/solve_fast_n32.json
   - python run.py --worker --baseline --output outputs/baseline_cpu.json

被测对象是仓库核心代码 ``soptx.fem.matrix_free.ElasticityEAOperator`` (门面) 及其底层:
``LagrangeFEMAnalyzer.assemble_stiff_matrix('ea')`` 用 ``LinearElasticIntegrator.const`` 缓存 K_e 与
cell2dof 并装进未 assembly 的 ``soptx.fem.BilinearForm``; ``@`` 走 ``BilinearForm.__matmul__``
(gather -> einsum -> index_add) 外包 ``DirichletBCOperator`` (Pi_I K Pi_I + Pi_D); Jacobi-PCG 用
``soptx.solvers.cg`` 与 ``DiagonalPreconditioner``, 对角由 ``assemble_operator_diagonal`` 给出.
本脚本不含任何算子或求解器的自有实现.

问题、网格、空间、材料的构建路径与 fa 完全相同 (``_common.fe_problem``); 分析器的积分阶
``degree + 3 = 4`` 与 fa 直接调用 ``LinearElasticIntegrator`` 的默认阶 ``p + 3`` 相同, 阶段 1 的 K_e
与 fa 阶段 1 是否逐位一致由 ``--verify-fa`` 在同一进程内用 ``np.array_equal`` 核对, 不靠代码同源推断.

内存口径 (CPU): 每个阶段先记 before = 当前 VmRSS, 再向 /proc/self/clear_refs 写 5 重置 VmHWM,
阶段结束读 VmHWM 作为该阶段的绝对峰值 peak, net = peak - before. 全程峰值 (process_max_rss)
= 各阶段峰值的最大值.

阶段划分:
  cache  面板: mesh (网格 + 空间 + 材料 + 分析器) -> cache (assemble_stiff_matrix: K_e + cell2dof)
  matvec 面板: mesh -> assemble (facade.assemble(): K_e + 体力右端 + Dirichlet 投影) -> warmup
               -> matvec (刚度算子乘 K x = operator.form @ x, 重复 repeats 次)
  solve  面板: mesh -> assemble -> setup_solve (对角 + 预条件子) -> solve (cg, 每步调用 facade @ x = (P_I K P_I + P_D) x)
  baseline 面板: 与网格无关, 单线程 (cases.toml 的 env 限制) memcpy 带宽与 dgemm 算力, 供阶段 2 换算占比
matvec / solve 面板不单列 cache 阶段: ``ElasticityEAOperator.assemble()`` 内部会再次调用
``assemble_stiff_matrix``, 单列会把 K_e 算两遍, 阶段 1 的数字以 cache 面板为准.

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
FLOPS_PER_CELL = 2 * LOCAL_DOFS * LOCAL_DOFS  # y_e = K_e x_e 的乘加次数
DEGREE = 1
DEVICE = "cpu"


def _element_data(facade: Any) -> tuple[np.ndarray, np.ndarray]:
    """从分析器持有的 const 积分子取出缓存的 K_e (NC, 12, 12) 与 cell2dof (NC, 12).

    ``assemble_stiff_matrix('ea')`` 把两者都放在 ``analyzer._const_integrator`` 里,
    ``assemble_operator_diagonal`` 也从这里复用; 本脚本只读不写.
    """
    const = getattr(facade.analyzer, "_const_integrator", None)
    if const is None:
        raise RuntimeError("K_e 尚未缓存: 需先调用 assemble_stiff_matrix() 或 assemble()")
    return np.asarray(const.value), np.asarray(const.to_gdof)


def _reference_kx(Ke: np.ndarray, cell2dof: np.ndarray, x: np.ndarray) -> np.ndarray:
    """核对用的纯 numpy 参考 y = sum_e G_e^T K_e G_e x (assembly-levels.md §2.3), 不计时."""
    y = np.zeros(x.shape[0], dtype=x.dtype)
    np.add.at(y, cell2dof.ravel(), np.einsum("cij,cj->ci", Ke, x[cell2dof]).ravel())
    return y


def _relerr(y: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_ref)) / max(float(np.max(np.abs(y_ref))), 1e-300))


def _build_facade(method: str, n: int) -> tuple[Dict[str, Any], StageMeter]:
    """构建网格 / 空间 / 材料与 ``ElasticityEAOperator`` 门面 (mesh 阶段), 不触发装配.

    Parameters
    ----------
    method : str
        单刚组装方式 (standard/voigt/fast), 透传给门面的 ``assembly_method``.
    n : int
        网格每方向段数.

    Returns
    -------
    ctx : dict
        含 mesh / vs / problem / material / facade / facts.
    meter : StageMeter
        已记录 mesh 阶段.
    """
    import_fe_stack_cpu()
    from soptx.fem.matrix_free import ElasticityEAOperator

    meter = StageMeter()
    with meter.stage("mesh"):
        problem, mesh, vs, material = build_problem_space(n)
        facade = ElasticityEAOperator(vs, problem, material, degree=DEGREE, assembly_method=method)

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
    """assemble 阶段: ``facade.assemble()`` 一次给出 K_e 缓存、体力右端与 Dirichlet 投影算子."""
    facade = ctx["facade"]
    with meter.stage("assemble"):
        operator, load = facade.assemble()
    Ke, cell2dof = _element_data(facade)
    ctx.update(
        {
            "operator": operator,  # DirichletBCOperator, 即 facade.system_operator
            "load": np.asarray(load),
            "Ke": Ke,
            "cell2dof": cell2dof,
            "is_bd": np.asarray(facade.boundary_dofs, dtype=bool),
        }
    )


def _finish(panel: str, ctx: Dict[str, Any], method: str, n: int, meter: StageMeter) -> Dict[str, Any]:
    """在全部阶段结束后组装公共字段 (网格事实、K_e 理论量、算子常驻、各阶段峰值 / 净增与单价)."""
    facts = ctx["facts"]
    Ndof = facts["Ndof"]
    Ke = ctx["Ke"]
    c2d = ctx["cell2dof"]
    ke_shape = [int(s) for s in Ke.shape]
    ke_theory = int(np.prod(ke_shape)) * 8
    persistent = int(Ke.nbytes + c2d.nbytes)

    def kb_per_dof(nbytes: float) -> float:
        return round(nbytes / Ndof / 1000, 2)

    out: Dict[str, Any] = {
        "panel": panel,
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": meter.memory_kind,
        "problem": PROBLEM_NAME,
        "mesh_type": MESH_TYPE,
        "operator_impl": "soptx.fem.matrix_free.ElasticityEAOperator",
        "method": method,
        "n": n,
        **facts,
        "Ke_shape": ke_shape,
        "Ke_theory_MiB": round(ke_theory / 2**20, 1),
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
                "cache_net_over_Ke_theory": round(meter.net_bytes("cache") / ke_theory, 2),
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
    """阶段 1 (panel cache): 只测 ``assemble_stiff_matrix('ea')`` 缓存 K_e 与 cell2dof (与 fa 阶段 1 同口径).

    常驻量与 fa ``measure_stage1`` 同一套动作: 阶段开始前先 ``gc`` 并归还建网格留下的 glibc 空闲页
    (否则它们被本阶段大块临时量占用后随 munmap 一起还给内核, 使"结束 RSS - 起点 RSS"不闭合),
    阶段结束后再取一次 RSS 增量, ``malloc_trim`` 前后各记一个值. EA 比 fa 多常驻一个 cell2dof.
    """
    import gc

    from _common.metrology import cur_rss_kib

    ctx, meter = _build_facade(method, n)
    facade = ctx["facade"]
    rss_before_trim_kib = cur_rss_kib()
    gc.collect()
    trimmed = _malloc_trim()
    with meter.stage("cache"):
        facade.analyzer.assemble_stiff_matrix()
    Ke, cell2dof = _element_data(facade)
    ctx.update({"Ke": Ke, "cell2dof": cell2dof})

    gc.collect()
    base_kib = meter.records["cache"].before_kib
    retained_kib = max(0, cur_rss_kib() - base_kib)
    if trimmed:
        _malloc_trim()
    after_kib = max(0, cur_rss_kib() - base_kib) if trimmed else retained_kib

    out = _finish("cache", ctx, method, n, meter)
    Ndof = out["Ndof"]
    out.update(
        {
            "cache_before_no_trim_MiB": round(rss_before_trim_kib / 1024, 1),
            "malloc_trim_supported": trimmed,
            "cache_retained_MiB": round(retained_kib / 1024, 1),
            "cache_retained_KB_per_dof": round(retained_kib * 1024 / Ndof / 1000, 2),
            "cache_retained_after_trim_MiB": round(after_kib / 1024, 1),
        }
    )
    return out


def measure_matvec_allocations(method: str, n: int) -> dict:
    """独立跟踪一次预热后算子乘的分配, 不测性能耗时.

    Parameters
    ----------
    method : str
        单刚组装方式.
    n : int
        网格每方向段数.

    Returns
    -------
    dict
        NumPy 跟踪校验, 可跟踪分配峰值及保留量, 探针 RSS.
        未接入 tracemalloc 的底层分配不在跟踪范围内.
    """
    import gc
    import tracemalloc

    from _common.metrology import cur_rss_kib, peak_rss_kib, reset_peak_rss

    if tracemalloc.is_tracing():
        raise RuntimeError("请在未启用 tracemalloc 的独立进程中运行探针")
    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    bform = ctx["operator"].form
    x = np.random.default_rng(0).standard_normal(ctx["facts"]["Ndof"])
    warmup = bform @ x
    del warmup
    gc.collect()

    # 已知大小的 NumPy 分配校验不计入算子乘.
    tracemalloc.start()
    try:
        check_before, _ = tracemalloc.get_traced_memory()
        check = np.empty(2**20, dtype=np.float64)
        check_current, _ = tracemalloc.get_traced_memory()
        expected = int(check.nbytes)
        observed = check_current - check_before
        check_ok = expected <= observed <= expected + 64 * 1024
        del check
    finally:
        tracemalloc.stop()
    if not check_ok:
        raise RuntimeError(f"NumPy 跟踪校验失败: 期望 {expected} B, 捕获 {observed} B")

    gc.collect()
    rss_before = cur_rss_kib()
    rss_reset = reset_peak_rss()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        y = bform @ x
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_after = cur_rss_kib()
    rss_peak = peak_rss_kib()
    return {
        "panel": "matvec_allocations",
        "method": method,
        "n": n,
        **ctx["facts"],
        "numpy_tracking_check_passed": check_ok,
        "numpy_tracking_expected_bytes": expected,
        "numpy_tracking_observed_bytes": observed,
        "allocation_peak_increment_bytes": peak - before,
        "allocation_retained_increment_bytes": current - before,
        "allocation_peak_minus_retained_bytes": peak - current,
        "output_bytes": int(y.nbytes),
        "probe_rss_before_MiB": rss_before / 1024,
        "probe_rss_after_MiB": rss_after / 1024,
        "probe_rss_peak_MiB": rss_peak / 1024 if rss_reset else None,
        "probe_rss_peak_increment_MiB": max(0, rss_peak - rss_before) / 1024 if rss_reset else None,
        "probe_rss_reset_supported": rss_reset,
        "allocation_scope": "一次预热后的 form @ x; 包含输出; 非累计分配量; 不含未接入跟踪的底层分配",
        "rss_scope": "探针 RSS 含跟踪器开销, 不用于无跟踪器性能结论",
    }


def _timed(fn: Any, repeats: int) -> tuple[list[float], Any]:
    times: list[float] = []
    y = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = fn()
        times.append(time.perf_counter() - t0)
    return times, y


def measure_matvec(method: str, n: int, repeats: int = 20, seed: int = 0) -> dict:
    """阶段 2 (panel matvec): 预热后重复 ``repeats`` 次刚度算子乘 K x, 记录中位 / 最小耗时与有效带宽下界.

    计时对象是 ``operator.form @ x``, 即 ``soptx.fem.BilinearForm.__matmul__``: gather ``x[cell2dof]`` ->
    ``einsum("cij, cj -> ci", K_e, x_e)`` -> ``index_add`` scatter-add, 对应 assembly-levels.md §2.3 的
    EA MatVec, 与 fa 的 CSR ``K @ x`` 同口径. 含 Dirichlet 投影的系统算子乘 ``facade @ x`` 只在阶段 3
    由 cg 调用, 不在本阶段单独计时. 结果与纯 numpy 参考实现核对 (相对误差).

    带宽下界按每次算子乘至少搬运 K_e + cell2dof + 读 x 写 y (16 B/dof) 计, 不含 gather 与
    scatter 的随机访问放大, 因此是真实带宽的下界.
    """
    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    bform = ctx["operator"].form
    Ke, cell2dof = ctx["Ke"], ctx["cell2dof"]
    Ndof = ctx["facts"]["Ndof"]
    NC = ctx["facts"]["NC"]

    x = np.random.default_rng(seed).standard_normal(Ndof)

    with meter.stage("warmup"):
        y = bform @ x

    with meter.stage("matvec"):
        times, y = _timed(lambda: bform @ x, repeats)

    y = np.asarray(y)
    y_ref = _reference_kx(Ke, cell2dof, x)

    t_med = statistics.median(times)
    bytes_moved_min = Ke.nbytes + cell2dof.nbytes + 16 * Ndof

    out = _finish("matvec", ctx, method, n, meter)
    out.update(
        {
            "repeats": repeats,
            "seed": seed,
            "matvec_impl": "soptx.fem.BilinearForm.__matmul__ (inherited from fealpy): gather -> einsum -> index_add",
            "matvec_seconds_median": round(t_med, 6),
            "matvec_seconds_min": round(min(times), 6),
            "matvec_seconds_all": [round(t, 6) for t in times],
            "bytes_moved_min_per_matvec": int(bytes_moved_min),
            "effective_gbps_lower_bound": round(bytes_moved_min / t_med / 1e9, 2),
            "gflops": round(FLOPS_PER_CELL * NC / t_med / 1e9, 2),
            "y_norm": float(np.linalg.norm(y)),
            "matvec_vs_reference_relerr": _relerr(y, y_ref),
        }
    )
    return out


def measure_solve(method: str, n: int, maxiter: int = 5000, tol: float = 1e-6) -> dict:
    """阶段 3 (panel solve): 核心 Jacobi-PCG (``soptx.solvers.cg`` + ``DiagonalPreconditioner``) 求解制造解问题.

    系统 A = Pi_I K Pi_I + Pi_D 与右端来自 ``facade.assemble()`` (体力 + Dirichlet 消去), 初值取
    ``prescribed_solution`` (边界为给定位移、内部为零), 对角由 ``assemble_operator_diagonal`` 给出
    (Dirichlet 处为 1). 停机判据为 cg 的 'natural' 口径 ||r_k||_{M^-1} <= tol ||r_0||_{M^-1};
    另报告求解后的真残差 ||b - A x|| / ||b||.
    """
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
            "solver_impl": "soptx.solvers.cg + DiagonalPreconditioner",
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


def measure_continuous(method: str, n: int, repeats: int = 20) -> dict:
    """连续测量面板: 同一进程内测量 cache -> input -> first_matvec -> repeat_matvec.

    用于观测工作区缓冲的初次物化净增 (首次算子乘) 以及稳态重复调用的零内存增长,
    同时提供跨阶段连续水位演进数据.
    """
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
    trim_supported = _malloc_trim()
    with stage("cache"):
        operator = ctx["facade"].analyzer.assemble_stiff_matrix()
    with stage("input"):
        x = rng.standard_normal(ctx["facts"]["Ndof"])
    with stage("first_matvec"):
        y = operator @ x
    with stage("repeat_matvec"):
        for i in range(repeats):
            start = time.perf_counter()
            y = operator @ x
            times[i] = time.perf_counter() - start

    # 正确性核对
    Ke = np.asarray(operator.element_matrices)
    c2d = np.asarray(operator.const_integrator.to_gdof)
    ref = _reference_kx(Ke, c2d, x)
    error = _relerr(y, ref)
    med = statistics.median(times)
    peak = max(mesh_meter.max_peak_kib(), *(r["peak_kib"] for r in stages.values()))
    facts = ctx["facts"]
    persistent_bytes = int(Ke.nbytes + c2d.nbytes)
    result = {
        "panel": "continuous",
        "n": n,
        **facts,
        "method": method,
        "operator_impl": type(operator).__module__ + "." + type(operator).__name__,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "同一进程: cache -> input -> first_matvec -> repeat_matvec; 无右端与边界处理; 阶段间不额外 gc/trim; 保留输入与输出",
        "trim_before_cache_supported": trim_supported,
        "stages": stages,
        "mesh_fields": mesh_meter.fields(),
        "process_peak_kib": peak,
        "operator_persistent_bytes": persistent_bytes,
        "matvec_seconds_median": med,
        "matvec_seconds_all": times,
        "effective_gbps_lower_bound": (persistent_bytes + 16 * facts["Ndof"]) / med / 1e9,
        "gflops": 288 * facts["NC"] / med / 1e9,
        "reference_relerr": error,
    }
    if error > 1e-12 or not all(r["reset_supported"] for r in stages.values()):
        raise RuntimeError("连续测量核对失败, 请检查原始记录")
    return result



# -----------------------------------------------------------------------------
# 1b. 与 fa 的逐位一致性核对
# -----------------------------------------------------------------------------

def verify_against_fa(n: int, methods: Sequence[str]) -> int:
    """逐位核对核心路径缓存的 K_e / cell2dof 与 fa 构建路径生成的是否完全相同.

    同一进程内用 ``fa_assembly_capability/run.py`` 的 ``_build_problem_space`` 建问题并直接调用
    ``LinearElasticIntegrator(material, method).assembly(vs)`` (fa 阶段 1 的做法), 再用本目录的
    ``ElasticityEAOperator(...).analyzer.assemble_stiff_matrix()`` 取 const 积分子缓存的 K_e 与
    cell2dof, 用 ``np.array_equal`` 逐位比较 (形状、dtype、数值). 只在 CPU 上核对, 不落盘.

    Parameters
    ----------
    n : int
        网格每方向段数, 默认 8 即可 (秒级).
    methods : sequence of str
        要核对的单刚组装方式.

    Returns
    -------
    int
        全部一致返回 0, 任一不一致返回 1.
    """
    import importlib.util

    fa_path = config.REPOSITORY_ROOT / "experiments" / "fa_assembly_capability" / "run.py"
    spec = importlib.util.spec_from_file_location("_fa_run", fa_path)
    fa_run = importlib.util.module_from_spec(spec)
    sys.modules["_fa_run"] = fa_run  # dataclass 装饰器要求模块已注册
    spec.loader.exec_module(fa_run)

    import_fe_stack_cpu()
    from soptx.fem.integrators import LinearElasticIntegrator
    from soptx.fem.matrix_free import ElasticityEAOperator

    _, _, vs_fa, mat_fa = fa_run._build_problem_space(n)
    problem, _, vs_ea, mat_ea = build_problem_space(n)

    def same_array(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)

    ok = True
    c_fa = np.asarray(vs_fa.cell_to_dof())
    for m in methods:
        k_fa = np.asarray(LinearElasticIntegrator(mat_fa, method=m).assembly(vs_fa))
        facade = ElasticityEAOperator(vs_ea, problem, mat_ea, degree=DEGREE, assembly_method=m)
        facade.analyzer.assemble_stiff_matrix()
        k_ea, c_ea = _element_data(facade)

        same_c = same_array(c_fa, c_ea)
        same_k = same_array(k_fa, k_ea)
        ok &= same_c and same_k
        if same_k:
            verdict = "identical (bitwise)"
        elif k_fa.shape == k_ea.shape:
            verdict = f"DIFFER, max|diff| = {float(np.max(np.abs(k_fa - k_ea))):.3e}"
        else:
            verdict = f"DIFFER, shape {k_fa.shape} vs {k_ea.shape}"
        print(
            f"n={n} method={m:<8} K_e {k_fa.shape} {k_fa.dtype}: {verdict} | "
            f"cell2dof {c_fa.shape} {c_fa.dtype}: {'identical' if same_c else 'DIFFER'}"
        )

    print("RESULT:", "ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


# -----------------------------------------------------------------------------
# 2. 控制台树状卡片看板
# -----------------------------------------------------------------------------

def _fmt_mib(mib: float | None) -> str:
    if mib is None:
        return "--"
    if mib >= 1024:
        return f"{mib / 1024:.2f} GiB ({mib:,.1f} MiB)"
    return f"{mib:,.1f} MiB"


def _fmt_s(t: float | None) -> str:
    if t is None:
        return "--"
    return f"{t * 1000:.1f} ms" if t < 1.0 else f"{t:.2f} s"


def _stage_line(out: dict, name: str, label: str, unit_key: str | None = None, unit: str = "KB/dof") -> str:
    peak = out.get(f"{name}_peak_MiB")
    net = out.get(f"{name}_net_MiB")
    text = f"{label:<16}: peak {_fmt_mib(peak)} | net {_fmt_mib(net)}"
    if unit_key and out.get(unit_key) is not None:
        text += f" | {out[unit_key]:.2f} {unit}"
    t = out.get(f"t_{name}_s")
    if t is not None:
        text += f" | {_fmt_s(t)}"
    return text


def print_dashboard(out: dict[str, Any]) -> None:
    """打印树状卡片式实测摘要 (每阶段绝对峰值 / 净增)."""
    n = out.get("n", 0)
    nc = out.get("NC", 0)
    ndof = out.get("Ndof", 0)
    mesh_line = f"{MESH_TYPE} (grid = {n}^3) | {nc:,} cells | {ndof:,} DOFs"
    panel = out.get("panel")
    titles = {"cache": "element-cache", "matvec": "ea-matvec", "solve": "ea-cg-solve"}
    rep = out.get("repeats", 0)

    print(f"\n● [{titles.get(panel, panel)}] {PROBLEM_NAME}")
    print(f"  ├── Mesh & DOFs   : {mesh_line}")
    print(
        f"  ├── Operator      : {out.get('operator_impl')} | method = {out.get('method')} | "
        f"K_e theory = {out.get('Ke_theory_MiB', 0):,.1f} MiB | "
        f"persistent = {_fmt_mib(out.get('operator_persistent_MiB'))} "
        f"({out.get('operator_persistent_KB_per_dof', 0):.2f} KB/dof)"
    )
    print(f"  ├── {_stage_line(out, 'mesh', 'Mesh & Space')}")
    if panel == "cache":
        print(f"  ├── {_stage_line(out, 'cache', 'Cache (K_e)', 'cache_KB_per_dof')}")
    else:
        print(f"  ├── {_stage_line(out, 'assemble', 'Assemble (bc)')}")
    if panel == "matvec":
        print(f"  ├── {_stage_line(out, 'warmup', 'Warmup')}")
        print(f"  ├── {_stage_line(out, 'matvec', f'K x (form@x) x{rep}')}")
        print(
            f"  ├── Per K x       : median {_fmt_s(out.get('matvec_seconds_median'))} | "
            f"min {_fmt_s(out.get('matvec_seconds_min'))} | "
            f"eff >= {out.get('effective_gbps_lower_bound', 0):.2f} GB/s | "
            f"{out.get('gflops', 0):.2f} GFLOP/s | relerr {out.get('matvec_vs_reference_relerr', 0):.1e}"
        )
    elif panel == "solve":
        print(f"  ├── {_stage_line(out, 'setup_solve', 'Solve setup')}")
        print(f"  ├── {_stage_line(out, 'solve', 'Jacobi-PCG')}")
        status = "converged" if out.get("converged") else f"NOT converged ({out.get('reason')})"
        print(
            f"  ├── Iterations    : {out.get('iterations', 0):,} ({status}, relres {out.get('final_relres', 0):.2e}, "
            f"true {out.get('true_relres', 0):.2e}) | "
            f"{out.get('iterations_per_n', 0):.2f} it/n | {_fmt_s(out.get('seconds_per_iteration'))} per it"
        )
    print(
        f"  └── Absolute Peak : {_fmt_mib(out.get('process_max_rss_MiB'))} "
        f"({out.get('process_max_rss_KB_per_dof', 0):.2f} KB/dof)\n"
    )


# -----------------------------------------------------------------------------
# 3. 调度: Case -> 子进程执行计划
# -----------------------------------------------------------------------------

LIST_COLUMNS = (
    ("panel", "panel", "-"),
    ("mesh", "mesh_type", MESH_TYPE),
    ("grid", "grid", "-"),
    ("problem", "problem", PROBLEM_NAME),
    ("method", "method", "fast"),
)


def resolve_runs(
    cases: tuple[config.Case, ...],
    is_all: bool,
    overrides: dict[str, Any],
) -> list[scheduler.Run]:
    """将选中的 Case 与参数覆盖解析为具体的单次子进程执行计划.

    三个面板默认只跑 case 的 method (fast); 仅显式 --method all 时展开 METHOD_NAMES, --all 不展开 (is_all 仅保留签名).
    """
    runs: list[scheduler.Run] = []
    for case in cases:
        if case.panel == "baseline":
            out_p = config.OUTPUT_DIR / case.artifact
            argv = [sys.executable, str(case.script_path), "--worker", "--baseline", "--output", str(out_p)]
            runs.append((case.id, argv, out_p, case.summary, case.subprocess_env()))
            continue
        n = overrides.get("n", case.extra.get("n", 32))
        env = case.subprocess_env()
        common = [sys.executable, str(case.script_path), "--worker"]
        tail = ["--n", str(n)]
        methods = scheduler.expand(
            overrides.get("method"),
            False,
            list(METHOD_NAMES),
            case.extra.get("method", "fast"),
        )

        for m in methods:
            if case.panel == "cache":
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("cache", [m], n, DEVICE)
                argv = [*common, "--cache", "--method", m, *tail, "--output", str(out_p)]
                detail = f"method={m}, n={n}"
            elif case.panel == "matvec":
                repeats = overrides.get("repeats", case.extra.get("repeats", 20))
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("matvec", [m], n, DEVICE)
                argv = [*common, "--matvec", "--method", m, *tail, "--repeats", str(repeats), "--output", str(out_p)]
                detail = f"method={m}, n={n}, repeats={repeats}"
                if overrides.get("probe_allocations"):
                    out_p = config.OUTPUT_DIR / scheduler.artifact_name("matvec_allocations", [m], n, DEVICE)
                    argv = [*common, "--matvec", "--probe-allocations", "--method", m, *tail, "--output", str(out_p)]
                    detail = f"method={m}, n={n}, allocation probe"
            elif case.panel == "continuous":
                repeats = overrides.get("repeats", case.extra.get("repeats", 20))
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("cache_matvec_continuous", [m], n, DEVICE)
                argv = [*common, "--continuous", "--method", m, *tail, "--repeats", str(repeats), "--output", str(out_p)]
                detail = f"method={m}, n={n}, repeats={repeats}"
            else:  # solve
                maxiter = overrides.get("maxiter", case.extra.get("maxiter", 5000))
                tol = overrides.get("tol", case.extra.get("tol", 1e-6))
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("solve", [m], n, DEVICE)
                argv = [
                    *common, "--solve", "--method", m, *tail,
                    "--maxiter", str(maxiter), "--tol", str(tol), "--output", str(out_p),
                ]
                detail = f"method={m}, n={n}, maxiter={maxiter}, tol={tol}"
            label = f"{case.id} [{m}]"
            runs.append((label, argv, out_p, f"{case.summary} ({detail})", env))
    return runs


# -----------------------------------------------------------------------------
# 4. 主入口与参数路由
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """调度与测量的主入口函数.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数列表, 缺省使用 sys.argv[1:].

    Returns
    -------
    int
        程序退出状态码.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="ea_assembly_capability 实验图面驱动: 测量核心 EA 算子的内存开销、算子乘耗时与 Jacobi-PCG 迭代",
    )

    # 1. 调度与工况选择参数
    parser.add_argument("--list", action="store_true", help="列出已注册数据点")
    parser.add_argument("--all", action="store_true", help="跑全部工况 (每个 case 一条, method 取 cases.toml 的 fast)")
    parser.add_argument("--cases", nargs="+", help="指定要跑的一个或多个 case id (如 --cases element-cache)")
    parser.add_argument("--case", help="指定单个 case id (等价于 --cases <id>)")
    parser.add_argument("--panel", choices=config.PANELS, help="只跑指定面板 (cache/matvec/solve/baseline) 数据点")
    parser.add_argument("--check-only", action="store_true", help="只打印将执行的子进程命令")
    parser.add_argument("--skip-existing", action="store_true", help="产物已存在时跳过")
    parser.add_argument("--monitor", action="store_true", help="运行时实时显示独立 Worker 的 CPU 与内存占用")
    parser.add_argument(
        "--monitor-interval", type=float, default=0.5, metavar="SECONDS",
        help="实时监控刷新间隔, 单位为秒 (默认 0.5)",
    )

    # 2. 工况动态覆盖参数 (Overrides)
    parser.add_argument("-n", "--n", "--grid", dest="n", type=int, default=None, help="动态覆盖网格剖分段数 (如 -n 32 或 --grid 32)")
    parser.add_argument("--method", choices=METHOD_NAMES + ("all",), default=None, help="指定或覆盖单刚算法")
    parser.add_argument("--repeats", type=int, default=None, help="matvec: 计时重复次数 (默认 20)")
    parser.add_argument("--maxiter", type=int, default=None, help="solve: PCG 最大迭代数 (默认 5000)")
    parser.add_argument("--tol", type=float, default=None, help="solve: 相对残差收敛阈值 (默认 1e-6)")

    # 3. Worker 测量层底层参数 (供子进程调用)
    parser.add_argument("--worker", action="store_true", help="进入子进程 worker 测量模式")
    parser.add_argument("--cache", action="store_true", help="cache 面板: K_e 与 cell2dof 缓存")
    parser.add_argument("--matvec", action="store_true", help="matvec 面板: 核心 EA 刚度算子乘 K x 计时")
    parser.add_argument("--solve", action="store_true", help="solve 面板: 核心 Jacobi-PCG 求解")
    parser.add_argument("--continuous", action="store_true", help="continuous 面板: 同一进程连续测量 cache -> input -> first_matvec -> repeat_matvec")
    parser.add_argument("--baseline", action="store_true", help="baseline 面板: 单核 memcpy 带宽与 dgemm 算力")
    parser.add_argument("--output", type=Path, default=None, help="产物落盘路径")

    # 4. 一致性核对 (进程内, 不落盘)
    parser.add_argument(
        "--verify-fa", action="store_true",
        help="逐位核对核心路径缓存的 K_e / cell2dof 与 fa_assembly_capability 的构建路径一致 (默认 --n 8, --method all)",
    )

    parser.add_argument("--probe-allocations", action="store_true", help="matvec: 独立测一次预热后的分配峰值, 不计时, 单独落盘")
    args = parser.parse_args(argv)
    if args.probe_allocations and (args.cache or args.solve or args.baseline or args.verify_fa or args.continuous):
        parser.error("--probe-allocations 仅适用于 matvec")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval 必须大于 0")

    # ------------------------------------------------ 与 fa 的逐位核对
    if args.verify_fa:
        methods = list(METHOD_NAMES) if args.method in (None, "all") else [args.method]
        return verify_against_fa(args.n if args.n is not None else 8, methods)

    # ------------------------------------------------ Worker 测量分支
    if args.baseline:
        out = measure_baseline()
        print_baseline(out)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.worker or args.cache or args.matvec or args.solve or args.continuous:
        if args.n is None:
            parser.error("Worker 模式必须指定 --n")
        if args.method == "all":
            parser.error("Worker 模式不接受 --method all, 由调度层展开")
        method = args.method or "fast"
        if args.continuous:
            out = measure_continuous(method, args.n, repeats=args.repeats or 20)
        elif args.matvec:
            out = (measure_matvec_allocations(method, args.n) if args.probe_allocations
                   else measure_matvec(method, args.n, repeats=args.repeats or 20))
        elif args.solve:
            out = measure_solve(
                method, args.n,
                maxiter=args.maxiter or 5000,
                tol=args.tol if args.tol is not None else 1e-6,
            )
        elif args.cache:
            out = measure_cache(method, args.n)
        else:
            parser.error("Worker 模式需指定 --cache, --matvec, --solve 或 --continuous")

        if args.continuous:
            print(f"EA 连续测量面板 (cache -> input -> first_matvec -> repeat_matvec x {args.repeats or 20})")
            for stage_name, sinfo in out["stages"].items():
                print(f"  [{stage_name}] before: {sinfo['before_kib']/1024:.1f} MiB | peak: {sinfo['peak_kib']/1024:.1f} MiB | net: {sinfo['net_kib']/1024:.1f} MiB | time: {sinfo['t_s']:.3f} s")
            print(f"  稳态耗时中位数: {out['matvec_seconds_median']*1000:.2f} ms | 有效带宽下界: {out['effective_gbps_lower_bound']:.2f} GB/s")
        elif args.probe_allocations:
            print("EA 分配探针 (不计时, NumPy 跟踪校验通过)")
            for key in ("allocation_peak_increment_bytes", "allocation_retained_increment_bytes",
                        "allocation_peak_minus_retained_bytes", "output_bytes"):
                print(f"  {key}: {out[key] / 2**20:.3f} MiB")
            print(f"  RSS peak: {out['probe_rss_peak_MiB']} MiB | increment: {out['probe_rss_peak_increment_MiB']} MiB")
        else:
            print_dashboard(out)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # ------------------------------------------------ 调度与执行分支
    try:
        figure, cases = config.load_cases()
    except config.ConfigError as error:
        print(f"cases.toml 有误: {error}", file=sys.stderr)
        return 2

    if args.list:
        return scheduler.print_case_table(cases, LIST_COLUMNS)

    target_case_ids: list[str] = []
    panel_filter: str | None = args.panel
    known_ids = {c.id for c in cases}
    is_all = args.all

    alias_map = {
        "elem-cache": "element-cache",
        "cg": "ea-cg-solve",
        "cg-solve": "ea-cg-solve",
        "continuous": "ea-continuous",
    }

    def resolve_case_id(name: str) -> str:
        if name in known_ids:
            return name
        return alias_map.get(name.lower(), name)

    raw_cases: list[str] = []
    if args.cases:
        raw_cases.extend(args.cases)
    if args.case:
        raw_cases.append(args.case)

    for item in raw_cases:
        item_lower = item.lower()
        if item_lower == "all":
            is_all = True
        elif item_lower in config.PANELS:
            panel_filter = item_lower
        else:
            target_case_ids.append(resolve_case_id(item))

    if not (is_all or target_case_ids or panel_filter):
        print(
            "错误: 必须通过 --case/--cases/--panel/--all 指定要运行的工况或面板。\n"
            "  常用示例:\n"
            "    python run.py --case element-cache --grid 32 --monitor\n"
            "    python run.py --case ea-matvec --grid 32 --repeats 20\n"
            "    python run.py --case cpu-baseline --monitor\n"
            "    python run.py --all --check-only\n"
            "  查看全部工况列表请使用: python run.py --list",
            file=sys.stderr,
        )
        return 2

    try:
        selected = config.select(
            cases,
            case_ids=target_case_ids if target_case_ids else None,
            panel=panel_filter,
        )
    except config.ConfigError as error:
        print(str(error), file=sys.stderr)
        return 2

    if args.probe_allocations and any(case.panel != "matvec" for case in selected):
        parser.error("--probe-allocations 只能选择 matvec 工况")
    overrides: dict[str, Any] = {"probe_allocations": args.probe_allocations}
    if args.n is not None:
        overrides["n"] = args.n
    if args.method is not None:
        overrides["method"] = args.method
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    if args.maxiter is not None:
        overrides["maxiter"] = args.maxiter
    if args.tol is not None:
        overrides["tol"] = args.tol

    runs = resolve_runs(selected, is_all=is_all, overrides=overrides)
    failed = scheduler.command_run(
        runs,
        cwd=config.REPOSITORY_ROOT,
        check_only=args.check_only,
        skip_existing=args.skip_existing,
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
