"""线程级并行扩展性 benchmark: 父进程派发, 子进程定死线程数后测量.

OpenBLAS 与 MKL 在 ``import numpy`` / ``import torch`` 的那一刻就建好线程池,
进程内再改 ``os.environ`` 对它们无效, ``torch.set_num_threads`` 也只管 ATen
自己那一个池. 因此本脚本分两层运行::

    父进程 (driver)  解析线程档位 -> 拼 env -> 起子进程 -> 收 JSON -> 出表
      +- 子进程 (worker)  env 已定死 -> 才 import -> 跑一档 -> 打 JSON

被测段分两类, 缺一不可: 只测计算受限段时, 曲线压不上去无法区分"代码没并行"
和"撞内存墙". 带宽受限段是对照, 它应当在很少的线程数上就饱和.

正确性优先于时间: 线程数只改归约顺序, 不改数学解. 每一档都要和单线程档比
指纹, 差异超过 ``--tol`` 即判失败, 这比性能数字难看严重得多.

运行::

    python examples/parallel_execution/benchmark_thread_scaling.py --list
    python examples/parallel_execution/benchmark_thread_scaling.py --threads 1,2,4,8
    python examples/parallel_execution/benchmark_thread_scaling.py \
        --threads 1,4,16 --segments assemble,filter_spmv --n 128 --json out.json

可选依赖 ``threadpoolctl``: 装了才能核实每档线程数是否真的生效, 未装时脚本
照跑, 但环境记录里标 ``verified: false``, 不得把结果当作核实过的数据引用.
"""

from __future__ import annotations

# 本模块顶层严禁 import numpy / torch / fealpy: 父进程也会加载本文件, 而
# 线程池必须在子进程的 env 生效之后才建立. 重型 import 一律放到函数内部.
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 必须同时钉死的线程环境变量. 只设其中一个会测到混合状态.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: 子进程用于分隔 JSON 的标记. FEALPy 与后端库会往 stdout 打日志, 直接
#: json.loads 整个 stdout 会失败.
JSON_BEGIN = "__THREAD_SCALING_JSON_BEGIN__"
JSON_END = "__THREAD_SCALING_JSON_END__"

DEFAULT_THREADS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# 段的注册
# ---------------------------------------------------------------------------

@dataclass
class SegmentSpec:
    """一个被测段的元信息.

    Parameters
    ----------
    kind : str
        ``'compute'`` 计算受限, 预期随线程数上涨; ``'bandwidth'`` 带宽受限,
        预期在少数线程上饱和. 这一栏是判读曲线的先验, 不是测量结果.
    desc : str
        中文一句话说明.
    fn : callable or None
        ``fn(ctx) -> dict``, 返回 ``{'seconds': float, 'signature': dict}``.
        ``None`` 表示尚未实现, 选中时给出明确报错而不是静默跳过.
    """

    kind: str
    desc: str
    fn: Optional[Callable[["Workload"], dict[str, Any]]] = None


def _register() -> dict[str, SegmentSpec]:
    """建立段注册表. 放在函数里是为了让引用的实现函数先定义完."""
    return {
        "assemble": SegmentSpec(
            kind="compute",
            desc="单元刚度装配 (单元积分 + einsum + 组装)",
            fn=_seg_assemble,
        ),
        "cg_solve": SegmentSpec(
            kind="compute",
            desc="CG 求解 (算子作用 x 迭代数, 最大杠杆)",
            fn=_seg_cg_solve,
        ),
        "filter_spmv": SegmentSpec(
            kind="bandwidth",
            desc="密度滤波卷积 (spmv + 两次 elementwise), 带宽受限对照",
            fn=_seg_filter_spmv,
        ),
        "simp_update": SegmentSpec(
            kind="bandwidth",
            desc="SIMP 插值 + OC 更新 (纯 elementwise), 带宽受限对照",
            fn=None,
        ),
        "direct_factor": SegmentSpec(
            kind="compute",
            desc="直接法数值分解 (MUMPS / PARDISO 的 BLAS3 段)",
            fn=None,
        ),
    }


# ---------------------------------------------------------------------------
# 工况: 只建一次, 不计入计时
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    """被测段共享的工况对象. 构造开销不计入任何段的计时."""

    args: argparse.Namespace
    analyzer: Any = None
    filter_obj: Any = None
    n_cells: int = 0
    n_dofs: int = 0
    cache: dict[str, Any] = field(default_factory=dict)


def build_workload(args: argparse.Namespace) -> Workload:
    """按命令行参数搭好网格、分析器与滤波器.

    Parameters
    ----------
    args : argparse.Namespace
        已解析的命令行参数.

    Returns
    -------
    Workload
        供各段复用的工况对象.
    """
    from fealpy.backend import backend_manager as bm
    from fealpy.mesh import QuadrangleMesh, TriangleMesh

    from soptx.fem.analyzers import LagrangeFEMAnalyzer
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems import SinusoidalPlaneStrainElasticity2D
    from soptx.topology.filters import Filter

    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device("cpu")

    if args.dim != 2:
        raise NotImplementedError("当前框架只搭了 2D 工况, 3D 待接.")

    problem = SinusoidalPlaneStrainElasticity2D()
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, plane_type="plane_strain",
    )
    constructor = {"tri": TriangleMesh, "quad": QuadrangleMesh}[args.mesh_type]
    mesh = constructor.from_box(list(problem.domain), nx=args.n, ny=args.n)

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=args.order,
        integration_order=args.order + 3,
        operator_level=args.operator_level,
        solve_method="cg",
        topopt_algorithm=None,
        enable_logging=False,
    )

    # 滤波器只在选到带宽受限段时才需要, 但构造很便宜, 一并建好.
    filter_obj = Filter(
        design_mesh=mesh,
        filter_type="density",
        rmin=args.rmin,
        density_location="element",
        enable_logging=False,
    )

    return Workload(
        args=args,
        analyzer=analyzer,
        filter_obj=filter_obj,
        n_cells=int(mesh.number_of_cells()),
        n_dofs=int(analyzer.tensor_space.number_of_global_dofs()),
    )


# ---------------------------------------------------------------------------
# 段的实现
# ---------------------------------------------------------------------------

def _seg_assemble(ctx: Workload) -> dict[str, Any]:
    """装配单元刚度矩阵. 计算受限."""
    from fealpy.backend import backend_manager as bm

    t0 = time.perf_counter()
    K = ctx.analyzer.assemble_stiff_matrix()
    seconds = time.perf_counter() - t0

    # 指纹取矩阵值的范数: 线程数不同只应改变归约顺序, 不应改变这个数.
    values = K.values() if hasattr(K, "values") else K.data
    return {
        "seconds": seconds,
        "signature": {"K_norm": float(bm.linalg.norm(bm.asarray(values)))},
    }


def _seg_cg_solve(ctx: Workload) -> dict[str, Any]:
    """CG 求解. 计算受限, 是算子作用乘以迭代数的总账."""
    from fealpy.backend import backend_manager as bm

    # 装配与边界条件不计入本段计时, 但结果要缓存, 免得每次 repeat 重装.
    if "system" not in ctx.cache:
        K0 = ctx.analyzer.assemble_stiff_matrix()
        F0 = ctx.analyzer.assemble_body_force_vector()
        ctx.cache["system"] = ctx.analyzer.apply_bc(K0, F0)
    K, F = ctx.cache["system"]

    uh = ctx.analyzer.tensor_space.function()
    t0 = time.perf_counter()
    _, info = ctx.analyzer.solve_system(K, F, uh, rtol=1e-12, atol=1e-12, maxiter=5000)
    seconds = time.perf_counter() - t0

    return {
        "seconds": seconds,
        "signature": {
            "u_norm": float(bm.linalg.norm(bm.asarray(uh))),
            "niter": int(info.get("niter", -1)),
        },
    }


def _seg_filter_spmv(ctx: Workload) -> dict[str, Any]:
    """密度滤波的卷积段. 带宽受限对照.

    走 ``filter_objective_sensitivities`` 而非 ``filter_design_variable``:
    前者收发都是裸张量, 后者要求传入 ``Function``, 对本段没有必要.
    """
    from fealpy.backend import backend_manager as bm

    if "filter_input" not in ctx.cache:
        rho = bm.ones((ctx.n_cells,), dtype=bm.float64) * 0.5
        grad = bm.linspace(0.0, 1.0, ctx.n_cells, dtype=bm.float64)
        ctx.cache["filter_input"] = (rho, grad)
    rho, grad = ctx.cache["filter_input"]

    t0 = time.perf_counter()
    out = ctx.filter_obj.filter_objective_sensitivities(
        design_variable=rho, obj_grad_rho=grad,
    )
    seconds = time.perf_counter() - t0

    return {
        "seconds": seconds,
        "signature": {"out_norm": float(bm.linalg.norm(bm.asarray(out)))},
    }


# ---------------------------------------------------------------------------
# 环境探测
# ---------------------------------------------------------------------------

def probe_environment(requested: int) -> dict[str, Any]:
    """记录本次子进程实际处于什么线程配置下.

    Parameters
    ----------
    requested : int
        本档请求的线程数.

    Returns
    -------
    dict
        含 ``env``、``pools``、``verified``、``trustworthy``、``caveats``.

    Notes
    -----
    ``verified`` 只在 ``threadpoolctl`` 可用时为真. 未装时各库的线程数是
    "声明值"而非"核实值"; scipy 自带的 OpenBLAS wheel 符号被改名,
    ``openblas_get_num_threads`` 取不到, 没有别的办法核实.
    """
    info: dict[str, Any] = {
        "requested_threads": requested,
        "env": {var: os.environ.get(var) for var in THREAD_ENV_VARS},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os_cpu_count": os.cpu_count(),
    }

    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity_cpus"] = None

    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        info["torch_threads"] = None

    pools, verified = _probe_pools()
    info["pools"] = pools
    info["verified"] = verified

    trustworthy, caveats = _assess_trust(info)
    info["trustworthy"] = trustworthy
    info["caveats"] = caveats
    return info


def _probe_pools() -> tuple[Optional[list[dict[str, Any]]], bool]:
    """用 threadpoolctl 列出已加载 native 库的真实线程数."""
    try:
        import threadpoolctl
    except ImportError:
        return None, False

    pools = [
        {
            "user_api": p.get("user_api"),
            "internal_api": p.get("internal_api"),
            "num_threads": p.get("num_threads"),
            "filepath": p.get("filepath"),
        }
        for p in threadpoolctl.threadpool_info()
    ]
    return pools, True


def _assess_trust(info: dict[str, Any]) -> tuple[bool, list[str]]:
    """判断本机是否具备可信的线程扩展性测量条件.

    Returns
    -------
    (bool, list of str)
        是否可信, 以及不可信的具体理由.

    Notes
    -----
    这一项存在的理由: 不可信的曲线一旦落盘, 迟早会被当成可信数据引用.
    宁可让脚本自己在产物里写死 ``trustworthy: false``.
    """
    caveats: list[str] = []

    try:
        version = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        version = ""
    if "microsoft" in version.lower():
        caveats.append(
            "运行在 WSL2 下: 暴露的是虚拟化后的同构拓扑, 核绑定无效, "
            "vCPU 到物理核的映射由 Windows 动态决定"
        )

    if not info.get("verified"):
        caveats.append(
            "未安装 threadpoolctl: 各库线程数为声明值而非核实值, "
            "不能断言请求的线程数已生效"
        )

    if info.get("affinity_cpus") is None:
        caveats.append("当前平台无 sched_getaffinity, 无法确认可用核集合")

    hybrid = _detect_hybrid_topology()
    if hybrid:
        caveats.append(hybrid)

    return (not caveats), caveats


def _detect_hybrid_topology() -> Optional[str]:
    """粗判 CPU 是否为大小核混合架构.

    Returns
    -------
    str or None
        命中时给出中文说明, 否则 ``None``.

    Notes
    -----
    只按 ``/proc/cpuinfo`` 的 model name 做字符串匹配, 属于启发式: 在 WSL
    下拓扑本来就是假的, 这里只求给出提示, 不求准确分类.
    """
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None

    model = ""
    for line in text.splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[-1].strip()
            break

    hybrid_marks = ("i9-12", "i9-13", "i9-14", "i7-12", "i7-13", "i7-14",
                    "i5-12", "i5-13", "i5-14", "Ultra")
    if any(mark in model for mark in hybrid_marks):
        return (
            f"CPU '{model}' 为 P-core/E-core 混合架构: 不绑核时线程会落在"
            f"性能不同的核上, 加速比没有单一分母"
        )
    return None


# ---------------------------------------------------------------------------
# 子进程 (worker)
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> int:
    """在已定死线程数的进程里跑完一档, 把 JSON 打到 stdout."""
    requested = args.worker
    segments = _resolve_segments(args.segments)

    # 兜底: env 只管 native 池, ATen 的池要显式设.
    try:
        import torch

        torch.set_num_threads(requested)
    except ImportError:
        pass

    ctx = build_workload(args)
    registry = _register()

    results: dict[str, Any] = {}
    for name in segments:
        spec = registry[name]
        fn = spec.fn
        samples: list[float] = []
        signature: dict[str, Any] = {}
        for i in range(args.warmup + args.repeat):
            out = fn(ctx)
            if i >= args.warmup:
                samples.append(out["seconds"])
                signature = out["signature"]
        samples.sort()
        results[name] = {
            "kind": spec.kind,
            "samples": samples,
            "median": samples[len(samples) // 2],
            "min": samples[0],
            "signature": signature,
        }

    payload = {
        "threads": requested,
        "environment": probe_environment(requested),
        "workload": {
            "backend": args.backend,
            "dim": args.dim,
            "n": args.n,
            "mesh_type": args.mesh_type,
            "order": args.order,
            "operator_level": args.operator_level,
            "n_cells": ctx.n_cells,
            "n_dofs": ctx.n_dofs,
        },
        "segments": results,
    }

    print(JSON_BEGIN)
    print(json.dumps(payload, ensure_ascii=False))
    print(JSON_END)
    return 0


# ---------------------------------------------------------------------------
# 父进程 (driver)
# ---------------------------------------------------------------------------

def run_child(threads: int, args: argparse.Namespace) -> dict[str, Any]:
    """起一个把线程数定死的干净子进程, 取回它的 JSON."""
    env = os.environ.copy()
    for var in THREAD_ENV_VARS:
        env[var] = str(threads)

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", str(threads),
        "--backend", args.backend,
        "--dim", str(args.dim),
        "--n", str(args.n),
        "--mesh-type", args.mesh_type,
        "--order", str(args.order),
        "--operator-level", args.operator_level,
        "--rmin", str(args.rmin),
        "--segments", args.segments,
        "--repeat", str(args.repeat),
        "--warmup", str(args.warmup),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    text = proc.stdout
    if JSON_BEGIN not in text or JSON_END not in text:
        raise RuntimeError(
            f"线程档 {threads} 的子进程没有产出 JSON (returncode="
            f"{proc.returncode}).\n--- stdout ---\n{text}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    body = text.split(JSON_BEGIN, 1)[1].split(JSON_END, 1)[0]
    return json.loads(body)


def check_invariants(runs: list[dict[str, Any]], tol: float) -> tuple[bool, list[str]]:
    """比较各线程档的指纹, 判解是否随线程数漂移.

    Parameters
    ----------
    runs : list of dict
        按线程数升序排列的各档结果, 第一档作为参照.
    tol : float
        浮点指纹的相对差上限.

    Returns
    -------
    (bool, list of str)
        是否全部通过, 以及失败说明.

    Notes
    -----
    线程数改变的是归约顺序而非数学解. 这一项挂了说明并行路径上有 race 或
    非确定性归约超出预期, 比任何性能数字都严重.
    """
    if len(runs) < 2:
        return True, []

    base = runs[0]
    failures: list[str] = []
    for run in runs[1:]:
        for name, seg in run["segments"].items():
            ref = base["segments"][name]["signature"]
            for key, value in seg["signature"].items():
                ref_value = ref.get(key)
                if ref_value is None:
                    continue
                if isinstance(value, int) and key == "niter":
                    if abs(value - ref_value) > 1:
                        failures.append(
                            f"{name}.{key}: {run['threads']} 线程 {value} vs "
                            f"{base['threads']} 线程 {ref_value}"
                        )
                    continue
                denom = max(abs(ref_value), 1e-30)
                rel = abs(value - ref_value) / denom
                if rel > tol:
                    failures.append(
                        f"{name}.{key}: 相对差 {rel:.3e} > {tol:.1e} "
                        f"({run['threads']} 线程 vs {base['threads']} 线程)"
                    )
    return (not failures), failures


def report(runs: list[dict[str, Any]], segments: list[str],
           invariant_ok: bool, failures: list[str]) -> None:
    """打印线程扩展性表格."""
    env = runs[0]["environment"]
    if not env["trustworthy"]:
        print("=" * 78)
        print("[警告] 本机不具备可信的线程扩展性测量条件, 下表只能自查, 不得外引:")
        for item in env["caveats"]:
            print(f"  - {item}")
        print("=" * 78)
        print("")

    work = runs[0]["workload"]
    print(f"后端 {work['backend']}  网格 {work['mesh_type']} {work['n']}x{work['n']}  "
          f"order {work['order']}  operator {work['operator_level']}  "
          f"n_cells {work['n_cells']}  n_dofs {work['n_dofs']}")
    print("")

    registry = _register()
    base = runs[0]
    for name in segments:
        kind = registry[name].kind
        tag = "计算受限" if kind == "compute" else "带宽受限"
        print(f"[{name}] {registry[name].desc}  (先验: {tag})")
        header = f"  {'threads':>8} {'median(s)':>12} {'speedup':>9} {'efficiency':>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        t1 = base["segments"][name]["median"]
        for run in runs:
            t = run["segments"][name]["median"]
            n = run["threads"]
            speedup = t1 / max(t, 1e-12)
            print(f"  {n:>8} {t:>12.4f} {speedup:>8.2f}x {speedup / n:>10.2f}")
        print("")

    if invariant_ok:
        print("[invariant] 各线程档指纹一致  [OK]")
    else:
        print("[invariant] 解随线程数漂移  [FAIL]")
        for item in failures:
            print(f"  - {item}")
    print("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_segments(spec: str) -> list[str]:
    """解析 --segments, 未实现的段直接报错而不是静默跳过."""
    registry = _register()
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for name in names:
        if name not in registry:
            raise SystemExit(
                f"未知段 '{name}'. 可选: {', '.join(registry)} (用 --list 查看)"
            )
        if registry[name].fn is None:
            raise SystemExit(f"段 '{name}' 尚未实现, 当前框架只搭了注册位.")
    if not names:
        raise SystemExit("--segments 不能为空.")
    return names


def list_segments() -> None:
    """打印可选段."""
    print("可选被测段:")
    for name, spec in _register().items():
        state = "已实现" if spec.fn is not None else "未实现"
        tag = "计算受限" if spec.kind == "compute" else "带宽受限"
        print(f"  {name:<15} [{tag}] [{state}]  {spec.desc}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="线程级并行扩展性 benchmark (父子两层, 每档一个干净进程)",
    )
    parser.add_argument("--list", action="store_true", help="列出可选被测段后退出")
    parser.add_argument("--threads", default=",".join(str(n) for n in DEFAULT_THREADS),
                        help="线程档位, 逗号分隔; 超过可用核数的档会被丢弃")
    parser.add_argument("--segments", default="assemble,cg_solve,filter_spmv",
                        help="被测段, 逗号分隔")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "pytorch"],
                        help="FEALPy 后端: numpy 走 OpenBLAS, pytorch 走 ATen")
    parser.add_argument("--dim", type=int, default=2, choices=[2])
    parser.add_argument("--n", type=int, default=128, help="每方向网格数")
    parser.add_argument("--mesh-type", default="quad", choices=["quad", "tri"])
    parser.add_argument("--order", type=int, default=1, help="位移空间次数")
    parser.add_argument("--operator-level", default="fa", choices=["fa", "ea"],
                        help="fa 全装配; ea 单元级 matrix-free")
    parser.add_argument("--rmin", type=float, default=2.4, help="滤波半径")
    parser.add_argument("--repeat", type=int, default=3, help="计时次数, 取中位数")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数, 不计时")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="跨线程档指纹的相对差上限")
    parser.add_argument("--json", type=Path, default=None, help="结果落盘路径")
    parser.add_argument("--worker", type=int, default=None,
                        help=argparse.SUPPRESS)  # 内部使用: 子进程模式
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """入口. ``--worker`` 走子进程分支, 否则走派发分支."""
    args = parse_args(argv)

    if args.list:
        list_segments()
        return 0

    if args.worker is not None:
        return run_worker(args)

    segments = _resolve_segments(args.segments)
    requested = [int(s) for s in args.threads.split(",") if s.strip()]
    available = os.cpu_count() or 1
    threads = sorted({n for n in requested if 1 <= n <= available})
    dropped = sorted(set(requested) - set(threads))
    if dropped:
        print(f"[note] 丢弃超出可用核数 {available} 的档位: {dropped}")
    if not threads:
        raise SystemExit("没有可用的线程档位.")

    runs: list[dict[str, Any]] = []
    for n in threads:
        print(f"[run] {n} 线程 ...", flush=True)
        runs.append(run_child(n, args))
    print("")

    invariant_ok, failures = check_invariants(runs, args.tol)
    report(runs, segments, invariant_ok, failures)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "invariant_passed": invariant_ok,
                    "invariant_failures": failures,
                    "runs": runs,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[json] {args.json}")

    return 0 if invariant_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
