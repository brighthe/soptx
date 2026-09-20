# -*- coding: utf-8 -*-
"""线弹性矩阵组装层级横向对比 (stored-B / EA / FA / PA / shared-Ke) 统一驱动.

本模块只测性能与容量, 不验正确性。

正确性证据的分布要看方案类别: fa / ea / pa 是装配层级分类法里的层级, 它们的求解链收敛阶
与彼此一致性由 ``experiments/assembly_level_consistency/`` 给出; stored-b / shared-ke 不是
层级, 只是本目录的对照实现, 其正确性前提由本目录自负, 眼下尚无取证 —— 引用本目录的性能
数字时要知道这一点。两边共用 ``experiments/_common/assembly_levels.py`` 的同一套问题与
算子实现。

本模块实现:
1. Worker 测量层:
   - bandwidth:   memcpy 带宽基线 (np.copyto 2 GiB float64)
   - matvec:      单一方案的常驻字节数、峰值 RSS、单次 MatVec 中位耗时与有效带宽
   - solve:       shared-Ke 算子搭载 Jacobi-PCG, 记录迭代数随 n 的增长
2. 调度与编排层: 读取 cases.toml, 以独立子进程 (单线程环境) 跑指定数据点
3. 终端看板: Style B (Modern Tree Card) 树状卡片输出

五种方案共用同一网格 (单位立方体 n^3 六面体 Q1)、同一材料 (E = 1, nu = 0.3)、
同一积分 (q = 2, 8 点)、同一交错自由度布局 (dof = 3 * node + comp)。

其中 ea / fa / pa 由 ``soptx.fem.levels.create_level`` 构造, 量的就是生产栈本身;
stored-b (臧昕禹推断方案) 与 shared-ke (均匀网格共享一份 K_e) 没有生产对应物, 是本
模块自带的对照实现。
"""

from __future__ import annotations

import os
import sys

# 直接以 --worker 启动 (不经调度器) 时也要在 import numpy 之前锁定单线程,
# 否则 OpenBLAS 会按核数开线程, 五方案的线程口径不一致。
if "--worker" in sys.argv:
    for _k, _v in (
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(_k, _v)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
for _p in (_THIS_DIR, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from _common.assembly_levels import (  # noqa: E402
    build_operator,
    common_header,
    fmt_bytes,
    fmt_seconds,
    get_current_rss_bytes,
    get_peak_rss_bytes,
    memory_tail,
    reference_K_e,
    setup_problem,
)

# -----------------------------------------------------------------------------
# 1. Worker 测量层
# -----------------------------------------------------------------------------

def measure_bandwidth(repeats: int = 10, size_bytes: int = 2 * 1024**3) -> dict:
    """bandwidth: np.copyto 大数组拷贝, 报告读+写 GB/s (中位数)."""
    count = size_bytes // 8
    src = np.ones(count, dtype=np.float64)
    dst = np.empty_like(src)
    np.copyto(dst, src)  # warmup / 首次触页
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    out = common_header("bandwidth_memcpy", "bandwidth", "machine-baseline", "memcpy", 0, None, config.THREAD_ENV)
    out["mesh_type"] = "-"
    out["grid"] = "-"
    out["problem"] = "-"
    out["metrics"] = {
        "array_bytes": int(src.nbytes),
        "repeats": repeats,
        "copy_seconds_median": med,
        "copy_seconds_all": times,
        "bandwidth_gbps_read_plus_write": 2.0 * src.nbytes / med / 1e9,
    }
    out.update(memory_tail(get_peak_rss_bytes(), 0))
    return out


def measure_matvec(scheme: str, n: int, repeats: int = 20, seed: int = 0) -> dict:
    """matvec: 单一方案的常驻字节、峰值 RSS、单次 MatVec 中位耗时与有效带宽下界."""
    t0 = time.perf_counter()
    ctx = setup_problem(n)
    t_setup = time.perf_counter() - t0
    gc.collect()
    rss_after_setup = get_current_rss_bytes()

    t1 = time.perf_counter()
    # 只有 shared-ke 还要外部的参考 K_e; ea / fa 的单元矩阵由生产层级的 build 内部算,
    # 仍落在 t1 计时区间内, build_seconds 的口径不变。
    K_e = reference_K_e(ctx) if scheme == "shared-ke" else None
    op = build_operator(scheme, ctx, K_e)
    del K_e
    gc.collect()
    t_build = time.perf_counter() - t1
    peak_after_build = get_peak_rss_bytes()
    rss_after_build = get_current_rss_bytes()

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(ctx["n_dofs"])
    y = op.apply(x)  # warmup
    times: List[float] = []
    for _ in range(repeats):
        t2 = time.perf_counter()
        y = op.apply(x)
        times.append(time.perf_counter() - t2)
    med = float(np.median(times))
    peak_bytes = get_peak_rss_bytes()
    rss_after_matvec = get_current_rss_bytes()

    persistent = op.persistent_bytes()
    # 下界: 常驻数据读一遍 + x 读一遍 + y 写一遍 (不含 gather/scatter 的临时向量往返)
    bytes_moved_min = persistent + 2 * 8 * ctx["n_dofs"]
    metrics: Dict[str, Any] = {
        "repeats": repeats,
        "seed": seed,
        "setup_seconds": t_setup,
        "build_seconds": t_build,
        "peak_after_build_bytes": peak_after_build,
        "rss_after_setup_bytes": rss_after_setup,
        "rss_after_build_bytes": rss_after_build,
        "rss_after_matvec_bytes": rss_after_matvec,
        "build_net_rss_bytes": rss_after_build - rss_after_setup,
        "persistent_bytes": persistent,
        "persistent_bytes_per_cell": persistent / ctx["n_cells"],
        "persistent_bytes_per_dof": persistent / ctx["n_dofs"],
        "theory_numbers_per_cell": op.numbers_per_cell,
        "matvec_seconds_median": med,
        "matvec_seconds_min": float(np.min(times)),
        "matvec_seconds_all": times,
        "bytes_moved_min_per_matvec": bytes_moved_min,
        "effective_gbps_lower_bound": bytes_moved_min / med / 1e9,
        "y_norm": float(np.linalg.norm(y)),
    }
    if scheme == "fa":
        metrics["nnz"] = op.nnz
        metrics["nnz_per_row"] = op.nnz / ctx["n_dofs"]
        metrics["index_dtype"] = op.index_dtype
    if scheme == "shared-ke":
        metrics["max_deviation_from_cell0"] = op.max_deviation

    role = "matvec-benchmark-student-scale" if n >= 100 else "matvec-benchmark"
    out = common_header(f"matvec_{scheme}_n{n}", "matvec", role, scheme, n, ctx, config.THREAD_ENV)
    out["metrics"] = metrics
    out.update(memory_tail(peak_bytes, ctx["n_dofs"]))
    return out


def measure_solve(scheme: str, n: int, maxiter: int = 5000, tol: float = 1e-6) -> dict:
    """solve: Jacobi-PCG (全边界 Dirichlet, 内部 b = 1), 记录迭代数与单步时间."""
    t0 = time.perf_counter()
    ctx = setup_problem(n)
    K_e = reference_K_e(ctx)
    op = build_operator(scheme, ctx, K_e)

    # Jacobi 对角: 单元对角 scatter-add
    n_dofs = ctx["n_dofs"]
    diag = np.zeros(n_dofs, dtype=np.float64)
    np.add.at(diag, ctx["cell2dof"].ravel(), np.einsum("cii->ci", K_e).ravel())
    if scheme != "ea":
        del K_e
    gc.collect()

    is_bd = np.asarray(ctx["tensor_space"].is_boundary_dof(), dtype=bool)
    is_int = ~is_bd
    diag[is_bd] = 1.0
    inv_diag = 1.0 / diag

    def A_op(v: np.ndarray) -> np.ndarray:
        return np.where(is_int, op.apply(v), v)

    b = np.zeros(n_dofs, dtype=np.float64)
    b[is_int] = 1.0
    b_norm = float(np.linalg.norm(b))

    x = np.zeros(n_dofs, dtype=np.float64)
    r = b.copy()
    z = inv_diag * r
    p = z.copy()
    rz = float(np.dot(r, z))
    rel_res = 1.0
    converged = False
    it_count = 0

    t_solve = time.perf_counter()
    for _ in range(maxiter):
        it_count += 1
        Ap = A_op(p)
        alpha = rz / float(np.dot(p, Ap))
        x += alpha * p
        r -= alpha * Ap
        r[is_bd] = 0.0
        rel_res = float(np.linalg.norm(r) / b_norm)
        if rel_res < tol:
            converged = True
            break
        z = inv_diag * r
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    solve_elapsed = time.perf_counter() - t_solve
    peak_bytes = get_peak_rss_bytes()

    out = common_header(f"solve_{scheme}_n{n}", "solve", "jacobi-pcg-iteration-growth", scheme, n, ctx, config.THREAD_ENV)
    out["metrics"] = {
        "preconditioner": "jacobi",
        "boundary": "all-boundary-dirichlet",
        "rhs": "ones-on-interior",
        "tolerance": tol,
        "maxiter": maxiter,
        "iterations": it_count,
        "converged": converged,
        "final_relres": rel_res,
        "iterations_per_n": it_count / n,
        "solve_seconds": solve_elapsed,
        "seconds_per_iteration": solve_elapsed / it_count if it_count else 0.0,
        "total_seconds": time.perf_counter() - t0,
        "persistent_bytes": op.persistent_bytes(),
    }
    out.update(memory_tail(peak_bytes, n_dofs))
    return out


# -----------------------------------------------------------------------------
# 2. 控制台树状卡片看板 (Style B Dashboard)
# -----------------------------------------------------------------------------

def print_dashboard(out: Dict[str, Any]) -> None:
    """以统一 Style B 树状卡片格式在控制台打印测量报告."""
    panel = out.get("panel", "")
    m = out.get("metrics", {})
    print(f"\n● [{out.get('case_id')}] {out.get('problem')}")
    if panel != "bandwidth":
        print(
            f"  ├── Mesh & DOFs   : {out.get('mesh_type')} (grid = {out.get('grid')}) | "
            f"{out.get('n_cells', 0):,} cells | {out.get('n_dofs', 0):,} DOFs"
        )
    print(f"  ├── Threads       : {out.get('env', {}).get('threads')}")

    if panel == "bandwidth":
        print(f"  ├── Array         : {fmt_bytes(m.get('array_bytes', 0))} x 2, repeats = {m.get('repeats')}")
        print(f"  ├── memcpy median : {fmt_seconds(m.get('copy_seconds_median', 0.0))}")
        print(f"  └── Bandwidth     : {m.get('bandwidth_gbps_read_plus_write', 0.0):.2f} GB/s (read + write)")

    elif panel == "matvec":
        print(f"  ├── Scheme        : {out.get('scheme')} | build = {fmt_seconds(m.get('build_seconds', 0.0))}")
        print(
            f"  ├── Persistent    : {fmt_bytes(m.get('persistent_bytes', 0))} "
            f"({m.get('persistent_bytes_per_cell', 0.0) / 1000:.2f} kB/cell, {m.get('theory_numbers_per_cell', 0.0):.0f} numbers/cell)"
        )
        print(
            f"  ├── RSS timeline  : setup {fmt_bytes(m.get('rss_after_setup_bytes', 0))} -> "
            f"build {fmt_bytes(m.get('rss_after_build_bytes', 0))} -> "
            f"matvec {fmt_bytes(m.get('rss_after_matvec_bytes', 0))} "
            f"(build net {fmt_bytes(m.get('build_net_rss_bytes', 0))})"
        )
        print(f"  ├── Peak RSS      : {fmt_bytes(out.get('peak_memory_bytes', 0))}")
        print(
            f"  ├── MatVec median : {fmt_seconds(m.get('matvec_seconds_median', 0.0))} "
            f"(min {fmt_seconds(m.get('matvec_seconds_min', 0.0))}, repeats = {m.get('repeats')})"
        )
        print(f"  └── Eff. GB/s     : {m.get('effective_gbps_lower_bound', 0.0):.2f} (lower bound)")

    elif panel == "solve":
        print(f"  ├── Scheme        : {out.get('scheme')} | Jacobi-PCG tol = {m.get('tolerance')}")
        print(
            f"  ├── Iterations    : {m.get('iterations')} (converged = {m.get('converged')}, "
            f"rel_res = {m.get('final_relres', 0.0):.2e}, iters/n = {m.get('iterations_per_n', 0.0):.2f})"
        )
        print(
            f"  ├── Solve time    : {fmt_seconds(m.get('solve_seconds', 0.0))} "
            f"({fmt_seconds(m.get('seconds_per_iteration', 0.0))} / iter)"
        )
        print(f"  └── Peak RSS      : {fmt_bytes(out.get('peak_memory_bytes', 0))}")
    print()


# -----------------------------------------------------------------------------
# 3. 调度层与子进程控制
# -----------------------------------------------------------------------------

def command_list(cases: Tuple[config.Case, ...], figure: dict) -> int:
    """列出已注册的数据点."""
    headers = ["case-id", "panel", "scheme", "grid", "repeats"]
    rows = [[c.id, c.panel, c.scheme, c.grid, c.repeats] for c in cases]
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))
    print(f"\nfigure: {figure.get('id')} — {figure.get('title')}")
    print("产物: outputs/<case-id>.json")
    print("  ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)))
    print("  ".join("-" * col_widths[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(r)))
    print()
    return 0


def command_run(
    selected: Tuple[config.Case, ...],
    check_only: bool = False,
    skip_existing: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> int:
    """按独立子进程 (单线程环境) 调度执行已选工况."""
    repo_root = Path(__file__).resolve().parents[2]
    total = len(selected)
    failed = 0

    print(f"\n============ assembly_level_capability 调度执行 ({total} 个任务) ============")
    for idx, case in enumerate(selected, 1):
        artifact_path = case.artifact_path
        if overrides:
            n_val = overrides.get("n", case.n)
            scheme_val = overrides.get("scheme", case.scheme)
            artifact_path = _OUTPUT_DIR / f"{case.panel}_{scheme_val}_n{n_val}.json"

        if skip_existing and artifact_path.is_file():
            print(f"[{idx}/{total}] 跳过已存在产物: {artifact_path.name}")
            continue

        cmd = case.to_command(repo_root, overrides)
        cmd.extend(["--output", str(artifact_path)])

        if check_only:
            env_str = " ".join(f"{k}={v}" for k, v in config.THREAD_ENV.items())
            print(f"[{idx}/{total}] [dry-run] {env_str} {' '.join(cmd)}")
            continue

        print(f"[{idx}/{total}] 调度子进程: {case.id} (输出 -> {artifact_path.name})")
        t_start = time.perf_counter()
        completed = subprocess.run(cmd, env=case.subprocess_env(), cwd=repo_root)
        elapsed = time.perf_counter() - t_start

        if completed.returncode != 0:
            failed += 1
            print(f"  失败: 退出码 {completed.returncode}, 用时 {elapsed:.1f} s")
        elif not artifact_path.is_file():
            failed += 1
            print(f"  失败: 进程正常退出但产物未生成 -> {artifact_path}")
        else:
            print(f"  完成: 用时 {elapsed:.1f} s")

    if failed:
        print(f"\n{failed} 个任务执行失败。")
    return failed


# -----------------------------------------------------------------------------
# 4. 主入口与参数路由
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="assembly_level_capability 实验驱动: 五种矩阵组装层级的内存与 MatVec 耗时",
    )

    # 1. 调度与工况选择参数
    parser.add_argument("--list", action="store_true", help="列出已注册数据点")
    parser.add_argument("--all", action="store_true", help="跑全部工况")
    parser.add_argument("--cases", nargs="+", help="指定要跑的一个或多个 case id")
    parser.add_argument("--case", help="指定单个 case id")
    parser.add_argument("--panel", choices=config.PANELS, help="只跑指定面板的数据点")
    parser.add_argument("--check-only", action="store_true", help="只打印将执行的子进程命令")
    parser.add_argument("--skip-existing", action="store_true", help="产物已存在时跳过")

    # 2. 工况动态覆盖参数
    parser.add_argument("-n", "--n", "--grid", dest="n", type=int, default=None, help="覆盖网格剖分段数")
    parser.add_argument("--scheme", choices=config.SCHEMES + config.PSEUDO_SCHEMES, default=None, help="覆盖方案")
    parser.add_argument("--repeats", type=int, default=None, help="覆盖 MatVec 计时重复次数")

    # 3. Worker 测量层底层参数
    parser.add_argument("--worker", action="store_true", help="进入子进程 worker 测量模式")
    parser.add_argument("--bandwidth", action="store_true", help="memcpy 带宽基线")
    parser.add_argument("--matvec", action="store_true", help="单一方案 MatVec 内存与耗时")
    parser.add_argument("--solve", action="store_true", help="Jacobi-PCG 迭代数")
    parser.add_argument("--output", type=Path, default=None, help="产物落盘路径")

    args = parser.parse_args(argv)

    # ------------------------------------------------ Worker 测量分支
    if args.worker or args.bandwidth or args.matvec or args.solve:
        if args.bandwidth:
            out = measure_bandwidth(repeats=args.repeats or 10)
        elif args.matvec:
            if args.n is None or args.scheme not in config.SCHEMES:
                parser.error("--matvec 必须指定 --n 与 --scheme (stored-b/ea/fa/pa/shared-ke)")
            out = measure_matvec(args.scheme, args.n, repeats=args.repeats or 20)
        elif args.solve:
            if args.n is None or args.scheme not in config.SCHEMES:
                parser.error("--solve 必须指定 --n 与 --scheme")
            out = measure_solve(args.scheme, args.n)
        else:
            parser.error("Worker 模式需指定 --bandwidth, --matvec 或 --solve")

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
        return command_list(cases, figure)

    target_case_ids: List[str] = []
    panel_filter: Optional[str] = args.panel
    is_all = args.all

    raw_cases: List[str] = []
    if args.cases:
        raw_cases.extend(args.cases)
    if args.case:
        raw_cases.append(args.case)
    for item in raw_cases:
        if item.lower() == "all":
            is_all = True
        elif item.lower() in config.PANELS:
            panel_filter = item.lower()
        else:
            target_case_ids.append(item)

    if not (is_all or target_case_ids or panel_filter):
        print(
            "错误: 必须通过 --case / --cases / --panel / --all 指定要运行的工况。\n"
            "  常用示例:\n"
            "    python run.py --case bandwidth_memcpy\n"
            "    python run.py --panel matvec --check-only\n"
            "    python run.py --case matvec_stored-b_n48\n"
            "  查看全部工况列表请使用: python run.py --list",
            file=sys.stderr,
        )
        return 2

    try:
        selected = config.select(cases, case_ids=target_case_ids or None, panel=panel_filter)
    except config.ConfigError as error:
        print(str(error), file=sys.stderr)
        return 2

    overrides: Dict[str, Any] = {}
    if args.n is not None:
        overrides["n"] = args.n
    if args.scheme is not None:
        overrides["scheme"] = args.scheme
    if args.repeats is not None:
        overrides["repeats"] = args.repeats

    failed = command_run(
        selected,
        check_only=args.check_only,
        skip_existing=args.skip_existing,
        overrides=overrides or None,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
