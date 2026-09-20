# -*- coding: utf-8 -*-
"""五种矩阵组装层级 (stored-B / EA / FA / PA / shared-Ke) 的对比报表.

直接读取 outputs/ 下的 JSON 产物, 打印:
1. bandwidth:   memcpy 带宽基线
2. matvec:      各 n 下的常驻 / 峰值内存、kB/cell、MatVec 中位耗时、有效带宽、相对 shared-ke 倍数
3. solve:       Jacobi-PCG 迭代数随 n 的增长

装配层级 (fa / ea / pa) 的正确性报表在 ``experiments/assembly_level_consistency/compare.py``,
那边看的是各层级求解链的制造解收敛阶与逐档一致性。stored-b / shared-ke 两个对照实现不在
那边取证, 其正确性前提由本目录自负。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config  # noqa: E402

SCHEME_LABEL = {
    "stored-b": "stored-B (B_q + D_q 逐点落盘)",
    "ea": "EA (逐单元 K_e)",
    "fa": "FA (全局 CSR)",
    "pa": "PA (J^{-1}_q + w detJ)",
    "shared-ke": "shared-Ke (共享 K_e)",
}


def load_artifact(filename: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """安全读取单点 JSON 产物."""
    p = (output_dir or _OUTPUT_DIR) / filename
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _gib(b: float) -> str:
    return f"{b / 1024**3:.3f}" if b >= 0.1 * 1024**3 else f"{b / 1024**2:.1f} MiB"


def _sec(s: float) -> str:
    return f"{s:.3f} s" if s >= 1.0 else f"{s * 1000:.2f} ms"


def show_bandwidth(output_dir: Optional[Path] = None) -> Optional[float]:
    data = load_artifact("bandwidth_memcpy.json", output_dir)
    print("\n" + "=" * 100)
    print("【bandwidth: memcpy 基线】")
    print("-" * 100)
    if not data:
        print("  [未生成产物: bandwidth_memcpy.json]")
        print("=" * 100 + "\n")
        return None
    m = data["metrics"]
    gbps = m["bandwidth_gbps_read_plus_write"]
    print(
        f"  np.copyto {m['array_bytes'] / 1024**3:.0f} GiB float64, repeats = {m['repeats']}, "
        f"median = {_sec(m['copy_seconds_median'])}, 读+写 = {gbps:.2f} GB/s"
    )
    print("=" * 100 + "\n")
    return gbps


def show_matvec_table(n: int, output_dir: Optional[Path] = None, bandwidth_gbps: Optional[float] = None) -> None:
    rows: List[Dict[str, Any]] = []
    for s in config.SCHEMES:
        data = load_artifact(f"matvec_{s}_n{n}.json", output_dir)
        if data:
            rows.append(data)
    print("\n" + "=" * 120)
    print(f"【matvec: 单次 MatVec 内存与耗时, n = {n}】")
    print("-" * 120)
    if not rows:
        print(f"  [未生成任何 matvec_*_n{n}.json]")
        print("=" * 120 + "\n")
        return
    ref = next((r for r in rows if r["scheme"] == "shared-ke"), None)
    ref_t = ref["metrics"]["matvec_seconds_median"] if ref else None
    print(
        f"  {'方案':<11} {'常驻 GiB':>10} {'build净增':>10} {'峰值 GiB':>10} {'kB/cell':>9} {'numbers/cell':>13} "
        f"{'build':>10} {'MatVec 中位':>12} {'eff GB/s':>9} {'带宽占比':>9} {'x shared-ke':>12}"
    )
    print("-" * 120)
    for r in rows:
        m = r["metrics"]
        t = m["matvec_seconds_median"]
        ratio = f"{t / ref_t:.1f}" if ref_t else "-"
        frac = f"{m['effective_gbps_lower_bound'] / bandwidth_gbps * 100:.0f}%" if bandwidth_gbps else "-"
        net = m.get("build_net_rss_bytes")
        net_s = f"{net / 1024**3:>10.3f}" if net is not None else f"{'-':>10}"
        print(
            f"  {r['scheme']:<11} {m['persistent_bytes'] / 1024**3:>10.3f} {net_s} {r['peak_memory_bytes'] / 1024**3:>10.3f} "
            f"{m['persistent_bytes_per_cell'] / 1000:>9.2f} {m['theory_numbers_per_cell']:>13.0f} "
            f"{_sec(m['build_seconds']):>10} {_sec(t):>12} {m['effective_gbps_lower_bound']:>9.2f} {frac:>9} {ratio:>12}"
        )
    print("-" * 120)
    r0 = rows[0]
    print(f"  n_cells = {r0['n_cells']:,}, n_dofs = {r0['n_dofs']:,}; 线程 = {r0['env']['threads']}")
    print("  eff GB/s = (常驻字节 + 16 B x n_dofs) / MatVec 时间, 是搬运字节数的下界口径。")
    print("  build净增 = /proc/self/statm 当前 RSS 在 build 前后的差; 峰值 GiB 是 ru_maxrss, 含 import 与建网格的固定开销。")
    print("=" * 120 + "\n")


def show_solve_table(ns=(32, 48, 64), scheme: str = "shared-ke", output_dir: Optional[Path] = None) -> None:
    print("\n" + "=" * 100)
    print(f"【solve: Jacobi-PCG 迭代数随 n 增长, 算子 = {scheme}】")
    print("-" * 100)
    print(f"  {'n':>5} {'n_dofs':>12} {'iterations':>11} {'iters/n':>8} {'s/iter':>12} {'solve 总时':>12} {'rel_res':>10}")
    print("-" * 100)
    found = False
    for n in ns:
        data = load_artifact(f"solve_{scheme}_n{n}.json", output_dir)
        if not data:
            print(f"  {n:>5} [未生成产物]")
            continue
        found = True
        m = data["metrics"]
        print(
            f"  {n:>5} {data['n_dofs']:>12,d} {m['iterations']:>11d} {m['iterations_per_n']:>8.2f} "
            f"{_sec(m['seconds_per_iteration']):>12} {_sec(m['solve_seconds']):>12} {m['final_relres']:>10.2e}"
        )
    print("-" * 100)
    if found:
        print("  iters/n 近似常数 => 迭代数随 n 线性增长 (Jacobi-CG 的条件数 O(h^-2) 行为)。")
    print("=" * 100 + "\n")


def show_summary_report(output_dir: Optional[Path] = None) -> None:
    gbps = show_bandwidth(output_dir)
    for n in (48, 104):
        show_matvec_table(n, output_dir, gbps)
    show_solve_table(output_dir=output_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="compare.py", description="assembly_level_capability 结果对比报表")
    parser.add_argument("--list", action="store_true", help="列出可打印的报表")
    parser.add_argument(
        "--case",
        choices=["bandwidth", "matvec", "solve", "all"],
        default="all",
        help="指定展示的报表 (默认 all)",
    )
    parser.add_argument("-n", "--n", type=int, default=None, help="matvec 报表的网格段数")
    parser.add_argument("--output-dir", type=Path, default=None, help="自定义产物目录")
    args = parser.parse_args(argv)

    if args.list:
        print("\ncase          description")
        print("------------  ------------------------------------------------------------")
        print("bandwidth     memcpy 带宽基线")
        print("matvec        指定 n 的五方案 MatVec 内存与耗时表 (默认 n = 48 与 104)")
        print("solve         Jacobi-PCG 迭代数随 n 增长 (n = 32/48/64)")
        print("all           以上全部\n")
        return 0

    if args.case == "bandwidth":
        show_bandwidth(args.output_dir)
    elif args.case == "matvec":
        gbps = None
        bw = load_artifact("bandwidth_memcpy.json", args.output_dir)
        if bw:
            gbps = bw["metrics"]["bandwidth_gbps_read_plus_write"]
        for n in ([args.n] if args.n else [48, 104]):
            show_matvec_table(n, args.output_dir, gbps)
    elif args.case == "solve":
        show_solve_table(output_dir=args.output_dir)
    else:
        show_summary_report(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
