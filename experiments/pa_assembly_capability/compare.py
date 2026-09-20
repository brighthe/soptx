# -*- coding: utf-8 -*-
"""pa_assembly_capability 产物对比表: 只从 outputs/*.json 计算, 不含任何硬编码结论数字.

用法:
  python compare.py --case cache -n 32      # 阶段 1 (cache): 积分点几何缓存 阶段峰值 / 净增 / KB/dof
  python compare.py --case matvec -n 32     # 阶段 2 (matvec): 算子常驻、刚度算子乘 K x 的耗时、有效带宽下界
  python compare.py --case solve            # 阶段 3 (solve): 核心 Jacobi-PCG 迭代数随 n 的增长与全程峰值
  python compare.py --case baseline         # 单核硬件基线: memcpy 带宽与 dgemm 算力
  python compare.py --case all              # 全部表格 + 天花板外推

口径: 单进程 CPU RSS, 峰值 = 阶段内 VmHWM 绝对高水位, 净增 = 峰值 - 阶段开始 RSS.
天花板 = MEMORY_BUDGET (45 GiB 峰值预算) / 最大成功 n 的全程峰值 KB/dof, 是外推值.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from _common.metrology import DEFAULT_MEMORY_TOTAL, MEMORY_BUDGET  # noqa: E402

METHOD_ORDER = ("fast", "standard", "voigt")


# ----------------------------------------------------------------------------- 工具
def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - display_width(text))
    return text + " " * gap if align == "left" else " " * gap + text


def print_table(header: list[str], rows: list[list[str]], right_cols: set[int] | None = None) -> None:
    right_cols = right_cols or set(range(1, len(header)))
    widths = [max(display_width(r[i]) for r in (header, *rows)) for i in range(len(header))]

    def fmt(row: list[str]) -> str:
        return "  ".join(
            pad(v, widths[i], "right" if i in right_cols else "left") for i, v in enumerate(row)
        ).rstrip()

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_mib(mib: float | None) -> str:
    if mib is None:
        return "--"
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


def fmt_num(v: float | None, digits: int = 2, suffix: str = "") -> str:
    return "--" if v is None else f"{v:.{digits}f}{suffix}"


def fmt_sci(v: float | None) -> str:
    return "--" if v is None else f"{v:.2e}"


def fmt_s(t: float | None) -> str:
    if t is None:
        return "--"
    if t < 1e-3:
        return f"{t * 1e6:.0f} us"
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def artifacts(pattern: str) -> list[tuple[Path, dict[str, Any]]]:
    """按 glob 收集产物 (跳过 .failed 旁车与旧 schema 的非 cpu 产物)."""
    items = []
    for p in sorted(_OUTPUT_DIR.glob(pattern)):
        if p.name.endswith(".failed.json"):
            continue
        d = load_json(p)
        if d.get("device_type", "cpu") != "cpu":
            continue
        items.append((p, d))
    return items


def _ndof(n: int) -> int:
    return 3 * (n + 1) ** 3


def load_baseline() -> dict[str, Any] | None:
    """读取单核硬件基线产物 (outputs/baseline_cpu.json); 不存在时返回 None."""
    p = _OUTPUT_DIR / "baseline_cpu.json"
    if not p.is_file():
        # 回落到 ea 目录共用的 baseline
        ea_p = _THIS_DIR.parent / "ea_assembly_capability" / "outputs" / "baseline_cpu.json"
        if ea_p.is_file():
            return load_json(ea_p)
        return None
    return load_json(p)


def fmt_pct(v: float | None) -> str:
    return "--" if v is None else f"{v * 100:.0f} %"


# ----------------------------------------------------------------------------- baseline
def show_baseline_table() -> None:
    base = load_baseline()
    print("\n[单核硬件基线 | panel baseline] memcpy 带宽 (读 + 写) 与 dgemm 算力, 线程数 1")
    if base is None:
        print("  (无产物; 需先跑 run.py --case cpu-baseline)")
        return
    pools = base.get("threadpools")
    pool_text = ", ".join(f"{p.get('internal_api')}={p.get('num_threads')}" for p in pools) if pools else "--"
    header = ["项目", "规模", "重复", "中位耗时", "结果"]
    rows = [
        ["memcpy", f"{base.get('memcpy_array_bytes', 0) / 2**30:.0f} GiB x2", str(base.get("memcpy_repeats")),
         fmt_s(base.get("memcpy_seconds_median")), fmt_num(base.get("memcpy_gbps"), 2, " GB/s")],
        ["dgemm", f"n = {base.get('dgemm_size')}", str(base.get("dgemm_repeats")),
         fmt_s(base.get("dgemm_seconds_median")), fmt_num(base.get("dgemm_gflops"), 2, " GFLOP/s")],
    ]
    print_table(header, rows, right_cols={2, 3, 4})
    blas = base.get("blas") or {}
    print(f"  numpy {base.get('numpy_version')} | BLAS {blas.get('name')} {blas.get('version')} | "
          f"OMP/OPENBLAS/MKL_NUM_THREADS = {base.get('env_OMP_NUM_THREADS')}/{base.get('env_OPENBLAS_NUM_THREADS')}/{base.get('env_MKL_NUM_THREADS')} | 线程池: {pool_text}")


# ----------------------------------------------------------------------------- cache
def show_cache_table(n: int | None) -> None:
    items = artifacts("cache_*_n*.json")
    by_n: dict[int, dict[str, dict]] = defaultdict(dict)
    for _, d in items:
        if "cache_peak_MiB" not in d:
            continue
        by_n[int(d["n"])][d.get("method", "fast")] = d
    if n is not None:
        by_n = {k: v for k, v in by_n.items() if k == n}
    print("\n[阶段 1 积分点数据缓存 | panel cache] 几何量 (J^-1, w*detJ) + cell2dof, CPU RSS")
    if not by_n:
        print("  (无产物; 需先跑 run.py --case element-cache)")
        return
    header = ["n", "Ndof", "NQ", "cache 绝对峰值", "净增", "净增 KB/dof", "J^-1 存储", "w*detJ 存储", "算子常驻", "常驻 KB/dof", "构建耗时"]
    rows = []
    for nn in sorted(by_n):
        group = by_n[nn]
        for m in METHOD_ORDER:
            d = group.get(m)
            if d is None:
                continue
            rows.append([
                str(nn),
                f"{d['Ndof']:,}",
                str(d.get("quadrature_points", d.get("quad_points_per_cell", "--"))),
                fmt_mib(d.get("cache_peak_MiB")),
                fmt_mib(d.get("cache_net_MiB")),
                fmt_num(d.get("cache_KB_per_dof")),
                fmt_mib(d.get("jacobi_inverse_MiB")),
                fmt_mib(d.get("weighted_measure_MiB")),
                fmt_mib(d.get("operator_persistent_MiB")),
                fmt_num(d.get("operator_persistent_KB_per_dof")),
                fmt_s(d.get("cache_seconds")),
            ])
    print_table(header, rows, right_cols=set(range(1, len(header))))


# ----------------------------------------------------------------------------- matvec
def show_matvec_table(n: int | None) -> None:
    items = artifacts("matvec_*_n*.json")
    by_n: dict[int, dict] = {}
    for _, d in items:
        by_n[int(d["n"])] = d
    if n is not None:
        by_n = {k: v for k, v in by_n.items() if k == n}
    print("\n[阶段 2 算子乘性能 | panel matvec] 刚度算子乘 K x (G^T B^T D B G x) 单核耗时与有效带宽")
    if not by_n:
        print("  (无产物; 需先跑 run.py --case pa-matvec)")
        return
    base = load_baseline()
    base_bw = base.get("memcpy_gbps") if base else None
    base_gflops = base.get("dgemm_gflops") if base else None

    header = ["n", "Ndof", "算子常驻", "常驻 KB/dof", "单次耗时", "最小搬运", "有效带宽", "占 memcpy 基线", "GFLOP/s", "占 dgemm 基线"]
    rows = []
    for nn in sorted(by_n):
        d = by_n[nn]
        bw = d.get("effective_gbps_lower_bound")
        gflops = d.get("gflops")
        bw_pct = (bw / base_bw) if (bw is not None and base_bw) else None
        gf_pct = (gflops / base_gflops) if (gflops is not None and base_gflops) else None
        rows.append([
            str(nn),
            f"{d['Ndof']:,}",
            fmt_mib(d.get("operator_persistent_MiB")),
            fmt_num(d.get("operator_persistent_KB_per_dof")),
            fmt_s(d.get("matvec_seconds_median")),
            fmt_mib(d.get("bytes_moved_min_per_matvec", 0) / 2**20),
            fmt_num(bw, 2, " GB/s"),
            fmt_pct(bw_pct),
            fmt_num(gflops, 2, " GFLOP/s"),
            fmt_pct(gf_pct),
        ])
    print_table(header, rows, right_cols=set(range(1, len(header))))


# ----------------------------------------------------------------------------- solve
def show_solve_table(n: int | None) -> None:
    items = artifacts("solve_*_n*.json")
    by_n: dict[int, dict] = {}
    for _, d in items:
        by_n[int(d["n"])] = d
    if n is not None:
        by_n = {k: v for k, v in by_n.items() if k == n}
    print("\n[阶段 3 求解迭代增长 | panel solve] 核心 PA 算子 + soptx.solvers.cg (Jacobi-PCG), natural 容差 1e-6")
    if not by_n:
        print("  (无产物; 需先跑 run.py --case pa-cg-solve)")
        return
    header = ["n", "Ndof", "迭代数", "迭代/n", "求解耗时", "单步耗时", "全程峰值", "峰值 KB/dof", "真残差"]
    rows = []
    for nn in sorted(by_n):
        d = by_n[nn]
        rows.append([
            str(nn),
            f"{d['Ndof']:,}",
            str(d.get("iterations")),
            fmt_num(d.get("iterations_per_n"), 3),
            fmt_s(d.get("solve_seconds")),
            fmt_s(d.get("seconds_per_iteration")),
            fmt_mib(d.get("process_max_rss_MiB")),
            fmt_num(d.get("process_max_rss_KB_per_dof")),
            fmt_sci(d.get("true_relres")),
        ])
    print_table(header, rows, right_cols=set(range(1, len(header))))


# ----------------------------------------------------------------------------- all
def show_all(n: int | None) -> None:
    show_baseline_table()
    show_cache_table(n)
    show_matvec_table(n)
    show_solve_table(n)

    # 天花板外推
    solves = artifacts("solve_*_n*.json")
    caches = artifacts("cache_*_n*.json")
    all_runs = solves or caches
    if all_runs:
        max_n_run = max(all_runs, key=lambda item: int(item[1].get("n", 0)))[1]
        peak_kb_dof = max_n_run.get("process_max_rss_KB_per_dof") or max_n_run.get("cache_peak_KB_per_dof")
        if peak_kb_dof and peak_kb_dof > 0:
            ceiling_dof = int(MEMORY_BUDGET / (peak_kb_dof * 1000))
            print(f"\n[天花板外推] 45 GiB 预算下 (基于 n={max_n_run.get('n')} 峰值 {peak_kb_dof} KB/dof): "
                  f"最大自由度天花板约为 {ceiling_dof:,} (~{ceiling_dof / 1e6:.1f} M dof)")


def main() -> int:
    p = argparse.ArgumentParser(description="PA 评测结果报表")
    p.add_argument("--case", choices=["cache", "matvec", "solve", "baseline", "all"], default="all")
    p.add_argument("-n", type=int, default=None, help="仅查看指定网格 n")
    args = p.parse_args()

    if args.case == "cache":
        show_cache_table(args.n)
    elif args.case == "matvec":
        show_matvec_table(args.n)
    elif args.case == "solve":
        show_solve_table(args.n)
    elif args.case == "baseline":
        show_baseline_table()
    elif args.case == "all":
        show_all(args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
