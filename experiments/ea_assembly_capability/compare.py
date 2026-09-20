# -*- coding: utf-8 -*-
"""ea_assembly_capability 产物对比表: 只从 outputs/*.json 计算, 不含任何硬编码结论数字.

用法:
  python compare.py --case cache -n 32      # 阶段 1 (cache): K_e 阶段峰值 / 净增 / KB/dof / 相对理论量
  python compare.py --case matvec -n 32     # 阶段 2 (matvec): 算子常驻、刚度算子乘 K x 的耗时、有效带宽下界
  python compare.py --case solve            # 阶段 3 (solve): 核心 Jacobi-PCG 迭代数随 n 的增长与全程峰值
  python compare.py --case baseline         # 单核硬件基线: memcpy 带宽与 dgemm 算力
  python compare.py --case all              # 四张表 + 天花板外推

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
    return load_json(p) if p.is_file() else None


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
        by_n[int(d["n"])][d["method"]] = d
    if n is not None:
        by_n = {k: v for k, v in by_n.items() if k == n}
    print("\n[阶段 1 单刚计算 | panel cache] K_e + cell2dof = LagrangeFEMAnalyzer.assemble_stiff_matrix('ea') (与 fa 阶段 1 同口径), CPU RSS")
    if not by_n:
        print("  (无产物; 需先跑 run.py --case element-cache)")
        return
    header = ["n", "Ndof", "method", "cache 绝对峰值", "净增", "净增 KB/dof", "相对 fast", "K_e 理论", "净增/理论", "常驻 RSS", "算子常驻", "常驻 KB/dof", "耗时"]
    rows = []
    for nn in sorted(by_n):
        group = by_n[nn]
        base = group.get("fast", {}).get("cache_KB_per_dof")
        for m in METHOD_ORDER:
            d = group.get(m)
            if d is None:
                continue
            kb = d.get("cache_KB_per_dof")
            ratio = None if (base is None or kb is None or base == 0) else kb / base
            rows.append([
                str(nn), f"{d['Ndof']:,}", m, fmt_mib(d.get("cache_peak_MiB")), fmt_mib(d.get("cache_net_MiB")),
                fmt_num(kb), fmt_num(ratio, 2, "x"), fmt_mib(d.get("Ke_theory_MiB")),
                fmt_num(d.get("cache_net_over_Ke_theory"), 2, "x"),
                fmt_mib(d.get("cache_retained_after_trim_MiB")),
                fmt_mib(d.get("operator_persistent_MiB")), fmt_num(d.get("operator_persistent_KB_per_dof")),
                fmt_s(d.get("t_cache_s")),
            ])
    print_table(header, rows, right_cols=set(range(len(header))) - {2})
    print("  峰值 = 阶段内 VmHWM (含网格、空间与分析器的常驻), 净增 = 峰值 - 阶段开始 RSS; 净增/理论 > 1 表示中间张量膨胀.")
    print("  常驻 RSS = 阶段结束 (gc + malloc_trim 后) 相对阶段起点的 RSS 净增, 与 fa stage1_retained_after_trim 同口径; 未 trim 的值见 cache_retained_MiB.")
    print("  算子常驻 = const 积分子缓存的 K_e + cell2dof 的字节数, 与 method 无关.")


# ----------------------------------------------------------------------------- matvec
def show_matvec_table(n: int | None) -> None:
    items = artifacts("matvec_*_n*.json")
    rows_src = [d for _, d in items if "matvec_seconds_median" in d]
    if n is not None:
        rows_src = [d for d in rows_src if int(d["n"]) == n]
    print("\n[阶段 2 算子乘 | panel matvec] 核心 EA 刚度算子乘 K x = operator.form @ x (soptx.fem.BilinearForm), CPU RSS")
    if not rows_src:
        print("  (无产物; 需先跑 run.py --case ea-matvec)")
        return
    base = load_baseline() or {}
    memcpy_gbps = base.get("memcpy_gbps")
    dgemm_gflops = base.get("dgemm_gflops")

    def ratio(v: float | None, ref: float | None) -> float | None:
        return None if (v is None or not ref) else v / ref

    header = ["n", "Ndof", "method", "算子常驻", "常驻 KB/dof", "中位耗时", "最小耗时", "有效带宽下界", "带宽占比", "GFLOP/s", "算力占比", "相对参考误差", "全程峰值", "峰值 KB/dof"]
    rows = []
    for d in sorted(rows_src, key=lambda x: (int(x["n"]), x["method"])):
        rows.append([
            str(d["n"]), f"{d['Ndof']:,}", d["method"],
            fmt_mib(d.get("operator_persistent_MiB")), fmt_num(d.get("operator_persistent_KB_per_dof")),
            fmt_s(d.get("matvec_seconds_median")), fmt_s(d.get("matvec_seconds_min")),
            fmt_num(d.get("effective_gbps_lower_bound"), 2, " GB/s"), fmt_pct(ratio(d.get("effective_gbps_lower_bound"), memcpy_gbps)),
            fmt_num(d.get("gflops")), fmt_pct(ratio(d.get("gflops"), dgemm_gflops)),
            fmt_sci(d.get("matvec_vs_reference_relerr")),
            fmt_mib(d.get("process_max_rss_MiB")), fmt_num(d.get("process_max_rss_KB_per_dof")),
        ])
    print_table(header, rows, right_cols=set(range(len(header))) - {2})
    print("  K x = operator.form @ x: gather x[cell2dof] -> einsum -> index_add (assembly-levels.md §2.3); 含 Dirichlet 投影的 facade @ x 只在阶段 3 由 cg 调用.")
    print("  有效带宽下界 = (算子常驻 + 16 B/dof) / 中位耗时, 不计 gather/scatter 的随机访问放大; 相对参考误差以纯 numpy 参考实现为准.")
    if memcpy_gbps and dgemm_gflops:
        print(f"  带宽占比 = 有效带宽下界 / 单核 memcpy {memcpy_gbps:.2f} GB/s; 算力占比 = GFLOP/s / 单核 dgemm {dgemm_gflops:.2f} GFLOP/s (outputs/baseline_cpu.json).")
    else:
        print("  带宽占比 / 算力占比需先跑 run.py --case cpu-baseline.")


# ----------------------------------------------------------------------------- solve
def show_solve_table() -> None:
    items = artifacts("solve_*_n*.json")
    rows_src = [d for _, d in items if "iterations" in d]
    print("\n[阶段 3 求解 | panel solve] 核心 ElasticityEAOperator + soptx.solvers.cg (Jacobi, 制造解问题的体力右端与 Dirichlet 边界), CPU RSS")
    if not rows_src:
        print("  (无产物; 需先跑 run.py --case ea-cg-solve)")
        return
    header = ["n", "Ndof", "method", "迭代数", "迭代数/n", "单步耗时", "求解耗时", "收敛", "停机相对残差", "真相对残差", "全程峰值", "峰值 KB/dof"]
    rows = []
    for d in sorted(rows_src, key=lambda x: (int(x["n"]), x["method"])):
        rows.append([
            str(d["n"]), f"{d['Ndof']:,}", d["method"], f"{d.get('iterations', 0):,}", fmt_num(d.get("iterations_per_n")),
            fmt_s(d.get("seconds_per_iteration")), fmt_s(d.get("solve_seconds")),
            "是" if d.get("converged") else "否", fmt_sci(d.get("final_relres")), fmt_sci(d.get("true_relres")),
            fmt_mib(d.get("process_max_rss_MiB")), fmt_num(d.get("process_max_rss_KB_per_dof")),
        ])
    print_table(header, rows, right_cols=set(range(len(header))) - {2, 7})
    print("  停机相对残差 = cg 的 ||r_k||_{M^-1} / ||r_0||_{M^-1} (natural 口径, 初值为 prescribed_solution); 真相对残差 = ||b - A x|| / ||b||.")
    print("  迭代数/n 近似常数即 Jacobi-CG 的条件数 ~ h^-2 行为 (迭代数 ~ n).")

    # 天花板外推 (以最大成功 n 的全程峰值 KB/dof)
    biggest = max(rows_src, key=lambda x: int(x["n"]))
    kb = biggest.get("process_max_rss_KB_per_dof")
    if kb:
        ceiling = MEMORY_BUDGET / (kb * 1000)
        n_ceiling = round((ceiling / 3) ** (1 / 3)) - 1
        print(
            f"\n  天花板外推 (峰值预算 {MEMORY_BUDGET / 2**30:.0f} GiB / n={biggest['n']} 的全程峰值 {kb:.2f} KB/dof;"
            f" 机器 MemTotal {DEFAULT_MEMORY_TOTAL / 2**30:.2f} GiB):"
        )
        print(f"    约 {ceiling / 1e6:.2f} M dof (n≈{n_ceiling}, Ndof {_ndof(n_ceiling):,}); 假设峰值 KB/dof 随 n 不变.")


# ----------------------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="all", help="cache | matvec | solve | baseline | all")
    parser.add_argument("-n", type=int, default=None, help="cache/matvec 表只看指定 n")
    parser.add_argument("--list", action="store_true", help="列出 outputs/ 下的产物")
    args = parser.parse_args(argv)

    if args.list:
        for p in sorted(_OUTPUT_DIR.glob("*.json")):
            print(p.name)
        return 0

    case = args.case.lower()
    known = {"cache", "element-cache", "matvec", "ea-matvec", "solve", "ea-cg-solve", "baseline", "cpu-baseline", "all"}
    if case not in known:
        parser.error(f"未知 --case {args.case!r}")
    if case in ("baseline", "cpu-baseline", "all"):
        show_baseline_table()
    if case in ("cache", "element-cache", "all"):
        show_cache_table(args.n)
    if case in ("matvec", "ea-matvec", "all"):
        show_matvec_table(args.n)
    if case in ("solve", "ea-cg-solve", "all"):
        show_solve_table()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
