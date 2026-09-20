# -*- coding: utf-8 -*-
"""fa_assembly_capability 产物对比表: 只从 outputs/*.json 计算, 不含任何硬编码结论数字.

用法:
  python compare.py --case mesh             # 只建网格与空间: 公共容量上界 (与装配层级无关)
  python compare.py --case stage1           # 阶段 1 单刚: 各 method 的绝对峰值 / 净增 / KB/dof
  python compare.py --case stage2 -n 32     # 阶段 2 合并: 四条路线的 B/triplet (含生产/原型标记)
  python compare.py --case full             # 端到端: 行 = n, 列 = 路线, 格 = 绝对峰值 GiB (KB/dof); OOM 行来自 .failed.json
  python compare.py --case all              # 三张表 + 天花板外推

口径: 全部为 CPU 单进程 RSS; `_cuda` 后缀产物不进表. 峰值 = 阶段内 VmHWM 绝对高水位, 净增 = 峰值 - 阶段开始 RSS.
天花板 = MEMORY_BUDGET (45 GiB 峰值预算) / 最大成功 n 的端到端峰值 KB/dof, 是外推值, 以首个 OOM 的 n 为上界.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"

DEFAULT_MEMORY_TOTAL = 47.04 * 2**30  # WSL 来宾 MemTotal, 与 run.py 保持一致
MEMORY_BUDGET = 45 * 2**30  # 峰值内存预算, 与 run.py 保持一致
T_TET4 = 288
METHOD_ORDER = ("fast", "standard", "voigt")
STAGE2_ROUTE_ORDER = ("pattern", "coalesce", "scipy")
FULL_ROUTE_ORDER = ("pattern", "coalesce", "scipy")


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


def fmt_s(t: float | None) -> str:
    if t is None:
        return "--"
    return f"{t * 1000:.0f} ms" if t < 1 else f"{t:.1f} s"


def cpu_artifacts(pattern: str) -> list[tuple[Path, dict[str, Any]]]:
    """按 glob 收集 CPU 产物 (跳过 _cuda / _bform / .failed)."""
    items = []
    for p in sorted(_OUTPUT_DIR.glob(pattern)):
        if p.name.endswith(".failed.json") or "_cuda" in p.name or "_bform" in p.name:
            continue
        d = load_json(p)
        if d.get("device_type", "cpu") != "cpu":
            continue
        items.append((p, d))
    return items


def peak_kb_per_dof(d: dict[str, Any]) -> float | None:
    """端到端峰值 KB/dof; 兼容只含 final_peak_MiB 的旧产物."""
    if d.get("final_peak_KB_per_dof") is not None:
        return float(d["final_peak_KB_per_dof"])
    if d.get("final_peak_MiB") is not None and d.get("Ndof"):
        return d["final_peak_MiB"] * 2**20 / d["Ndof"] / 1000
    return None


# ----------------------------------------------------------------------------- 建网格
def show_mesh_table() -> None:
    items = cpu_artifacts("mesh_build_n*.json")
    print("\n[mesh] 只建网格与空间 (TetrahedronMesh.from_box + LagrangeFESpace/TensorFunctionSpace), CPU RSS")
    if not items:
        print("  (无产物; 需先跑 --case mesh-build --grid <N>)")
        return
    header = ["n", "NC", "Ndof", "建网格峰值", "空间峰值", "合计绝对峰值", "峰值 KB/dof", "构建后常驻", "常驻 KB/dof", "耗时"]
    rows = []
    ceilings = []
    for _, d in sorted(items, key=lambda kv: int(kv[1]["n"])):
        kb = peak_kb_per_dof(d)
        ceilings.append((int(d["n"]), kb))
        rows.append([
            str(d["n"]), f"{d.get('NC', 0):,}", f"{d.get('Ndof', 0):,}",
            fmt_mib(d.get("meshbuild_peak_MiB")), fmt_mib(d.get("space_peak_MiB")),
            fmt_mib(d.get("final_peak_MiB")), fmt_num(kb),
            fmt_mib(d.get("retained_MiB")), fmt_num(d.get("retained_KB_per_dof")),
            fmt_s(d.get("t_mesh_s")),
        ])
    print_table(header, rows, right_cols=set(range(1, len(header))))
    print("  峰值 = 阶段内 VmHWM 绝对高水位; 常驻 = 构建结束 gc 后相对进程基线的 RSS 净增 (只持有网格与空间的代价).")
    print(f"\n  {MEMORY_BUDGET / 2**30:.0f} GiB 预算下的公共上界外推 (只建网格, 不含任何装配):")
    for n, kb in ceilings:
        if not kb:
            continue
        ceiling = MEMORY_BUDGET / (kb * 1000)
        n_ceiling = round((ceiling / 3) ** (1 / 3)) - 1
        print(f"    n={n:<4}: 以 {kb:.2f} KB/dof 外推 -> 约 {ceiling / 1e6:.2f} M dof (n≈{n_ceiling})")
    print("  该上界与装配层级无关: FA/EA/PA/UA 都要先建出网格与空间, 任何层级都不可能突破它.")


# ----------------------------------------------------------------------------- 阶段 1
def show_stage1_table(n: int | None) -> None:
    items = cpu_artifacts("stage1_*_n*.json")
    by_n: dict[int, dict[str, dict]] = defaultdict(dict)
    for p, d in items:
        if "stage1_peak_MiB" not in d:
            continue  # 旧口径产物 (整条装配而非单刚), 不进表
        by_n[int(d["n"])][d["method"]] = d
    if n is not None:
        by_n = {k: v for k, v in by_n.items() if k == n}
    print("\n[stage1] 阶段 1 单刚 K_e = LinearElasticIntegrator.assembly(vs), CPU RSS")
    if not by_n:
        print("  (无新口径产物; 需先跑 --case element-stiffness --method all)")
        return
    header = ["n", "Ndof", "method", "阶段1绝对峰值", "净增", "净增 KB/dof", "相对 fast", "K_e 理论", "净增/理论", "耗时"]
    rows = []
    for nn in sorted(by_n):
        group = by_n[nn]
        base = group.get("fast", {}).get("stage1_KB_per_dof")
        for m in METHOD_ORDER:
            d = group.get(m)
            if d is None:
                continue
            kb = d.get("stage1_KB_per_dof")
            ratio = None if (base is None or kb is None or base == 0) else kb / base
            theory = d.get("Ke_theory_MiB")
            net = d.get("stage1_net_MiB")
            over = None if (theory in (None, 0) or net is None) else net / theory
            rows.append([
                str(nn), f"{d['Ndof']:,}", m, fmt_mib(d.get("stage1_peak_MiB")), fmt_mib(net),
                fmt_num(kb), fmt_num(ratio, 2, "x"), fmt_mib(theory), fmt_num(over, 2, "x"),
                fmt_s(d.get("t_stage1_s")),
            ])
    print_table(header, rows, right_cols={0, 1, 3, 4, 5, 6, 7, 8, 9})
    print("  峰值 = 阶段内 VmHWM (含网格与空间的常驻), 净增 = 峰值 - 阶段开始 RSS; 净增/理论 > 1 表示中间张量膨胀.")


# ----------------------------------------------------------------------------- 阶段 2
def show_stage2_table(n: int | None) -> None:
    items = cpu_artifacts("stage2_*_n*.json")
    by_n: dict[int, dict[str, dict]] = defaultdict(dict)
    for p, d in items:
        if "merge_peak_MiB" not in d:
            continue  # 旧口径产物, 不进表
        by_n[int(d["n"])][d["route"]] = d
    if n is not None:
        by_n = {k: v for k, v in by_n.items() if k == n}
    print("\n[stage2] 阶段 2 总刚合并 (合成输入, 与网格拓扑一致), CPU RSS")
    if not by_n:
        print("  (无新口径产物; 需先跑 --case global-merge --route all)")
        return
    header = ["n", "Ndof", "route", "合并绝对峰值", "合并净增", "B/triplet", "KB/dof", "含输入 B/triplet", "nnz", "耗时"]
    rows = []
    for nn in sorted(by_n):
        for r in STAGE2_ROUTE_ORDER:
            d = by_n[nn].get(r)
            if d is None:
                continue
            rows.append([
                str(nn), f"{d['Ndof']:,}", r,
                fmt_mib(d.get("merge_peak_MiB")), fmt_mib(d.get("merge_net_MiB")),
                fmt_num(d.get("merge_B_per_triplet")), fmt_num(d.get("merge_KB_per_dof")),
                fmt_num(d.get("merge_incl_inputs_B_per_triplet")), f"{d.get('nnz', 0):,}",
                fmt_s(d.get("t_merge_s")),
            ])
    print_table(header, rows, right_cols={0, 1, 3, 4, 5, 6, 7, 8, 9})
    print(
        "  B/triplet = 合并净增 / (144*NC); '含输入' 把 I/J/V 或 K_e 的生成也算进净增."
        " pattern = 生产 CSRPattern (build_csr_pattern + assemble_csr)."
    )


# ----------------------------------------------------------------------------- 端到端
_FAILED_RE = re.compile(r"^full_(?P<method>[a-z]+)_(?P<route>[a-z\-]+)_n(?P<n>\d+)\.failed\.json$")


def collect_full(method: str) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]:
    ok: dict[int, dict[str, dict]] = defaultdict(dict)
    failed: dict[int, dict[str, dict]] = defaultdict(dict)
    for p, d in cpu_artifacts(f"full_{method}_*_n*.json"):
        ok[int(d["n"])][d["route"]] = d
    for p in sorted(_OUTPUT_DIR.glob(f"full_{method}_*_n*.failed.json")):
        m = _FAILED_RE.match(p.name)
        if not m or "_cuda" in p.name:
            continue
        failed[int(m["n"])][m["route"]] = load_json(p)
    return ok, failed


def _ndof(n: int) -> int:
    return 3 * (n + 1) ** 3


def show_full_table(method: str = "fast") -> None:
    ok, failed = collect_full(method)
    all_n = sorted(set(ok) | set(failed))
    print(f"\n[full] 端到端真组装 method={method}: 绝对峰值 RSS (峰值 KB/dof), CPU 单进程")
    if not all_n:
        print("  (无产物; 需先跑 --case full-assembly --route all)")
        return
    header = ["n", "Ndof", *FULL_ROUTE_ORDER]
    rows = []
    for n in all_n:
        row = [str(n), f"{_ndof(n):,}"]
        for r in FULL_ROUTE_ORDER:
            d = ok.get(n, {}).get(r)
            f = failed.get(n, {}).get(r)
            if d is not None:
                cell = f"{fmt_mib(d.get('final_peak_MiB'))} ({fmt_num(peak_kb_per_dof(d))} KB/dof)"
            elif f is not None:
                seen = f.get("kernel_killed_anon_rss_MiB") or (f.get("observed") or {}).get("max_rss_MiB")
                cell = f"OOM @ {fmt_mib(seen)}" if f.get("suspected_oom") else f"失败 rc={f.get('returncode')}"
            else:
                cell = "--"
            row.append(cell)
        rows.append(row)
    print_table(header, rows, right_cols={0, 1, 2, 3, 4})

    # 阶段拆分 (仅新口径产物)
    detail_rows = []
    for n in all_n:
        for r in FULL_ROUTE_ORDER:
            d = ok.get(n, {}).get(r)
            if d is None or "stage2_peak_MiB" not in d:
                continue
            detail_rows.append([
                str(n), r,
                fmt_mib(d.get("stage1_net_MiB")), fmt_num(d.get("stage1_KB_per_dof")),
                fmt_mib(d.get("stage2_net_MiB")), fmt_num(d.get("stage2_KB_per_dof")),
                fmt_mib(d.get("symbolic_net_MiB")), fmt_mib(d.get("numeric_net_MiB")),
                fmt_mib(d.get("final_peak_MiB")), fmt_num(peak_kb_per_dof(d)),
                fmt_s(d.get("t_total_s")),
            ])
    if detail_rows:
        print("\n  阶段拆分 (净增 = 阶段峰值 - 阶段开始 RSS):")
        print_table(
            ["n", "route", "阶段1净增", "KB/dof", "阶段2净增", "KB/dof", "symbolic", "numeric", "绝对峰值", "峰值 KB/dof", "耗时"],
            detail_rows, right_cols={0, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )

    # 天花板外推
    print(f"\n  天花板外推 (峰值预算 {MEMORY_BUDGET / 2**30:.0f} GiB / 最大成功 n 的峰值 KB/dof; 机器 MemTotal {DEFAULT_MEMORY_TOTAL / 2**30:.2f} GiB):")
    for r in FULL_ROUTE_ORDER:
        succ = [(n, ok[n][r]) for n in all_n if r in ok.get(n, {})]
        if not succ:
            continue
        n_max, d = succ[-1]
        kb = peak_kb_per_dof(d)
        if not kb:
            continue
        ceiling = MEMORY_BUDGET / (kb * 1000)
        n_ceiling = round((ceiling / 3) ** (1 / 3)) - 1
        oom_ns = [n for n in all_n if r in failed.get(n, {}) and failed[n][r].get("suspected_oom")]
        bound = f", 上界 n < {min(oom_ns)} (Ndof {_ndof(min(oom_ns)):,}, 实测 OOM)" if oom_ns else ", 尚无 OOM 上界"
        print(
            f"    {r:<10}: 以 n={n_max} 的 {kb:.2f} KB/dof 外推 -> 约 {ceiling / 1e6:.2f} M dof (n≈{n_ceiling}){bound}"
        )
    print("  外推假设峰值 KB/dof 随 n 不变; 实测 KB/dof 通常随 n 略降 (边界占比减小), 外推偏保守.")

    # 生产入口一致性校验
    bform = [(p, load_json(p)) for p in sorted(_OUTPUT_DIR.glob(f"full_{method}_*_n*_bform.json")) if "_cuda" not in p.name]
    if bform:
        print("\n  生产入口校验 (--via-bilinearform vs 手拆阶段, 绝对峰值):")
        for p, d in bform:
            ref = ok.get(int(d["n"]), {}).get(d["route"])
            if ref is None:
                print(f"    {p.name}: 手拆产物缺失")
                continue
            a, b = d.get("final_peak_MiB"), ref.get("final_peak_MiB")
            diff = None if not (a and b) else 100.0 * (a - b) / b
            print(f"    n={d['n']} {d['route']:<9}: BilinearForm {fmt_mib(a)} vs 手拆 {fmt_mib(b)} ({fmt_num(diff, 1, '%')})")


# ----------------------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="all", help="mesh | stage1 | stage2 | full | all")
    parser.add_argument("-n", type=int, default=None, help="stage1/stage2 表只看指定 n")
    parser.add_argument("--method", default="fast", help="full 表的单刚方法 (默认 fast)")
    parser.add_argument("--list", action="store_true", help="列出 outputs/ 下的产物")
    args = parser.parse_args(argv)

    if args.list:
        for p in sorted(_OUTPUT_DIR.glob("*.json")):
            print(p.name)
        return 0

    case = args.case.lower()
    if case in ("mesh", "mesh-build", "all"):
        show_mesh_table()
    if case in ("stage1", "element-stiffness", "all"):
        show_stage1_table(args.n)
    if case in ("stage2", "global-merge", "all"):
        show_stage2_table(args.n)
    if case in ("full", "full-assembly", "all"):
        show_full_table(args.method)
    if case not in ("mesh", "mesh-build", "stage1", "element-stiffness", "stage2", "global-merge", "full", "full-assembly", "all"):
        parser.error(f"未知 --case {args.case!r}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
