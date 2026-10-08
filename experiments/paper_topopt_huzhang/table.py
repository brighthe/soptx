"""Hu--Zhang 拓扑优化投稿论文实验的表格入口 (论文 5.2 节).

在各 run 脚本的优化产物之上做冻结设计再分析, 再按论文表格的列排出 Markdown::

    python table.py --list
    python table.py compliance-reanalysis   # 固支梁 6x6 交叉再分析 (表 5.3)
    python table.py bearing-reanalysis      # 轴承 3x4 交叉再分析 (表 5.4)

再分析的完整结果 (含能量分量与自检) 照旧写入 ``results/<case>/postprocess/
frozen_reanalysis.json`` (轴承两组材料合在 ``results/bearing/postprocess/`` 一个文件里); 本入口只在其后把论文表格写到 ``results/tables/``.
子命令沿用语义名, 表号只出现在说明与输出文件名里.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any, Callable

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import config


def _reanalysis(case_id: str) -> dict[str, Any]:
    path = config.OUTPUT_DIR / case_id / "postprocess" / "frozen_reanalysis.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_name(label: str) -> str:
    method, _, order = label.rpartition("-")
    return f"LFEM $p={order}$" if method == "lfem" else f"HZMFEM $k={order}$"


def compliance_table() -> str:
    """表 5.3: 六个设计的迭代步数、优化所得柔顺度与 p=3,4 / k=3,4 再分析值 (完整结构)."""
    cross = _reanalysis("compliance-fixed-fixed-half")["cross"]
    factor = cross["full_structure_factor"]
    columns = ("lfem-3", "lfem-4", "huzhang-3", "huzhang-4")
    lines = [
        "**表 5.3**  两端固支梁六组最终设计的优化所得柔顺度与统一再分析柔顺度"
        "（完整结构，单位 N·mm；$p$ 为 LFEM 位移阶次，$k$ 为 HZMFEM 应力阶次）",
        "",
        "| 优化设计 | 迭代步数 | 优化所得柔顺度 | "
        + " | ".join(f"再分析 ${'p' if c.startswith('lfem') else 'k'}={c[-1]}$" for c in columns)
        + " |",
        "|:---:" * (3 + len(columns)) + "|",
    ]
    for design, info in cross["designs"].items():
        cells = [
            _paper_name(design),
            str(info["optimization_iterations"]),
            f"{factor * info['summary_compliance']:.2f}",
        ] + [f"{factor * cross['compliance'][design][c]:.2f}" for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _poisson(nu: float) -> str:
    """泊松比至少保留两位小数: 0.3 -> 0.30, 0.4999 -> 0.4999."""
    text = f"{nu:.6f}".rstrip("0")
    return text + "0" * max(0, 2 - len(text.partition(".")[2]))


def bearing_table() -> str:
    """表 5.4: 两组泊松比的三个设计, k=4 再分析值及其余离散相对它的偏差."""
    reference = "huzhang-4"
    deviations = ("lfem-1", "lfem-2", "huzhang-2")
    lines = [
        "**表 5.4**  二维轴承装置六组最终设计的优化迭代步数、优化所得柔顺度、"
        "HZMFEM $k=4$ 再分析柔顺度（单位 N·mm）及各离散再分析值相对后者的偏差",
        "",
        "| 泊松比 $\\nu_0$ | 优化设计 | 迭代步数 | 优化所得柔顺度 | HZMFEM $k=4$ 再分析柔顺度 | "
        + " | ".join(_paper_name(d) for d in deviations) + " |",
        "| :---: | :--- | :---: | :---: | :---: |" + " :---: |" * len(deviations),
    ]
    groups = _reanalysis("bearing")["groups"]
    for group in ("nu-0.3", "nu-0.4999"):
        cross = groups[group]
        nu = _poisson(cross["poisson_ratio"])
        for design, info in cross["designs"].items():
            row = cross["compliance"][design]
            cells = [
                f"${nu}$",
                _paper_name(design),
                str(info["optimization_iterations"]),
                f"${info['summary_compliance']:.2f}$",
                f"${row[reference]:.2f}$",
            ] + [f"${100.0 * (row[d] / row[reference] - 1.0):+.1f}\\%$" for d in deviations]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 子命令 -> (再分析入口 "模块:函数", 表格生成函数, 输出文件名, 说明)
TABLES: dict[str, tuple[str, Callable[[], str], str, str]] = {
    "compliance-reanalysis": (
        "analysis.compliance_reanalysis:run_compliance_reanalysis", compliance_table, "table5_3.md",
        "固支梁六个冻结设计 x 六种离散的柔顺度交叉再分析 (论文表 5.3)",
    ),
    "bearing-reanalysis": (
        "analysis.bearing_reanalysis:run_bearing_reanalysis", bearing_table, "table5_4.md",
        "轴承冻结设计 x 四种离散的柔顺度交叉再分析 (论文表 5.4)",
    ),
}


def run_table(command: str) -> int:
    """做再分析, 再把论文表格写到 ``config.TABLE_DIR``."""
    entry, render, filename, _ = TABLES[command]
    module_name, _, function_name = entry.partition(":")
    status = getattr(import_module(module_name), function_name)() or 0
    if status != 0:
        return status
    markdown = render()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    target = config.TABLE_DIR / filename
    target.write_text(markdown + "\n", encoding="utf-8")
    print("\n" + markdown + f"\n\n写入 {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="table.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="列出表格子命令")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for command, (_, _, _, description) in TABLES.items():
        sub.add_parser(command, help="[重分析] " + description)
    arguments = parser.parse_args(argv)
    if arguments.list:
        for command, (_, _, filename, description) in TABLES.items():
            print(f"{command:24s}{filename:14s}{description}")
        return 0
    if arguments.command in TABLES:
        return run_table(arguments.command)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
