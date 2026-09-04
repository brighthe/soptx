# -*- coding: utf-8 -*-
"""汇总 outputs/ 下各工况的运行结果, 生成一张紧凑的验收表.

只读已落盘的 summary.json / history.json, 不触发任何计算。

用法:
    python collect.py
    python collect.py --json outputs/collected.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import OUTPUT_DIR, ConfigError, load


def _row(case: Any, output_root: Path) -> Dict[str, Any]:
    directory = output_root / case.id
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return {
            "case_id": case.id,
            "trace": case.trace,
            "status": "missing",
            "detail": f"未运行 (缺 {summary_path.relative_to(output_root.parent)})",
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "case_id": case.id,
        "trace": summary.get("trace"),
        "reduction": summary.get("reduction"),
        "status": "ok",
        "dimension": summary.get("dimension"),
        "grid": summary.get("grid"),
        "n_substructures": summary.get("n_substructures"),
        "n_cells": summary.get("n_cells"),
        "n_global_trace_dofs": summary.get("n_global_trace_dofs"),
        "iterations": summary.get("iterations"),
        "converged": summary.get("converged"),
        "final_compliance": summary.get("final_compliance"),
        "final_volume_fraction": summary.get("final_volume_fraction"),
        "mean_iteration_time": summary.get("mean_iteration_time"),
        "total_time": summary.get("total_time"),
        "fa_reference": summary.get("fa_reference"),
        "git_head": (summary.get("provenance") or {}).get("git", {}).get("head"),
    }


def collect(output_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    _meta, cases = load()
    root = output_root or OUTPUT_DIR
    return [_row(case, root) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总子结构缩聚拓扑优化结果")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="覆盖输出根目录 (默认 outputs/)"
    )
    parser.add_argument("--json", type=Path, default=None, help="把汇总写入 JSON 文件")
    arguments = parser.parse_args()

    try:
        rows = collect(arguments.output_dir)
    except ConfigError as error:
        print(f"[error] {error}")
        return 2

    header = (
        f"{'case_id':24s} {'trace':14s} {'iters':>6} {'converged':>10} "
        f"{'compliance':>14} {'volfrac':>9} {'mean_it/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['case_id']:24s} {str(row['trace']):14s} {row['detail']}")
            continue
        print(
            f"{row['case_id']:24s} {str(row['trace']):14s} "
            f"{row['iterations']:6d} {str(row['converged']):>10s} "
            f"{row['final_compliance']:14.6f} "
            f"{row['final_volume_fraction']:9.4f} "
            f"{row['mean_iteration_time']:10.3f}"
        )

    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[collect] 汇总已写入 {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
