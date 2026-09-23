# -*- coding: utf-8 -*-
"""两端固支梁算例的冻结设计交叉再分析: 6 设计 x 6 离散 (论文 5.2.1 节).

优化跑出的六个柔顺度 (LFEM p=2,3,4 与 Hu--Zhang k=2,3,4) 各在自身离散与自身泛函下
评价: LFEM 报外载功 f^T u, Hu--Zhang 报互补能 sigma^T A(rho) sigma, 且 k<=2 的跳量
稳定化格式互补能与牵引对偶功一般不相等. 六个数字的差异因此混有两种来源: (i) 设计
不同, (ii) 离散与泛函不同, 不能直接横比. 本模块把六个最终设计 (density_final.vtu)
冻结, 每个设计分别用六种离散重新求解一次, 得到按"设计"逐行、按"分析离散"逐列的
6x6 交叉表: 同一列内六个数字出自同一泛函, 其列内极差才是"设计相近"的定量表述.
只做前向求解, 不做优化.

入口由 ``compare.py`` 派发, 本模块不直接执行::

    compare.py compliance-reanalysis -> run_compliance_reanalysis()

产出写到 ``outputs/compliance-fixed-fixed-half/postprocess/frozen_reanalysis.json``,
带 provenance 戳记与六个 density_final.vtu 的 sha256. JSON 内柔顺度与能量均为半域
原值, 另记 ``full_structure_factor``; 终端 Markdown 表按论文口径给完整结构值
(半域 x 2). 每个 (设计, 分析) 组合同时记录 driver.energy_identity_diagnostics 的
能量分量 (LFEM: 外载功 / 应变能; Hu--Zhang: 互补能 / 耦合功 / 牵引对偶功), 供正文
说明两类泛函在离散层面的差别.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    CASES_FILE,
    OUTPUT_DIR,
    bootstrap_source_path,
    flatten_parameters,
    load_cases,
)

bootstrap_source_path()

from bearing_reanalysis import (  # noqa: E402
    SELF_CHECK_RTOL,
    _label,
    _run_dir,
    _split,
    load_design,
)
from driver import energy_identity_diagnostics  # noqa: E402
from pipeline import (  # noqa: E402
    build_fixed_fixed_analysis_pipeline,
    build_fixed_fixed_config,
)
import provenance  # noqa: E402

CASE = "compliance-fixed-fixed-half"

# 六种离散 = 论文 5.2.1 节正文对比组 (cases.toml comparison_orders = 2/3/4 x 两条链);
# 既是设计来源 (行) 也是再分析离散 (列).
DISCRETIZATIONS: tuple[tuple[str, int], ...] = (
    ("lfem", 2), ("lfem", 3), ("lfem", 4),
    ("huzhang", 2), ("huzhang", 3), ("huzhang", 4),
)

# 论文正文摘出的两列: 两条链各自的最高阶, 作为"统一泛函"给六个设计打分.
PAPER_COLUMNS: tuple[str, ...] = ("lfem-4", "huzhang-4")


def case_record(case_id: str) -> dict[str, Any]:
    for case in load_cases(CASES_FILE):
        if case["id"] == case_id:
            return case
    raise SystemExit(f"cases.toml 中没有算例 {case_id}.")


def build_pipeline(case_id: str, method: str, order: int):
    """按 cases.toml 口径组装一条分析链 (不含优化器), 与 run.py 的优化运行同参数.

    模型名取自 [cases.model] name (左半域对称降维模型), 载荷走 P1 迹 L2 投影,
    与优化运行完全一致, 故对角线自检可复现 summary.json 的 compliance.
    """
    case = case_record(case_id)
    parameters = flatten_parameters(case)
    config = build_fixed_fixed_config(parameters)
    pipeline = build_fixed_fixed_analysis_pipeline(
        config, parameters, method, order, case["model"]["name"]
    )
    return pipeline, config


def frozen_solve(pipeline, rho_np: np.ndarray) -> tuple[float, dict[str, Any], float]:
    """把冻结密度写进分析链, 前向求解一次, 返回 (柔顺度, 能量分量, 耗时)."""
    rho = pipeline.density_distribution
    if rho.shape[0] != rho_np.shape[0]:
        raise ValueError(f"网格单元数 {rho.shape[0]} 与构型 {rho_np.shape[0]} 不符.")
    rho[:] = rho_np
    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    compliance = float(pipeline.objective.fun(density=rho, state=state))
    energy = {
        key: (float(value) if not isinstance(value, str) else value)
        for key, value in energy_identity_diagnostics(pipeline, state).items()
    }
    return compliance, energy, time.perf_counter() - started


# ============================================ 一、交叉表: 6 设计 x 6 离散

def cross_table(case_id: str = CASE) -> dict[str, Any]:
    designs = {_label(m, o): load_design(case_id, m, o) for m, o in DISCRETIZATIONS}
    factors = {float(summary.get("full_structure_factor", 1.0)) for _, summary in designs.values()}
    if len(factors) != 1:
        raise SystemExit(f"{case_id} 六个运行的 full_structure_factor 不一致: {sorted(factors)}.")
    factor = factors.pop()

    table: dict[str, dict[str, float]] = {name: {} for name in designs}
    energy: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in designs}
    self_check: dict[str, dict[str, Any]] = {}
    nu = None
    for method, order in DISCRETIZATIONS:
        analysis = _label(method, order)
        pipeline, config = build_pipeline(case_id, method, order)
        nu = float(config.poisson_ratio)
        for design, (rho, summary) in designs.items():
            compliance, diagnostics, elapsed = frozen_solve(pipeline, rho)
            table[design][analysis] = compliance
            energy[design][analysis] = diagnostics
            note = ""
            if design == analysis:
                # 对角线自检不中止: 设计是在历史代码版本下优化出来的, 当前代码的同名
                # 离散若已改变 (如 k<=GD 的跳量稳定化定标), 自检差就是这处改动的量度,
                # 记进 JSON 供判断, 表照样出全; 通过与否由 SELF_CHECK_RTOL 判定.
                reference = float(summary["compliance"])
                rel = abs(compliance - reference) / abs(reference)
                passed = rel <= SELF_CHECK_RTOL
                self_check[design] = {
                    "summary_compliance": reference,
                    "reanalysis_compliance": compliance,
                    "rel_diff": rel,
                    "passed": passed,
                }
                note = f"  summary={reference:.6f} rel_diff={rel:.2e}"
                if not passed:
                    note += f"  !! 自检未通过 (> {SELF_CHECK_RTOL}): 当前代码的 {analysis} 离散与优化运行时不同"
            print(f"[cross] {case_id} design={design:10s} analysis={analysis:10s}: "
                  f"C_half={compliance:.4f} C_full={factor * compliance:.4f} "
                  f"({elapsed:.1f}s){note}", flush=True)
    return {
        "poisson_ratio": nu,
        "full_structure_factor": factor,
        "self_check": self_check,
        "designs": {
            name: {
                "run_dir": str(_run_dir(case_id, *_split(name)).relative_to(OUTPUT_DIR)),
                "optimization_iterations": summary.get("optimization_iterations"),
                "converged": summary.get("converged"),
                "volume_fraction": summary.get("volume_fraction"),
                "summary_compliance": float(summary["compliance"]),
                "density_final": provenance.file_digest(
                    _run_dir(case_id, *_split(name)) / "density_final.vtu"),
            }
            for name, (_, summary) in designs.items()
        },
        "compliance": table,
        "energy": energy,
    }


# ============================================ 二、Markdown 表与落盘

def _column_spread(block: dict[str, Any], analysis: str) -> tuple[float, float, float]:
    """某一分析列的 (最小值, 最大值, 极差/最小值)."""
    values = [row[analysis] for row in block["compliance"].values()]
    low, high = min(values), max(values)
    return low, high, (high - low) / low


def _relative(value: float, reference: float) -> str:
    return f"{100.0 * (value / reference - 1.0):+.2f}%"


def _paper(method: str, order: int) -> str:
    return f"LFEM p={order}" if method == "lfem" else f"HZ k={order}"


def print_cross_markdown(block: dict[str, Any]) -> None:
    labels = [_label(m, o) for m, o in DISCRETIZATIONS]
    factor = block["full_structure_factor"]
    print(f"\n{CASE}: nu={block['poisson_ratio']}, 完整结构柔顺度 (半域 x {factor:g}); "
          "行: 优化设计, 列: 再分析离散; 对角线 = 自评值")
    print("| 设计 \\ 分析 | " + " | ".join(_paper(*_split(a)) for a in labels) + " |")
    print("|---|" + "---|" * len(labels))
    for design, row in block["compliance"].items():
        iters = block["designs"][design]["optimization_iterations"]
        cells = " | ".join(
            (f"**{factor * row[a]:.2f}**" if a == design else f"{factor * row[a]:.2f}")
            for a in labels
        )
        print(f"| {_paper(*_split(design))} ({iters} 步) | {cells} |")
    spreads = " | ".join(f"{100.0 * _column_spread(block, a)[2]:.2f}%" for a in labels)
    print(f"| 列内极差 (max/min - 1) | {spreads} |")


def print_paper_markdown(block: dict[str, Any]) -> None:
    factor = block["full_structure_factor"]
    print("\n论文表 (5.2.1 节): 自评柔顺度 vs 统一泛函再分析")
    header = "| 设计 | 迭代步数 | 自评 C |"
    for column in PAPER_COLUMNS:
        header += f" {_paper(*_split(column))} 再分析 | 相对列内最小 |"
    print(header)
    print("|---|---|---|" + "---|---|" * len(PAPER_COLUMNS))
    minima = {column: _column_spread(block, column)[0] for column in PAPER_COLUMNS}
    for design, row in block["compliance"].items():
        info = block["designs"][design]
        line = (f"| {_paper(*_split(design))} | {info['optimization_iterations']} | "
                f"{factor * info['summary_compliance']:.2f} |")
        for column in PAPER_COLUMNS:
            line += f" {factor * row[column]:.2f} | {_relative(row[column], minima[column])} |"
        print(line)
    self_values = [factor * info["summary_compliance"] for info in block["designs"].values()]
    print(f"\n自评值极差: {100.0 * (max(self_values) / min(self_values) - 1.0):.2f}% "
          f"({min(self_values):.2f} ~ {max(self_values):.2f})")
    for column in PAPER_COLUMNS:
        low, high, spread = _column_spread(block, column)
        print(f"{_paper(*_split(column))} 列内极差: {100.0 * spread:.2f}% "
              f"({factor * low:.2f} ~ {factor * high:.2f})")


def print_energy_markdown(block: dict[str, Any]) -> None:
    """Hu--Zhang 各列: 互补能 (= 再分析 C) 与牵引对偶功的差, 看泛函差别随 k 的走势."""
    factor = block["full_structure_factor"]
    hz = [_label(m, o) for m, o in DISCRETIZATIONS if m == "huzhang"]
    print("\nHu--Zhang 列能量分量 (完整结构): 互补能 sigma^T A sigma / 牵引对偶功 / 相对耦合比")
    print("| 设计 | " + " | ".join(f"{_paper(*_split(a))} 互补能 | 对偶功 | 耦合比" for a in hz) + " |")
    print("|---|" + "---|---|---|" * len(hz))
    for design, row in block["energy"].items():
        cells = []
        for analysis in hz:
            e = row[analysis]
            cells.append(f"{factor * e['complementary_energy']:.2f} | "
                         f"{factor * e['traction_dual_work']:.2f} | "
                         f"{100.0 * e['relative_coupling_ratio']:.2f}%")
        print(f"| {_paper(*_split(design))} | " + " | ".join(cells) + " |")


def write_json(payload: dict[str, Any], case_id: str = CASE) -> Path:
    target = OUTPUT_DIR / case_id / "postprocess" / "frozen_reanalysis.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def run_compliance_reanalysis() -> int:
    stamp = provenance.run_stamp()
    result = {
        "case_id": CASE,
        "discretizations": [list(d) for d in DISCRETIZATIONS],
        "paper_columns": list(PAPER_COLUMNS),
        "provenance": stamp,
        "cross": cross_table(CASE),
    }
    print_cross_markdown(result["cross"])
    print_paper_markdown(result["cross"])
    print_energy_markdown(result["cross"])
    print()
    failed = {name: check for name, check in result["cross"]["self_check"].items()
              if not check["passed"]}
    for name, check in failed.items():
        print(f"自检未通过: {name} 再分析 {check['reanalysis_compliance']:.6f} 与 summary.json "
              f"{check['summary_compliance']:.6f} 相对差 {check['rel_diff']:.2e}; "
              f"该列出自当前代码的离散, 与优化运行时的离散不同, 不可与自评值混用.")
    print(f"写入 {write_json(result)}")
    if not stamp.get("reproducible"):
        print("注意: 工作区不干净, 本次数字不满足 provenance.reproducible, 不可直接引用.")
    return 0
