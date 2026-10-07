# -*- coding: utf-8 -*-
"""轴承算例的冻结设计再分析: 交叉表 (论文表 5.4).

优化跑出的柔顺度只在各自离散下可比: 低阶位移元在近不可压缩材料上因体积闭锁
低估柔顺度, 不同离散优化出的设计不能直接横比. 本模块只做前向求解, 不做优化:
把每组材料 (nu = 0.3 / 0.4999) 的三个最终设计 (density_final.vtu) 冻结, 分别用四种离散 (三种
参赛离散 lfem p=1 / lfem p=2 / Hu--Zhang k=2, 加一列参考离散 Hu--Zhang k=4)
重新求解一次柔顺度, 得到按"设计"逐行的交叉表. 参考列的作用是
给每个设计一个与参赛离散无关的柔顺度基准, 使 p=1 的闭锁、p=2 与 k=2 的残余
离散误差各自可量化, 而不是互相为分母.

入口由 ``table.py`` 派发, 本模块不直接执行::

    table.py bearing-reanalysis -> run_bearing_reanalysis()

分析链由 ``run_bearing.build`` 组装, 与优化运行同一份代码. 两组材料的交叉表写进同一个
``results/bearing/postprocess/frozen_reanalysis.json``, 带 provenance 戳记与六个
density_final.vtu 的 sha256. 终端同时打印两张 Markdown 表, 偏差列一律相对 Hu--Zhang
k=4 参考值.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import OUTPUT_DIR, bootstrap_source_path

bootstrap_source_path()

import provenance  # noqa: E402
import run_bearing  # noqa: E402

CASE = run_bearing.CASE_ID
GROUPS = tuple(run_bearing.GROUPS)

# 参赛离散 (各自跑过优化, 有 density_final.vtu): lfem p=1 /
# lfem p=2 / Hu--Zhang k=2 (跳量稳定化, 位移分片 P1)
DESIGNS: tuple[tuple[str, int], ...] = (("lfem", 1), ("lfem", 2), ("huzhang", 2))
# 参考离散 (只做再分析, 不跑优化): Hu--Zhang k=4
REFERENCES: tuple[tuple[str, int], ...] = (("huzhang", 4),)
ANALYSES: tuple[tuple[str, int], ...] = DESIGNS + REFERENCES
# 偏差列的分母
REFERENCE_LABEL = "huzhang-4"

# 自检容差: 设计用自身离散再分析必须复现 summary.json 的 compliance
SELF_CHECK_RTOL = 1e-8


def _label(method: str, order: int) -> str:
    return f"{method}-{order}"


def _split(label: str) -> tuple[str, int]:
    method, _, order = label.rpartition("-")
    return method, int(order)


def _run_dir(case_id: str, method: str, order: int) -> Path:
    """运行目录; ``case_id`` 可含子目录, 如轴承的 ``bearing/nu-0.3``."""
    return OUTPUT_DIR / case_id / f"analyzer-{method}__order-{order}"


def load_design(case_id: str, method: str, order: int) -> tuple[np.ndarray, dict[str, Any]]:
    run_dir = _run_dir(case_id, method, order)
    density_file = run_dir / "density_final.vtu"
    summary_file = run_dir / "summary.json"
    for path in (density_file, summary_file):
        if not path.is_file():
            raise SystemExit(f"缺少 {path}; 先运行对应算例的 run 脚本 "
                             f"(--analyzer {method} --order {order}).")
    rho = np.asarray(read_vtu_cell_data(density_file, "density"), dtype=np.float64)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return rho, summary


def frozen_compliance(parts: dict[str, Any], rho_np: np.ndarray) -> tuple[float, float]:
    """把冻结密度写进 ``run_bearing.build`` 组装的分析链, 前向求解一次."""
    rho = parts["density"]
    if rho.shape[0] != rho_np.shape[0]:
        raise ValueError(f"网格单元数 {rho.shape[0]} 与构型 {rho_np.shape[0]} 不符.")
    rho[:] = rho_np
    started = time.perf_counter()
    state = parts["analyzer"].solve_state(rho_val=rho)
    compliance = float(parts["objective"].fun(density=rho, state=state))
    return compliance, time.perf_counter() - started


# ============================================ 一、交叉表: 每组材料 3 设计 x 4 离散

def cross_table(group: str) -> dict[str, Any]:
    case_id = f"{CASE}/{group}"
    nu, interpolation = run_bearing.GROUPS[group]
    designs = {_label(m, o): load_design(case_id, m, o) for m, o in DESIGNS}
    table: dict[str, dict[str, float]] = {name: {} for name in designs}
    for method, order in ANALYSES:
        analysis = _label(method, order)
        parts = run_bearing.build(group, method, order)
        for design, (rho, summary) in designs.items():
            compliance, elapsed = frozen_compliance(parts, rho)
            table[design][analysis] = compliance
            note = ""
            if design == analysis:
                reference = float(summary["compliance"])
                rel = abs(compliance - reference) / abs(reference)
                note = f"  summary={reference:.6f} rel_diff={rel:.2e}"
                if rel > SELF_CHECK_RTOL:
                    raise SystemExit(
                        f"{case_id} {design} 自检失败: 再分析 {compliance:.8f} 与 "
                        f"summary.json {reference:.8f} 相对差 {rel:.2e} > {SELF_CHECK_RTOL}."
                    )
            print(f"[cross] {case_id} nu={nu} design={design:10s} analysis={analysis}: "
                  f"C={compliance:.4f} ({elapsed:.1f}s){note}", flush=True)
    return {
        "poisson_ratio": nu,
        "interpolation_variables": interpolation,
        "designs": {
            name: {
                "run_dir": str(_run_dir(case_id, *_split(name)).relative_to(OUTPUT_DIR)),
                "optimization_iterations": summary.get("optimization_iterations"),
                "converged": summary.get("converged"),
                "summary_compliance": float(summary["compliance"]),
                "density_final": provenance.file_digest(
                    _run_dir(case_id, *_split(name)) / "density_final.vtu"),
            }
            for name, (_, summary) in designs.items()
        },
        "compliance": table,
    }


# ============================================ 二、Markdown 表与落盘

def _deviation(value: float, reference: float) -> str:
    return f"{100.0 * (value / reference - 1.0):+.2f}%"


def _deviation_header() -> str:
    return " | ".join(f"{_label(m, o)} 偏差" for m, o in DESIGNS)


def _deviation_cells(c: dict[str, float]) -> str:
    ref = c[REFERENCE_LABEL]
    return " | ".join(_deviation(c[_label(m, o)], ref) for m, o in DESIGNS)


def print_cross_markdown(group: str, block: dict[str, Any]) -> None:
    labels = [_label(m, o) for m, o in ANALYSES]
    print(f"\n{CASE}/{group}: nu={block['poisson_ratio']}, 插值 {block['interpolation_variables']} "
          f"(行: 优化设计, 列: 再分析离散, 偏差相对 {REFERENCE_LABEL})")
    print("| 设计 \\ 分析 | " + " | ".join(labels) + " | " + _deviation_header() + " |")
    print("|---|" + "---|" * (len(labels) + len(DESIGNS)))
    for design, row in block["compliance"].items():
        iters = block["designs"][design]["optimization_iterations"]
        cells = " | ".join(f"{row[a]:.2f}" for a in labels)
        print(f"| {design} ({iters} 步) | {cells} | {_deviation_cells(row)} |")


def write_json(payload: dict[str, Any]) -> Path:
    target = OUTPUT_DIR / CASE / "postprocess" / "frozen_reanalysis.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def run_bearing_reanalysis() -> int:
    stamp = provenance.run_stamp()
    payload = {
        "case_id": CASE,
        "designs": [list(d) for d in DESIGNS],
        "analyses": [list(d) for d in ANALYSES],
        "reference": REFERENCE_LABEL,
        "provenance": stamp,
        "groups": {group: cross_table(group) for group in GROUPS},
    }
    for group, block in payload["groups"].items():
        print_cross_markdown(group, block)
    print(f"\n写入 {write_json(payload)}")
    if not stamp.get("reproducible"):
        print("注意: 工作区不干净, 本次数字不满足 provenance.reproducible, 不可直接引用.")
    return 0
