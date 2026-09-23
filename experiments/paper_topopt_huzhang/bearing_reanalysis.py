# -*- coding: utf-8 -*-
"""轴承算例的冻结设计再分析: 交叉表、nu 扫描与全实体域闭锁考察 (论文图 5.5、表 5.4).

优化跑出的柔顺度只在各自离散下可比: 低阶位移元在近不可压缩材料上因体积闭锁
低估柔顺度, 不同离散优化出的设计不能直接横比. 本模块做三件事, 全部只做前向求解,
不做优化:

1. 交叉表: 把每条轴承 case 的三个最终设计 (density_final.vtu) 冻结, 分别用五种
   离散 (三种参赛离散 lfem p=1 / lfem p=2 / Hu--Zhang k=2, 加两列参考离散 lfem p=4 /
   Hu--Zhang k=4) 重新求解一次柔顺度, 得到按"设计"逐行的交叉表. 参考列的作用是
   给每个设计一个与参赛离散无关的柔顺度基准, 使 p=1 的闭锁、p=2 与 k=2 的残余
   离散误差各自可量化, 而不是互相为分母.
2. nu 扫描: 把近不可压缩组的 Hu--Zhang k=2 设计冻结, 在 6 档 nu 下用五种离散求解,
   看闭锁偏差随 nu -> 0.5 的走势. 设计含中间密度, nu 插值随材料是否近不可压缩切换.
3. 全实体域考察: rho == 1, 无任何材料插值, 同样 6 档 nu 与五种离散. 这是
   Bruggi (2016) 5.1 节固定结构闭锁考察的移植, 把插值切换从闭锁判断中剔除,
   并给出 C_0 (全域实体) 供归一化.

入口由 ``compare.py`` 派发, 本模块不直接执行::

    compare.py bearing-reanalysis -> run_bearing_reanalysis()

产出写到 ``outputs/<case>/postprocess/frozen_reanalysis.json``, 带 provenance 戳记
与三个 density_final.vtu 的 sha256; nu 扫描与全实体域考察挂在近不可压缩组的 JSON 下.
终端同时打印四张 Markdown 表, 偏差列一律相对 Hu--Zhang k=4 参考值.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import (
    CASES_FILE,
    OUTPUT_DIR,
    bootstrap_source_path,
    flatten_parameters,
    load_cases,
)

bootstrap_source_path()

from pipeline import (  # noqa: E402
    build_bearing_analysis_pipeline,
    build_bearing_config,
    build_bearing_problem,
    build_material,
    resolve_interpolation_variables,
)
import provenance  # noqa: E402

CASES = ("bearing-compressible", "bearing-incompressible")
SWEEP_CASE = "bearing-incompressible"
SWEEP_DESIGN = ("huzhang", 2)

# 参赛离散 (各自跑过优化, 有 density_final.vtu): lfem p=1 (supplementary_orders) /
# lfem p=2 / Hu--Zhang k=2 (跳量稳定化, 位移分片 P1)
DESIGNS: tuple[tuple[str, int], ...] = (("lfem", 1), ("lfem", 2), ("huzhang", 2))
# 参考离散 (只做再分析, 不跑优化): 两族各取 4 阶, 近不可压缩组已登记 comparison_orders 4
REFERENCES: tuple[tuple[str, int], ...] = (("lfem", 4), ("huzhang", 4))
ANALYSES: tuple[tuple[str, int], ...] = DESIGNS + REFERENCES
# 偏差列的分母
REFERENCE_LABEL = "huzhang-4"

# nu 扫描档位; nu < 0.49 时 E+nu 插值被 resolve_interpolation_variables 拒绝,
# 故扫描一律用 "auto" (is_incompressible 时 E+nu, 否则只插值 E), 实际生效值记入输出.
NU_SWEEP: tuple[float, ...] = (0.3, 0.45, 0.49, 0.499, 0.4999, 0.49999)

# 自检容差: 设计用自身离散再分析必须复现 summary.json 的 compliance
SELF_CHECK_RTOL = 1e-8


def _label(method: str, order: int) -> str:
    return f"{method}-{order}"


def _split(label: str) -> tuple[str, int]:
    method, _, order = label.rpartition("-")
    return method, int(order)


def _run_dir(case_id: str, method: str, order: int) -> Path:
    """注册缺省组合的运行目录 (run.py 不给覆盖时的目录名, 不带 optimizer 标签)."""
    return OUTPUT_DIR / case_id / f"analyzer-{method}__order-{order}"


def case_parameters(case_id: str) -> dict[str, Any]:
    for case in load_cases(CASES_FILE):
        if case["id"] == case_id:
            return flatten_parameters(case)
    raise SystemExit(f"cases.toml 中没有算例 {case_id}.")


def load_design(case_id: str, method: str, order: int) -> tuple[np.ndarray, dict[str, Any]]:
    run_dir = _run_dir(case_id, method, order)
    density_file = run_dir / "density_final.vtu"
    summary_file = run_dir / "summary.json"
    for path in (density_file, summary_file):
        if not path.is_file():
            raise SystemExit(f"缺少 {path}; 先运行 run.py --case {case_id} "
                             f"--analyzer {method} --order {order}.")
    rho = np.asarray(read_vtu_cell_data(density_file, "density"), dtype=np.float64)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return rho, summary


def build_pipeline(case_id: str, method: str, order: int, nu: float | None = None):
    """按 cases.toml 口径组装分析链; 给定 nu 时覆盖泊松比并把插值对象交给 auto 判定.

    返回 (分析链, 配置, 实际生效的插值对象): 配置里的 ``auto`` 要按材料是否近
    不可压缩落成 ``E`` 或 ``E+nu``, 写进产出的是落成后的值.
    """
    parameters = case_parameters(case_id)
    parameters["comparison_orders"] = [order]
    if nu is not None:
        parameters["poisson_ratio"] = float(nu)
        parameters["interpolation_variables"] = "auto"
    config = build_bearing_config(parameters)
    pipeline = build_bearing_analysis_pipeline(config, parameters, method, order)
    effective = resolve_interpolation_variables(
        build_material(build_bearing_problem(parameters)), config.interpolation_variables
    )
    return pipeline, config, effective


def frozen_compliance(pipeline, rho_np: np.ndarray) -> tuple[float, float]:
    rho = pipeline.density_distribution
    if rho.shape[0] != rho_np.shape[0]:
        raise ValueError(f"网格单元数 {rho.shape[0]} 与构型 {rho_np.shape[0]} 不符.")
    rho[:] = rho_np
    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    compliance = float(pipeline.objective.fun(density=rho, state=state))
    return compliance, time.perf_counter() - started


# ============================================ 一、交叉表: 每条 case 3 设计 x 5 离散

def cross_table(case_id: str) -> dict[str, Any]:
    designs = {_label(m, o): load_design(case_id, m, o) for m, o in DESIGNS}
    table: dict[str, dict[str, float]] = {name: {} for name in designs}
    interpolation = None
    nu = None
    for method, order in ANALYSES:
        analysis = _label(method, order)
        pipeline, config, interpolation = build_pipeline(case_id, method, order)
        nu = float(config.poisson_ratio)
        for design, (rho, summary) in designs.items():
            compliance, elapsed = frozen_compliance(pipeline, rho)
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


# ============================================ 二、nu 扫描: 冻结设计 / 全实体域

def _sweep_rows(case_id: str, rho: np.ndarray, tag: str) -> list[dict[str, Any]]:
    rows = []
    for nu in NU_SWEEP:
        row: dict[str, Any] = {"poisson_ratio": nu, "compliance": {}}
        for method, order in ANALYSES:
            analysis = _label(method, order)
            pipeline, _, effective = build_pipeline(case_id, method, order, nu=nu)
            row["interpolation_variables"] = effective
            compliance, elapsed = frozen_compliance(pipeline, rho)
            row["compliance"][analysis] = compliance
            print(f"[{tag}] nu={nu:<8} interp={effective:5s} "
                  f"analysis={analysis}: C={compliance:.4f} ({elapsed:.1f}s)", flush=True)
        rows.append(row)
    return rows


def nu_sweep(case_id: str = SWEEP_CASE) -> dict[str, Any]:
    """冻结近不可压缩组 Hu--Zhang k=2 设计, 6 档 nu x 五种离散."""
    rho, _ = load_design(case_id, *SWEEP_DESIGN)
    return {"design": _label(*SWEEP_DESIGN), "rows": _sweep_rows(case_id, rho, "nu")}


def solid_sweep(case_id: str = SWEEP_CASE) -> dict[str, Any]:
    """全实体域 rho == 1, 无材料插值, 6 档 nu x 五种离散 (Bruggi 2016 5.1 节式闭锁考察)."""
    rho, _ = load_design(case_id, *SWEEP_DESIGN)
    solid = np.ones_like(rho)
    return {"design": "solid", "rows": _sweep_rows(case_id, solid, "solid")}


# ============================================ 三、Markdown 表与落盘

def _deviation(value: float, reference: float) -> str:
    return f"{100.0 * (value / reference - 1.0):+.2f}%"


def _deviation_header() -> str:
    return " | ".join(f"{_label(m, o)} 偏差" for m, o in DESIGNS) + " | lfem-4 偏差"


def _deviation_cells(c: dict[str, float]) -> str:
    ref = c[REFERENCE_LABEL]
    cells = [_deviation(c[_label(m, o)], ref) for m, o in DESIGNS]
    cells.append(_deviation(c["lfem-4"], ref))
    return " | ".join(cells)


def print_cross_markdown(case_id: str, block: dict[str, Any]) -> None:
    labels = [_label(m, o) for m, o in ANALYSES]
    print(f"\n{case_id}: nu={block['poisson_ratio']}, 插值 {block['interpolation_variables']} "
          f"(行: 优化设计, 列: 再分析离散, 偏差相对 {REFERENCE_LABEL})")
    print("| 设计 \\ 分析 | " + " | ".join(labels) + " | " + _deviation_header() + " |")
    print("|---|" + "---|" * (len(labels) + len(DESIGNS) + 1))
    for design, row in block["compliance"].items():
        iters = block["designs"][design]["optimization_iterations"]
        cells = " | ".join(f"{row[a]:.2f}" for a in labels)
        print(f"| {design} ({iters} 步) | {cells} | {_deviation_cells(row)} |")


def print_sweep_markdown(title: str, block: dict[str, Any]) -> None:
    labels = [_label(m, o) for m, o in ANALYSES]
    print(f"\n{title} (设计 {block['design']}, 偏差相对 {REFERENCE_LABEL})")
    print("| nu | 插值 | " + " | ".join(labels) + " | " + _deviation_header() + " |")
    print("|---|---|" + "---|" * (len(labels) + len(DESIGNS) + 1))
    for row in block["rows"]:
        c = row["compliance"]
        cells = " | ".join(f"{c[a]:.2f}" for a in labels)
        print(f"| {row['poisson_ratio']} | {row['interpolation_variables']} | {cells} | "
              f"{_deviation_cells(c)} |")


def write_json(case_id: str, payload: dict[str, Any]) -> Path:
    target = OUTPUT_DIR / case_id / "postprocess" / "frozen_reanalysis.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def run_bearing_reanalysis() -> int:
    stamp = provenance.run_stamp()
    results: dict[str, dict[str, Any]] = {}
    for case_id in CASES:
        results[case_id] = {
            "case_id": case_id,
            "designs": [list(d) for d in DESIGNS],
            "analyses": [list(d) for d in ANALYSES],
            "reference": REFERENCE_LABEL,
            "provenance": stamp,
            "cross": cross_table(case_id),
        }
    results[SWEEP_CASE]["nu_sweep"] = nu_sweep(SWEEP_CASE)
    results[SWEEP_CASE]["solid_sweep"] = solid_sweep(SWEEP_CASE)

    for case_id in CASES:
        print_cross_markdown(case_id, results[case_id]["cross"])
    print_sweep_markdown(f"nu 扫描 (冻结 {SWEEP_CASE} 设计)", results[SWEEP_CASE]["nu_sweep"])
    print_sweep_markdown("全实体域 nu 扫描 (rho == 1, 无插值)", results[SWEEP_CASE]["solid_sweep"])

    print()
    for case_id in CASES:
        print(f"写入 {write_json(case_id, results[case_id])}")
    if not stamp.get("reproducible"):
        print("注意: 工作区不干净, 本次数字不满足 provenance.reproducible, 不可直接引用.")
    return 0
