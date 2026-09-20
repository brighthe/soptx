# -*- coding: utf-8 -*-
"""悬臂梁应力算例的冻结设计交叉评价: 一份构型, 七条离散.

优化跑出来的应力比只在各自的最终构型上有意义, 跨构型不可直接横比: Hu--Zhang k=2
在载荷贴片端点外侧单元上的实体应力比 4.31 与 LFEM k=2 的 3.94 差在哪里 —— 是两条
离散对奇异点的分辨能力不同, 还是两份构型在该单元的密度本就不同 —— 单看各自的
summary.json 分不开. 本模块把一份构型冻结, 用 LFEM k=1..4 与 Hu--Zhang k=2..4 各
前向求解一次, 在同一个密度场上读同一批单元, 把"离散"这个变量单独隔离出来.

判据的代数 (epsilon 松弛的表观应力约束, MSIMP 插值)::

    m_E(rho) = E_min/E_0 + (1 - E_min/E_0) rho^p
    s        = sigma^sol_vM / sigma_bar                   实体应力比
    g        = m_E (s - 1 + epsilon) - epsilon            约束值
    g <= 0  <=>  m_E <= epsilon / (s - 1 + epsilon)
            <=>  rho <= rho_crit = ((t - e) / (1 - e))^(1/p),
                 t = epsilon / (s - 1 + epsilon), e = E_min/E_0
    eta      = rho / rho_crit - 1                          余量, 负值可行

贴片端点是牵引间断点, 其 s 不随网格加密收敛, 因此 rho_crit 随阶次单调下降, 端点
邻域能否可行取决于构型在该处的 rho 落在 rho_crit 的哪一侧, 而不取决于哪条离散
"看得见"这个奇异性. 本表就是为核对这句话而写.

评价一律在不豁免 (radius=0) 的约束对象上做: 要报的正是被豁免的那些数, 豁免只用来
划分"进验收集合"与"不进验收集合"两栏.

入口由 ``compare.py`` 派发, 本模块不直接执行::

    compare.py stress-cross-eval [--design lfem-2] [--run-dir <目录名>] [--radius 1.5]

产出写到 ``outputs/<case>/postprocess/stress_cross_evaluation.json``, 带 provenance
戳记与所用 density_final.vtu 的 sha256; 终端同时打印两张 Markdown 表.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.postprocess.vtk_export import read_vtu_cell_data
from soptx.topology.constraints import build_exemption_mask

from config import (
    CASES_FILE,
    OUTPUT_DIR,
    bootstrap_source_path,
    flatten_parameters,
    load_cases,
)

bootstrap_source_path()

from pipeline import (  # noqa: E402
    build_stress_analysis_pipeline,
    build_stress_config,
)
import provenance  # noqa: E402
# 平衡残差与能量恒等式的唯一实现在 driver.py, 此处复用而不另写一份: 两处口径
# 必须一致, 否则交叉评价报的残差与 summary.json 里的对不上。
from driver import (  # noqa: E402
    energy_identity_diagnostics,
    relative_residual,
)

CASE_ID = "cantilever-middle-2d-stress"
# 交叉评价的离散组合: LFEM k=1..4 与 Hu--Zhang k=2..4 (HZ k=1 的 P0 位移空间缺
# 刚体旋转模态, 不进对照, 见 driver.py 模块首注).
DISCRETIZATIONS: tuple[tuple[str, int], ...] = (
    ("lfem", 1), ("lfem", 2), ("lfem", 3), ("lfem", 4),
    ("huzhang", 2), ("huzhang", 3), ("huzhang", 4),
)
DEFAULT_DESIGN = "lfem-2"


def case_parameters(case_id: str = CASE_ID) -> dict[str, Any]:
    """从 cases.toml 取该算例的扁平参数, 保证与优化时同口径."""
    for case in load_cases(CASES_FILE):
        if case["id"] == case_id:
            return flatten_parameters(case)
    raise SystemExit(f"cases.toml 中没有算例 {case_id}.")


def _split(label: str) -> tuple[str, int]:
    method, _, order = label.rpartition("-")
    return method, int(order)


def resolve_design_dir(label: str, run_dir: str | None) -> Path:
    """定位提供构型的 run 目录.

    ``--run-dir`` 显式给目录名时原样采用: 带垫片与不带垫片的产物同名不同尾,
    交叉评价常常要读上一代 (无垫片) 的构型, 不能被注册默认值绑死.
    """
    root = OUTPUT_DIR / CASE_ID
    if run_dir is not None:
        return root / run_dir
    method, order = _split(label)
    from metrics import resolve_run_dir

    # 目录名由 driver._run_label 按字段名排序生成, 且注册默认值本身会变迁,
    # 故统一走 metrics 的解析: 它按注册口径逐项核对 summary, 不靠拼名字.
    return resolve_run_dir(method, order, announce=False)[0]


def load_design(design_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """读取冻结构型及其运行摘要."""
    density_file = design_dir / "density_final.vtu"
    summary_file = design_dir / "summary.json"
    for path in (density_file, summary_file):
        if not path.is_file():
            raise SystemExit(f"缺少 {path}; 先把该组合跑出来, 或用 --run-dir 指定目录.")
    density = np.asarray(read_vtu_cell_data(density_file, "density"), dtype=np.float64)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return density, summary


def critical_density(
    stress_ratio: np.ndarray,
    epsilon: float,
    penalty: float,
    void_ratio: float,
) -> np.ndarray:
    """可行性临界密度 rho_crit: 给定实体应力比 s, 约束 g<=0 允许的最大密度.

    Parameters
    ----------
    stress_ratio : ndarray
        实体应力比 s = sigma^sol_vM / sigma_bar.
    epsilon : float
        epsilon 松弛参数.
    penalty : float
        MSIMP 惩罚指数 p.
    void_ratio : float
        空洞刚度比 E_min / E_0.

    Returns
    -------
    ndarray
        与 ``stress_ratio`` 同形状; s <= 1 时约束恒成立, 取 1.0; 松弛量不足以
        容纳该应力水平时取 0.0 (此时连空洞都不可行).
    """
    s = np.asarray(stress_ratio, dtype=np.float64)
    denominator = s - 1.0 + epsilon
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = np.where(denominator > 0.0, epsilon / denominator, np.inf)
        base = (threshold - void_ratio) / (1.0 - void_ratio)
    base = np.clip(base, 0.0, 1.0)
    return base ** (1.0 / penalty)


def _per_cell_max(values: Any) -> np.ndarray:
    """把 (NC, ...) 的评价点量约化为逐单元最大值."""
    array = np.asarray(bm.to_numpy(values), dtype=np.float64)
    return array.reshape(array.shape[0], -1).max(axis=1)


def evaluate(
    parameters: dict[str, Any],
    method: str,
    order: int,
    design: np.ndarray,
) -> dict[str, Any]:
    """在冻结构型上做一次前向求解, 返回逐单元的应力比与约束值."""
    run_parameters = {
        **parameters,
        "comparison_orders": [order],
        # 交叉评价要报的正是被垫片掩盖掉的数, 故在不豁免的约束对象上求值;
        # 实体保留在这里也一并关掉, 反正构型由 rho[:] = design 整体覆写,
        # 不经过过滤链, 保留与否不影响前向求解.
        "load_pad_radius": 0.0,
    }
    config = build_stress_config(run_parameters)
    pipeline = build_stress_analysis_pipeline(config, run_parameters, method, order)
    rho = pipeline.density_distribution
    if rho.shape[0] != design.shape[0]:
        raise SystemExit(
            f"{method}-{order}: 网格单元数 {rho.shape[0]} 与构型 {design.shape[0]} 不符; "
            "交叉评价必须与产出该构型的运行同网格."
        )
    rho[:] = design
    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    constraint = pipeline.stress_constraint
    constraint_value = constraint.fun(rho, state)
    solid_ratio = constraint.compute_solid_stress_ratio(rho, state)
    stress_measure = constraint.compute_stress_measure(rho, state)
    return {
        "pipeline": pipeline,
        "config": config,
        # 状态解一并返回: 灵敏度探针要在同一状态上继续做伴随, 不必重解一次.
        "state": state,
        "constraint": constraint,
        "density": rho,
        "solid_stress_ratio": _per_cell_max(solid_ratio),
        "apparent_stress_ratio": _per_cell_max(stress_measure),
        "constraint_value": _per_cell_max(constraint_value),
        # 未经 _per_cell_max 约化的原形 (NC, NQ): 离散敏感性探针要在评价点层面
        # 比较不同采样阶次, 逐单元取最大之后就看不出胞内起伏了.
        "raw_solid_stress_ratio": solid_ratio,
        "raw_apparent_stress_ratio": stress_measure,
        "raw_constraint_value": constraint_value,
        # 冻结构型把"设计"这个变量固定住之后, 残差随阶次的变化才是离散本身的
        # 性质; 各自优化产物里的残差测在各自不同的构型上, 不可横比。
        "relative_equilibrium_residual": relative_residual(pipeline, state),
        "energy_diagnostics": energy_identity_diagnostics(pipeline, state),
        "seconds": time.perf_counter() - started,
    }


def probe_cells(mesh, centers, radius: float) -> np.ndarray:
    """返回落在垫片半径内的单元编号 (即被移出约束集合且被钉为实体的那批)."""
    mask = build_exemption_mask(mesh=mesh, centers=centers, radius=radius)
    return np.flatnonzero(np.asarray(bm.to_numpy(mask)).astype(bool))


def cross_table(
    design_label: str,
    design_dir: Path,
    radius: float,
) -> dict[str, Any]:
    """一份构型 x 七条离散的交叉评价."""
    parameters = case_parameters()
    design, design_summary = load_design(design_dir)
    epsilon = float(parameters["epsilon"])
    penalty = float(parameters["penalty_factor"])
    void_ratio = float(parameters["void_youngs_modulus"]) / float(parameters["youngs_modulus"])

    rows: dict[str, Any] = {}
    exempt_cells: list[int] = []
    centers: list[list[float]] = []
    for method, order in DISCRETIZATIONS:
        label = f"{method}-{order}"
        result = evaluate(parameters, method, order, design)
        pipeline = result["pipeline"]
        if not exempt_cells:
            centers = [
                [float(value) for value in center]
                for center in pipeline.problem.traction_patch_endpoints
            ]
            exempt_cells = probe_cells(pipeline.mesh, centers, radius).tolist()
            barycenter = np.asarray(bm.to_numpy(pipeline.mesh.entity_barycenter("cell")))
        s = result["solid_stress_ratio"]
        g = result["constraint_value"]
        inside = np.zeros(s.shape[0], dtype=bool)
        inside[exempt_cells] = True
        # 豁免区内应力比最高的单元: 优化卡住时就是它
        hot = int(np.asarray(exempt_cells)[np.argmax(s[exempt_cells])]) if exempt_cells else -1
        rho_crit = critical_density(s, epsilon, penalty, void_ratio)
        margin = np.where(rho_crit > 0.0, design / np.maximum(rho_crit, 1e-300) - 1.0, np.inf)
        # 区外最大应力比所在单元的密度: 该最大值常落在 rho~0 的空洞单元上
        # (那里 m_E ~ 1e-9, g 恒等于 -epsilon, 无害), 不报 rho 会被误读成过载.
        outside_index = int(np.flatnonzero(~inside)[np.argmax(s[~inside])])
        rows[label] = {
            "max_apparent_stress_ratio": float(result["apparent_stress_ratio"].max()),
            "relative_equilibrium_residual": result["relative_equilibrium_residual"],
            "energy_diagnostics": result["energy_diagnostics"],
            "max_solid_stress_ratio": float(s.max()),
            "max_solid_stress_ratio_outside": float(s[~inside].max()),
            "max_solid_stress_ratio_outside_cell": {
                "index": outside_index,
                "barycenter": [float(v) for v in barycenter[outside_index]],
                "density": float(design[outside_index]),
            },
            "max_solid_stress_ratio_exempt": float(s[inside].max()) if exempt_cells else None,
            "max_constraint": float(g.max()),
            "max_constraint_outside": float(g[~inside].max()),
            "hot_cell": {
                "index": hot,
                "barycenter": [float(v) for v in barycenter[hot]],
                "density": float(design[hot]),
                "solid_stress_ratio": float(s[hot]),
                "critical_density": float(rho_crit[hot]),
                "margin": float(margin[hot]),
                "constraint_value": float(g[hot]),
            } if hot >= 0 else None,
            "seconds": result["seconds"],
        }

    return {
        "case_id": CASE_ID,
        "design": design_label,
        "design_run_dir": str(design_dir.relative_to(OUTPUT_DIR)),
        "design_digest": provenance.file_digest(design_dir / "density_final.vtu"),
        "design_summary": {
            key: design_summary.get(key)
            for key in (
                "optimization_iterations", "converged", "volume_fraction",
                "max_constraint", "max_relative_violation", "max_solid_stress_ratio",
                "load_pad_radius",
            )
        },
        "exemption_radius": radius,
        "exemption_centers": centers,
        "exempt_cells": exempt_cells,
        "epsilon": epsilon,
        "penalty_factor": penalty,
        "void_stiffness_ratio": void_ratio,
        "rows": rows,
    }


def print_markdown(block: dict[str, Any]) -> None:
    """终端打印三张表: 全域/分区应力比, 端点热点单元的可行性余量, 以及离散诊断."""
    print(
        f"\n{block['case_id']}: 冻结构型 {block['design']} "
        f"({block['design_run_dir']}), 垫片半径 R={block['exemption_radius']:g}, "
        f"覆盖 {len(block['exempt_cells'])} 个单元"
    )
    print("\n(一) 实体应力比与约束值 (行: 再分析离散)")
    print(
        "| 离散 | max s 全域 | max s 垫片外 | 该单元 rho | max s 垫片内 | "
        "max g 全域 | max g 垫片外 |"
    )
    print("|---|---|---|---|---|---|---|")
    for label, row in block["rows"].items():
        inside = row["max_solid_stress_ratio_exempt"]
        outside_cell = row["max_solid_stress_ratio_outside_cell"]
        print(
            f"| {label} | {row['max_solid_stress_ratio']:.3f} | "
            f"{row['max_solid_stress_ratio_outside']:.3f} | "
            f"{outside_cell['density']:.4f} | "
            f"{'-' if inside is None else f'{inside:.3f}'} | "
            f"{row['max_constraint']:+.5f} | {row['max_constraint_outside']:+.5f} |"
        )

    print("\n(二) 垫片内应力最高单元的可行性余量 eta = rho / rho_crit - 1 (负值可行)")
    print("| 离散 | 单元重心 | rho | s | rho_crit | eta | g |")
    print("|---|---|---|---|---|---|---|")
    for label, row in block["rows"].items():
        cell = row["hot_cell"]
        if cell is None:
            continue
        x, y = cell["barycenter"]
        print(
            f"| {label} | ({x:.2f}, {y:.2f}) | {cell['density']:.4f} | "
            f"{cell['solid_stress_ratio']:.3f} | {cell['critical_density']:.4f} | "
            f"{cell['margin']:+.1%} | {cell['constraint_value']:+.5f} |"
        )

    print("\n(三) 同一构型下各离散的表观应力读数与平衡残差")
    print("| 离散 | max 表观应力比 | 相对平衡残差 | 能量相对缺陷 | 耗时 (s) |")
    print("|---|---|---|---|---|")
    for label, row in block["rows"].items():
        energy = row["energy_diagnostics"]
        # LFEM 报 f^T u 与 u^T K u 的相对缺陷; 混合元没有同构的单一恒等式, 留空.
        defect = energy.get("relative_defect")
        print(
            f"| {label} | {row['max_apparent_stress_ratio']:.6f} | "
            f"{row['relative_equilibrium_residual']:.3e} | "
            f"{'-' if defect is None else f'{defect:.3e}'} | "
            f"{row['seconds']:.2f} |"
        )


def write_json(payload: dict[str, Any]) -> Path:
    target = OUTPUT_DIR / CASE_ID / "postprocess" / "stress_cross_evaluation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare.py stress-cross-eval",
        description="悬臂梁应力算例的冻结设计交叉评价 (一份构型, 七条离散).",
    )
    parser.add_argument(
        "--design", default=DEFAULT_DESIGN, metavar="<method>-<order>",
        help=f"提供构型的运行组合, 默认 {DEFAULT_DESIGN}.",
    )
    parser.add_argument(
        "--run-dir", default=None, metavar="<目录名>",
        help="直接指定 outputs/<case>/ 下的运行目录名 (读上一代无垫片产物时用).",
    )
    parser.add_argument(
        "--radius", type=float, default=None, metavar="<mm>",
        help="划分垫片区的半径, 默认取 cases.toml 的 load_pad_radius.",
    )
    return parser


def run_stress_cross_evaluation(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv or [])
    radius = arguments.radius
    if radius is None:
        radius = float(case_parameters().get("load_pad_radius", 0.0))
    design_dir = resolve_design_dir(arguments.design, arguments.run_dir)

    payload = cross_table(arguments.design, design_dir, float(radius))
    payload["provenance"] = provenance.run_stamp()
    print_markdown(payload)
    target = write_json(payload)
    print(f"\n[cross-eval] {target.relative_to(OUTPUT_DIR.parent)}")
    return 0
