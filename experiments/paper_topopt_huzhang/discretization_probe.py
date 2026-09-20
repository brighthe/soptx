# -*- coding: utf-8 -*-
"""实验 A: 冻结构型下的离散敏感性探针.

§5.2.3 的论证链条是"离散精度高 -> 应力场更准 -> 约束值更准 -> 设计更优". 首环由
§5.1 的收敛阶坐实, 末环被现有产物否定 (两种垫片设置下 Hu--Zhang 的体积分数均高于
LFEM 1.6%--1.9%), 中间两环从未测过. 优化产物之间无法分离归因: 设计与离散一起变,
体积差既可能来自离散精度, 也可能来自分叉或谁先碰到停止线.

本模块把设计钉死, 只让离散变化, 于是应力场的任何差异都只能归因于离散. 四项分析::

    A1  七条离散在同一构型上的逐单元 g 的散布, 按密度带分解, 与停止容差 delta_g
        = 5e-3 对比; 并核对各离散是否把 g_max 放在同一个单元上 —— 若不是, 横比
        本身就没有意义.
    A2  胞内采样敏感性. 两族的 compute_stress_state 默认 integration_order=1,
        三角形一点求积即重心, 故约束无论 k 多大都只读单元形心一个点. 提高采样
        阶次后 max_q g_e 比形心值高多少, 以及该差随 k 如何变化.
    A3  内边法向牵引跳量 (见 edge_jump). Hu--Zhang 因 H(div, S) 协调该量恒为 0,
        LFEM 不为 0; 若其相对量与实体带的可行余量 (1%--4%) 同量级, 则"LFEM 下
        单元应力值本身有定义歧义"成立.
    A4  细网格双参考 (可选). 在 160x80 上各以 Hu--Zhang k=4 与 LFEM k=4 求一次,
        把粗离散的读数与之对照; 两条参考彼此的分歧同时用来检验参考自身是否可信.

入口由 ``compare.py`` 派发, 本模块不直接执行::

    compare.py discretization-probe [--design <目录名>]... [--reference] [--no-jump]

产出写到 ``outputs/<case>/postprocess/discretization_probe/``: 逐构型一份 npz
(逐单元原始场, 使求解成为一次性成本)、一份 json (全部聚合表 + provenance) 与
逐离散的 vtu; 终端同时打印 Markdown 表.

被动实体区 (summary 的 ``load_pad_centers`` / ``load_pad_radius`` 圈出的单元, 优化中
不施加约束且钉为实体) 的 g 照常求出并写入 npz, 但不进入 A1 的密度带、前若干单元与
A3 的实体带统计; 三个最大值 (全域 / 区外 / 判据集合 rho >= acceptance_solid_threshold)
分开报. 构型自身离散在判据集合上的最大值须与 summary 的
``max_relative_violation_solid_region`` 一致, 作为验收 3.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.postprocess.vtk_export import write_vtu

from config import OUTPUT_DIR

from edge_jump import interior_edge_traction_jump
from pipeline import create_mesh
import provenance
from stress_cross_evaluation import (
    CASE_ID,
    DISCRETIZATIONS,
    case_parameters,
    critical_density,
    evaluate,
    load_design,
    probe_cells,
)

# 停止准则的容差, 一切散布都要跟它比; 与 cases.toml 的 stress_tolerance 同值.
DELTA_G = 5.0e-3
# 可信子集: 排除 LFEM k=1 (应力逐单元常值, O(h)) 与 Hu--Zhang k=2 (跳量稳定化的
# 惩罚系数用基材模量, 不随密度插值, 空区虚假应力 +1428%, 见 docs/known-issues).
TRUSTED_LABELS: tuple[str, ...] = ("lfem-3", "lfem-4", "huzhang-3", "huzhang-4")
# 逐单元采样阶次: 1 = 形心 (生产口径), 3 与 4 用来看胞内起伏.
SAMPLING_ORDERS: tuple[int, ...] = (1, 3, 4)
DENSITY_EDGES = np.linspace(0.0, 1.0, 11)
TOP_CELLS = 5
# 细网格参考的加密倍数与所用离散.
REFERENCE_REFINEMENT = 2
REFERENCE_DISCRETIZATIONS: tuple[tuple[str, int], ...] = (("huzhang", 4), ("lfem", 4))

# §5.2.3 主对比的两条 k=3 产物 (pad 1.5 mm, 判据集合 rho >= 0.5, lambda_max 3000);
# 两份构型互为交叉验证, 避免重蹈旧 stress_cross_evaluation.json 在自己的构型上
# 自评的来源偏置.
DEFAULT_DESIGNS: tuple[str, ...] = (
    "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-3__solid_thr-0.5",
    "analyzer-lfem__lfem_constraint-apparent__load_pad_radius-1.5__order-3__solid_thr-0.5",
)


# ================================================================ 一、工具


def design_label(run_dir: str) -> str:
    """把运行目录名压成短标签, 如 ``huzhang-3-nopad`` / ``huzhang-3-pad-solid``.

    ``-solid`` 后缀表示该运行用判据集合 (solid_thr) 验收, 与同 pad 半径的旧运行区分.
    """
    method = re.search(r"analyzer-(\w+?)__", run_dir)
    order = re.search(r"order-(\d+)", run_dir)
    if method is None or order is None:
        return run_dir
    pad = "pad" if "load_pad_radius" in run_dir else "nopad"
    solid = "-solid" if "solid_thr" in run_dir else ""
    return f"{method.group(1)}-{order.group(1)}-{pad}{solid}"


def design_discretization(run_dir: str) -> str | None:
    """产出该构型的离散标签, 如 ``huzhang-3``; 解析不出返回 None."""
    method = re.search(r"analyzer-(\w+?)__", run_dir)
    order = re.search(r"order-(\d+)", run_dir)
    if method is None or order is None:
        return None
    return f"{method.group(1)}-{order.group(1)}"


def pad_mask_from_summary(mesh, summary: dict[str, Any], n_cells: int) -> np.ndarray:
    """按运行摘要记录的圆心与半径复原被动实体区的单元掩码.

    半径为 0 或摘要无圆心时返回全 False; 单元数与摘要 ``pad_cells`` 不符即报错.
    """
    mask = np.zeros(n_cells, dtype=bool)
    radius = float(summary.get("load_pad_radius", 0.0) or 0.0)
    centers = summary.get("load_pad_centers") or []
    if radius <= 0.0 or not centers:
        return mask
    mask[probe_cells(mesh, centers, radius)] = True
    expected = summary.get("load_pad_cells", summary.get("pad_cells"))
    if expected is not None and int(mask.sum()) != int(expected):
        raise RuntimeError(
            f"被动实体区单元数 {int(mask.sum())} 与摘要记录的 {expected} 不符."
        )
    return mask


def _as_numpy(values: Any) -> np.ndarray:
    return np.asarray(bm.to_numpy(values), dtype=np.float64)


def _jsonify(value: Any) -> Any:
    """把含 numpy 标量/数组的嵌套结构转成可序列化形式."""
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def resample_constraint(
    result: dict[str, Any],
    integration_order: int,
    stress_limit: float,
) -> np.ndarray:
    """在同一状态解上按给定采样阶次重算逐评价点的约束值 ``g``.

    不重解, 只重采样: 复用 ``result['state']`` 与约束对象自己的 formulation, 因此
    与生产口径 (``constraint.fun``) 严格同代数, 唯一差别是求积点的位置与个数.

    Parameters
    ----------
    result : dict
        ``stress_cross_evaluation.evaluate`` 的返回.
    integration_order : int
        单元求积阶次; 1 即形心, 与两族 analyzer 的默认一致.
    stress_limit : float
        材料许用应力, 取自 cases.toml, 与构造约束对象时同源.

    Returns
    -------
    ndarray, shape (NC, NQ)
        该采样下的约束值, 未施加豁免.
    """
    pipeline = result["pipeline"]
    state = result["state"]
    constraint = result["constraint"]
    analyzer = pipeline.analyzer
    formulation = constraint.formulation

    if pipeline.method == "huzhang":
        raw = analyzer.compute_stress_state(
            state=state,
            rho_val=result["density"],
            integration_order=integration_order,
        )["stress_apparent"]
        representation = "apparent"
    else:
        raw = analyzer.compute_stress_state(
            state, integration_order=integration_order
        )["stress_solid"]
        representation = "solid"

    von_mises = analyzer.material.calculate_von_mises_stress(raw)
    value = formulation.constraint_value(
        stress_ratio=von_mises / stress_limit,
        stiffness_ratio=state["stiffness_ratio"],
        stress_representation=representation,
    )
    return _as_numpy(value)


# ================================================================ 二、A1 散布


def _bin_index(density: np.ndarray) -> np.ndarray:
    """密度落入 [0,0.1),...,[0.9,1.0] 的档位编号."""
    index = np.digitize(density, DENSITY_EDGES[1:-1], right=False)
    return np.clip(index, 0, len(DENSITY_EDGES) - 2)


def density_band_table(
    density: np.ndarray,
    constraint_by_label: dict[str, np.ndarray],
    keep: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """按密度带汇总各离散的 ``g``, 并给出离散间散布 Delta = max - min.

    ``keep`` 为 False 的单元 (被动实体区) 不计入任何密度带.
    """
    bins = _bin_index(density)
    labels = list(constraint_by_label)
    trusted = [label for label in labels if label in TRUSTED_LABELS]
    if keep is None:
        keep = np.ones(density.shape[0], dtype=bool)

    rows: list[dict[str, Any]] = []
    for band in range(len(DENSITY_EDGES) - 1):
        selected = (bins == band) & keep
        count = int(selected.sum())
        entry: dict[str, Any] = {
            "band": [float(DENSITY_EDGES[band]), float(DENSITY_EDGES[band + 1])],
            "cells": count,
        }
        if count == 0:
            entry["max_g"] = {}
            rows.append(entry)
            continue

        maxima = {label: float(constraint_by_label[label][selected].max()) for label in labels}
        entry["max_g"] = maxima
        entry["mean_g"] = {
            label: float(constraint_by_label[label][selected].mean()) for label in labels
        }
        # 逐单元散布再取最大, 比"各自最大值之差"严格: 后者可能来自不同单元.
        stacked = np.stack([constraint_by_label[label][selected] for label in labels])
        entry["spread_all"] = float((stacked.max(axis=0) - stacked.min(axis=0)).max())
        if len(trusted) >= 2:
            stacked_trusted = np.stack(
                [constraint_by_label[label][selected] for label in trusted]
            )
            spread = float((stacked_trusted.max(axis=0) - stacked_trusted.min(axis=0)).max())
        else:
            spread = float("nan")
        entry["spread_trusted"] = spread
        entry["spread_trusted_over_delta_g"] = spread / DELTA_G
        rows.append(entry)
    return rows


def active_cell_table(
    density: np.ndarray,
    barycenter: np.ndarray,
    constraint_by_label: dict[str, np.ndarray],
    keep: np.ndarray | None = None,
) -> dict[str, Any]:
    """各离散 ``g`` 最大的前若干单元, 以及这些单元集合的两两交集大小.

    ``keep`` 为 False 的单元 (被动实体区) 不参与排序.
    """
    top: dict[str, Any] = {}
    sets: dict[str, set[int]] = {}
    if keep is None:
        keep = np.ones(density.shape[0], dtype=bool)
    for label, values in constraint_by_label.items():
        ranked = np.where(keep, values, -np.inf)
        order = np.argsort(ranked)[::-1][:TOP_CELLS]
        sets[label] = set(int(index) for index in order)
        top[label] = [
            {
                "index": int(index),
                "barycenter": [float(v) for v in barycenter[index]],
                "density": float(density[index]),
                "constraint_value": float(values[index]),
            }
            for index in order
        ]

    labels = list(constraint_by_label)
    overlap = {
        f"{a}|{b}": len(sets[a] & sets[b])
        for position, a in enumerate(labels)
        for b in labels[position + 1:]
    }
    return {"top_cells": top, "pairwise_overlap": overlap}


# ================================================================ 三、主流程


def probe_design(
    design_dir: Path,
    with_reference: bool,
    with_jump: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """对一份冻结构型做 A1--A4.

    Returns
    -------
    payload : dict
        聚合结果, 可直接序列化为 json.
    fields : dict of ndarray
        逐单元原始场, 写入 npz.
    meshes : dict
        出 vtu 需要的 mesh 与逐离散场.
    """
    parameters = case_parameters()
    design, design_summary = load_design(design_dir)
    label = design_label(design_dir.name)
    own = design_discretization(design_dir.name)
    stress_limit = float(parameters["stress_limit"])
    solid_threshold = design_summary.get("acceptance_solid_threshold")

    constraint_by_label: dict[str, np.ndarray] = {}
    fields: dict[str, np.ndarray] = {"density": design}
    pad = np.zeros(design.shape[0], dtype=bool)
    accepted = np.ones(design.shape[0], dtype=bool)
    rows: dict[str, Any] = {}
    sampling: dict[str, Any] = {}
    jumps: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    mesh = None
    problem = None
    barycenter = None

    for method, order in DISCRETIZATIONS:
        discretization = f"{method}-{order}"
        result = evaluate(parameters, method, order, design)
        pipeline = result["pipeline"]
        if mesh is None:
            mesh = pipeline.mesh
            problem = pipeline.problem
            barycenter = _as_numpy(mesh.entity_barycenter("cell"))
            fields["barycenter"] = barycenter
            # 被动实体区与判据集合: 前者不施加约束且钉为实体, 后者是停止判据的
            # 评价集合 (区外且 rho >= 阈值); 无阈值记录时判据集合即区外全部单元.
            pad = pad_mask_from_summary(mesh, design_summary, design.shape[0])
            accepted = ~pad
            if solid_threshold is not None:
                accepted &= design >= float(solid_threshold)
            fields["pad_mask"] = pad
            fields["accepted_mask"] = accepted

        centroid = result["constraint_value"]
        constraint_by_label[discretization] = centroid
        fields[f"g__{discretization}"] = centroid
        fields[f"s__{discretization}"] = result["solid_stress_ratio"]
        fields[f"vm_app__{discretization}"] = result["apparent_stress_ratio"]

        # --- A2 胞内采样敏感性 ---
        per_order: dict[str, Any] = {}
        for sampling_order in SAMPLING_ORDERS:
            values = resample_constraint(result, sampling_order, stress_limit)
            per_cell = values.reshape(values.shape[0], -1).max(axis=1)
            fields[f"g_q{sampling_order}__{discretization}"] = per_cell
            per_order[str(sampling_order)] = {
                "points_per_cell": int(values.shape[1]),
                "max_g": float(per_cell.max()),
                "mean_excess_over_centroid": float((per_cell - centroid).mean()),
                "max_excess_over_centroid": float((per_cell - centroid).max()),
                "p95_excess_over_centroid": float(np.percentile(per_cell - centroid, 95)),
            }
        # 验收 2: q=1 必须与生产口径逐单元 bit 级一致.
        reproduced = fields[f"g_q1__{discretization}"]
        checks[f"resample_q1_matches_fun__{discretization}"] = bool(
            np.array_equal(reproduced, centroid)
        )
        sampling[discretization] = per_order

        # --- A3 内边法向牵引跳量 ---
        if with_jump:
            jump = interior_edge_traction_jump(pipeline, result["state"])
            usable = jump["well_scaled"]
            relative = jump["relative_jump"][usable]
            edge_density = design[jump["cells"]].min(axis=1)[usable]
            edge_in_pad = pad[jump["cells"]].any(axis=1)[usable]
            solid_edges = (edge_density > 0.9) & ~edge_in_pad
            jumps[discretization] = {
                "integration_order": jump["integration_order"],
                "interior_faces": int(jump["interior_faces"].size),
                "usable_faces": int(usable.sum()),
                "relative_jump_median": float(np.median(relative)) if relative.size else None,
                "relative_jump_p95": float(np.percentile(relative, 95)) if relative.size else None,
                "relative_jump_max": float(relative.max()) if relative.size else None,
                "solid_band_faces": int(solid_edges.sum()),
                "solid_band_median": (
                    float(np.median(relative[solid_edges])) if solid_edges.any() else None
                ),
                "solid_band_p95": (
                    float(np.percentile(relative[solid_edges], 95)) if solid_edges.any() else None
                ),
                "solid_band_max": (
                    float(relative[solid_edges].max()) if solid_edges.any() else None
                ),
            }

            # 绝对跳量 (逐边 RMS / 许用应力): 与 g 同尺度, 直接和 DELTA_G、余量 -g
            # 对照. 逐面统计取两侧均为实体且不在被动实体区的内边 (不需 well_scaled,
            # 那是比值才有的地板); 逐单元 A_e 取单元各内边的最大值, 统计限于实体带.
            abs_face = jump["rms_jump"] / stress_limit
            abs_solid_edges = (design[jump["cells"]].min(axis=1) > 0.9) & ~pad[jump["cells"]].any(axis=1)
            abs_cell = jump["cell_rms_jump"] / stress_limit
            solid_cells = (design > 0.9) & ~pad
            headroom = -centroid
            face_values = abs_face[abs_solid_edges]
            cell_values = abs_cell[solid_cells]
            jumps[discretization].update({
                "abs_solid_band_faces": int(abs_solid_edges.sum()),
                "abs_solid_band_median": float(np.median(face_values)) if face_values.size else None,
                "abs_solid_band_p95": float(np.percentile(face_values, 95)) if face_values.size else None,
                "abs_solid_band_max": float(face_values.max()) if face_values.size else None,
                "cell_abs_solid_cells": int(solid_cells.sum()),
                "cell_abs_solid_median": float(np.median(cell_values)) if cell_values.size else None,
                "cell_abs_solid_p95": float(np.percentile(cell_values, 95)) if cell_values.size else None,
                "cell_abs_solid_max": float(cell_values.max()) if cell_values.size else None,
                "share_cell_abs_gt_delta": float(np.mean(cell_values > DELTA_G)) if cell_values.size else None,
                "share_cell_abs_gt_4delta": float(np.mean(cell_values > 4.0 * DELTA_G)) if cell_values.size else None,
                "share_cell_abs_gt_headroom": (
                    float(np.mean(cell_values > headroom[solid_cells])) if cell_values.size else None
                ),
            })
            fields[f"absjump__{discretization}"] = abs_cell
            fields[f"absjumpF__{discretization}"] = abs_face
            fields[f"jump__{discretization}"] = jump["cell_relative_jump"]

            # 逐边原始量: 两侧牵引的法向/切向分量 (已除以许用应力) 与逐边相对
            # 跳量, 供剖面图 (论文图 5.14(a)(b)) 与表 5.5 直接从 npz 复算; 边几何
            # 只写一次. trace 形状 (NF_int, 2 侧, NQ, 2 分量), 侧序同 face_cells.
            normal = jump["normal"]
            tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
            n_quadrature = jump["points"].shape[1]
            trace = np.empty((normal.shape[0], 2, n_quadrature, 2), dtype=np.float64)
            for side, traction in enumerate((jump["left_traction"], jump["right_traction"])):
                trace[:, side, :, 0] = (traction * normal[:, None, :]).sum(-1) / stress_limit
                trace[:, side, :, 1] = (traction * tangent[:, None, :]).sum(-1) / stress_limit
            fields[f"trace__{discretization}"] = trace
            fields[f"trace_points__{discretization}"] = jump["points"]
            fields[f"trace_weights__{discretization}"] = jump["weights"]
            fields[f"jumpF__{discretization}"] = jump["relative_jump"]
            fields[f"well_scaled__{discretization}"] = jump["well_scaled"]
            if "face_index" not in fields:
                fields["face_index"] = jump["interior_faces"]
                fields["face_cells"] = jump["cells"]
                fields["face_normal"] = normal
                fields["face_measure"] = jump["measure"]
                fields["face_midpoint"] = jump["points"].mean(axis=1)

    # 验收 1 (硬门): Hu--Zhang 的相对跳量必须落在舍入量级.
    if with_jump:
        gate = {
            discretization: value["relative_jump_max"]
            for discretization, value in jumps.items()
            if discretization.startswith("huzhang")
        }
        checks["huzhang_jump_gate"] = {
            "values": gate,
            "threshold": 1e-10,
            "passed": all(
                item is not None and item < 1e-10 for item in gate.values()
            ),
        }

    # 验收 3: 构型自身离散在判据集合上的最大 g 必须复现 summary 的记录; 不一致
    # 说明掩码或阈值口径与优化器不同, 后面所有"判据集合"统计都不可信.
    recorded = design_summary.get("max_relative_violation_solid_region")
    if own in constraint_by_label and recorded is not None:
        reproduced = float(constraint_by_label[own][accepted].max())
        checks["accepted_max_g_matches_summary"] = {
            "discretization": own,
            "reproduced": reproduced,
            "recorded": float(recorded),
            "passed": bool(np.isclose(reproduced, float(recorded), rtol=1e-6, atol=1e-9)),
        }

    epsilon = float(parameters["epsilon"])
    penalty = float(parameters["penalty_factor"])
    void_ratio = float(parameters["void_youngs_modulus"]) / float(parameters["youngs_modulus"])
    rho_crit = critical_density(
        fields["s__huzhang-4"], epsilon, penalty, void_ratio
    )
    fields["rho_crit__huzhang-4"] = rho_crit

    payload: dict[str, Any] = {
        "case_id": CASE_ID,
        "design": label,
        "design_run_dir": str(design_dir.relative_to(OUTPUT_DIR)),
        "design_digest": provenance.file_digest(design_dir / "density_final.vtu"),
        "design_summary": {
            key: design_summary.get(key)
            for key in (
                "optimization_iterations", "converged", "volume_fraction",
                "max_constraint", "max_relative_violation",
                "max_relative_violation_solid_region", "acceptance_solid_threshold",
                "load_pad_radius", "pad_cells", "lambda_max", "multiplier_capped_count",
            )
        },
        "delta_g": DELTA_G,
        "trusted_labels": list(TRUSTED_LABELS),
        "design_discretization": own,
        "pad_cells": int(pad.sum()),
        "accepted_cells": int(accepted.sum()),
        "outside_pad_cells": int((~pad).sum()),
        "global_max_g": {
            discretization: float(values.max())
            for discretization, values in constraint_by_label.items()
        },
        "max_g_outside_pad": {
            discretization: float(values[~pad].max())
            for discretization, values in constraint_by_label.items()
        },
        "max_g_accepted": {
            discretization: float(values[accepted].max())
            for discretization, values in constraint_by_label.items()
        },
        "density_bands": density_band_table(design, constraint_by_label, keep=~pad),
        "active_cells": active_cell_table(
            design, barycenter, constraint_by_label, keep=~pad
        ),
        "sampling": sampling,
        "traction_jump": jumps,
        "checks": checks,
    }

    # --- A4 细网格双参考 ---
    if with_reference:
        block, reduced = reference_block(
            parameters, design, problem, mesh, constraint_by_label,
            pad=pad, accepted=accepted,
        )
        payload["reference"] = block
        fields.update(reduced)

    return payload, fields, {"mesh": mesh, "labels": list(constraint_by_label)}


def reference_block(
    parameters: dict[str, Any],
    design: np.ndarray,
    problem: Any,
    coarse_mesh: Any,
    constraint_by_label: dict[str, np.ndarray],
    pad: np.ndarray | None = None,
    accepted: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """在加密网格上以两条高阶离散各求一次参考解, 并与粗离散读数对照.

    全域字段 (``max_g_fine`` / ``max_g_reduced`` / ``reference_disagreement`` /
    ``deviation``) 含被动实体区, 有 pad 的运行里被载荷奇异性主导, 只作留档;
    带 ``_outside_pad`` / ``_accepted`` 后缀的字段分别屏蔽 pad、限于判据集合,
    与优化停止判据 (``max_g_accepted``) 同口径, 是有 pad 运行应引用的量.

    密度用 ``singularity_h_probe.map_density_to_mesh`` 做面积保持的分片常数延拓
    (本目录既有做法). 把单元编号本身当作"密度"喂给同一个映射, 得到的就是逐细单元
    的父单元编号, 因此父映射与密度映射严格同源, 不必另写一份点定位.

    两条参考互为检验: 若它们彼此的分歧已经超过 ``delta_g``, 则参考解本身不可信,
    A4 的结论作废, 只保留 A1--A3.

    Returns
    -------
    block : dict
        可序列化的对照结果.
    fields : dict of ndarray
        归约回粗网格的逐单元参考 ``g``.
    """
    from singularity_h_probe import map_density_to_mesh

    nx, ny = int(parameters["nx"]), int(parameters["ny"])
    mesh_type = str(parameters["mesh_type"])
    domain = list(coarse_mesh.meshdata["domain"])
    fine_nx = nx * REFERENCE_REFINEMENT
    fine_ny = ny * REFERENCE_REFINEMENT
    fine_parameters = {**parameters, "nx": fine_nx, "ny": fine_ny}

    fine_mesh = create_mesh(problem, fine_nx, fine_ny, mesh_type)
    n_coarse = design.shape[0]
    parent = map_density_to_mesh(
        np.arange(n_coarse, dtype=np.float64), coarse_mesh, (nx, ny), fine_mesh, domain
    ).astype(np.int64)
    if parent.min() < 0 or parent.max() >= n_coarse:
        raise RuntimeError("父单元映射越界, 加密网格与粗网格不同域.")
    design_fine = design[parent]

    reduced: dict[str, np.ndarray] = {}
    block: dict[str, Any] = {
        "refinement": REFERENCE_REFINEMENT,
        "fine_shape": [fine_nx, fine_ny],
        "fine_cells": int(design_fine.shape[0]),
        "rows": {},
    }
    for method, order in REFERENCE_DISCRETIZATIONS:
        label = f"{method}-{order}"
        result = evaluate(fine_parameters, method, order, design_fine)
        fine_g = result["constraint_value"]
        # 细单元按父单元取最大: 约束是逐单元的上确界语义, 取最大才与粗读数同口径.
        coarse_g = np.full(n_coarse, -np.inf, dtype=np.float64)
        np.maximum.at(coarse_g, parent, fine_g)
        reduced[f"g_ref__{label}"] = coarse_g
        row = {
            "max_g_fine": float(fine_g.max()),
            "max_g_reduced": float(coarse_g.max()),
            "seconds": float(result["seconds"]),
        }
        if pad is not None:
            row["max_g_reduced_outside_pad"] = float(coarse_g[~pad].max())
        if accepted is not None:
            g_acc = coarse_g[accepted]
            row["max_g_reduced_accepted"] = float(g_acc.max())
            row["accepted_cells"] = int(accepted.sum())
            row["violation_share_accepted"] = float((g_acc > 0.0).mean())
            row["violation_share_accepted_over_delta"] = float((g_acc > DELTA_G).mean())
        block["rows"][label] = row

    reference_labels = [f"{method}-{order}" for method, order in REFERENCE_DISCRETIZATIONS]
    first, second = (reduced[f"g_ref__{label}"] for label in reference_labels)
    disagreement = float(np.abs(first - second).max())
    block["reference_disagreement"] = disagreement
    block["reference_disagreement_over_delta_g"] = disagreement / DELTA_G
    # 硬判据: 两条参考彼此分歧超过容差时, 它们谁也不能当标尺.
    block["reference_trustworthy"] = bool(disagreement < DELTA_G)
    if pad is not None:
        keep = ~pad
        disagreement_outside = float(np.abs(first - second)[keep].max())
        block["reference_disagreement_outside_pad"] = disagreement_outside
        block["reference_disagreement_outside_pad_over_delta_g"] = disagreement_outside / DELTA_G
        block["reference_trustworthy_outside_pad"] = bool(disagreement_outside < DELTA_G)
    if accepted is not None:
        disagreement_accepted = float(np.abs(first - second)[accepted].max())
        block["reference_disagreement_accepted"] = disagreement_accepted
        block["reference_disagreement_accepted_over_delta_g"] = disagreement_accepted / DELTA_G
        block["reference_trustworthy_accepted"] = bool(disagreement_accepted < DELTA_G)

    def _deviation(keep: np.ndarray | None) -> dict[str, dict[str, float]]:
        return {
            discretization: {
                label: float(
                    np.abs(values - reduced[f"g_ref__{label}"])[
                        slice(None) if keep is None else keep
                    ].max()
                )
                for label in reference_labels
            }
            for discretization, values in constraint_by_label.items()
        }

    block["deviation"] = _deviation(None)
    if pad is not None:
        block["deviation_outside_pad"] = _deviation(~pad)
    if accepted is not None:
        block["deviation_accepted"] = _deviation(accepted)
    return block, reduced


def print_markdown(payload: dict[str, Any]) -> None:
    """终端打印 A1--A3 的三张表."""
    print(
        f"\n{payload['case_id']}: 冻结构型 {payload['design']} "
        f"({payload['design_run_dir']}), delta_g = {payload['delta_g']:g}"
    )

    labels = list(payload["global_max_g"])
    print(
        f"被动实体区 {payload['pad_cells']} 单元 (不计入下列统计), "
        f"区外 {payload['outside_pad_cells']}, 判据集合 {payload['accepted_cells']}"
    )
    print("| 离散 | max g 全域 | max g 区外 | max g 判据集合 |")
    print("|---|---|---|---|")
    for label in labels:
        print(
            f"| {label} | {payload['global_max_g'][label]:+.5f} | "
            f"{payload['max_g_outside_pad'][label]:+.5f} | "
            f"{payload['max_g_accepted'][label]:+.5f} |"
        )
    gate3 = payload["checks"].get("accepted_max_g_matches_summary")
    if gate3 is not None:
        verdict = "通过" if gate3["passed"] else "未通过 —— 掩码口径与优化器不符"
        print(
            f"[验收 3] {gate3['discretization']} 判据集合 max g 复现 = "
            f"{gate3['reproduced']:+.6e}, summary 记录 = {gate3['recorded']:+.6e}: {verdict}"
        )

    print("\n(一) A1 逐单元 g 的离散间散布, 按密度带 (Delta 为逐单元散布的最大值; 被动实体区已剔除)")
    print("| rho 区间 | 单元数 | " + " | ".join(f"max g {label}" for label in labels)
          + " | Delta 全部 | Delta 可信 | Delta/delta_g |")
    print("|---" * (len(labels) + 5) + "|")
    for row in payload["density_bands"]:
        if not row["max_g"]:
            continue
        low, high = row["band"]
        cells = " | ".join(f"{row['max_g'][label]:+.5f}" for label in labels)
        print(
            f"| [{low:.1f}, {high:.1f}) | {row['cells']} | {cells} | "
            f"{row['spread_all']:.5f} | {row['spread_trusted']:.5f} | "
            f"{row['spread_trusted_over_delta_g']:.1f} |"
        )

    print("\n(二) A2 胞内采样敏感性: max_q g_e 相对形心值的抬升")
    print("| 离散 | q=1 max g | q=3 点数 | q=3 max 抬升 | q=4 点数 | q=4 max 抬升 | q=4 P95 抬升 |")
    print("|---|---|---|---|---|---|---|")
    for label, block in payload["sampling"].items():
        one, three, four = block["1"], block["3"], block["4"]
        print(
            f"| {label} | {one['max_g']:+.5f} | {three['points_per_cell']} | "
            f"{three['max_excess_over_centroid']:.5f} | {four['points_per_cell']} | "
            f"{four['max_excess_over_centroid']:.5f} | "
            f"{four['p95_excess_over_centroid']:.5f} |"
        )

    if payload["traction_jump"]:
        print("\n(三) A3 内边法向牵引相对跳量 ||[[sigma n]]|| / ||{sigma n}||")
        print("| 离散 | 可用内边 | 中位数 | P95 | 最大 | 实体带边数 | 实体带中位数 | 实体带 P95 |")
        print("|---|---|---|---|---|---|---|---|")
        for label, block in payload["traction_jump"].items():
            def fmt(value: float | None) -> str:
                return "-" if value is None else f"{value:.3e}"
            print(
                f"| {label} | {block['usable_faces']} | "
                f"{fmt(block['relative_jump_median'])} | {fmt(block['relative_jump_p95'])} | "
                f"{fmt(block['relative_jump_max'])} | {block['solid_band_faces']} | "
                f"{fmt(block['solid_band_median'])} | {fmt(block['solid_band_p95'])} |"
            )
        gate = payload["checks"].get("huzhang_jump_gate")
        if gate is not None:
            verdict = "通过" if gate["passed"] else "未通过 —— A3 结果无效"
            print(f"\n[验收门] Hu--Zhang 相对跳量 < {gate['threshold']:g}: {verdict}")

        print("\n(三b) A3 内边法向牵引绝对跳量 RMS / sigma_bar (实体带) 与逐单元 A_e = max_F")
        print("| 离散 | 实体带边数 | 中位数 | P95 | 最大 | 实体单元数 | A_e 中位 | A_e P95 | A_e>δg | A_e>4δg | A_e>-g_e |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, block in payload["traction_jump"].items():
            if "abs_solid_band_median" not in block:
                continue
            def fmt(value: float | None) -> str:
                return "-" if value is None else f"{value:.3e}"
            def pct(value: float | None) -> str:
                return "-" if value is None else f"{100 * value:.1f}%"
            print(
                f"| {label} | {block['abs_solid_band_faces']} | "
                f"{fmt(block['abs_solid_band_median'])} | {fmt(block['abs_solid_band_p95'])} | "
                f"{fmt(block['abs_solid_band_max'])} | {block['cell_abs_solid_cells']} | "
                f"{fmt(block['cell_abs_solid_median'])} | {fmt(block['cell_abs_solid_p95'])} | "
                f"{pct(block['share_cell_abs_gt_delta'])} | {pct(block['share_cell_abs_gt_4delta'])} | "
                f"{pct(block['share_cell_abs_gt_headroom'])} |"
            )

    print("\n(四) A1 各离散 g 最大的前若干单元的交集大小 (0 表示完全不重合)")
    for pair, count in payload["active_cells"]["pairwise_overlap"].items():
        if count < TOP_CELLS:
            print(f"    {pair}: {count}/{TOP_CELLS}")

    reference = payload.get("reference")
    if reference:
        shape = "x".join(str(value) for value in reference["fine_shape"])
        print(
            f"\n(五) A4 细网格双参考 ({shape}, {reference['fine_cells']} 单元, "
            f"{reference['refinement']} 倍加密)"
        )
        labels = list(reference["rows"])
        print("| 参考离散 | 细网格 max g | 归约回粗网格 max g | 秒 |")
        print("|---|---|---|---|")
        for label, row in reference["rows"].items():
            print(
                f"| {label} | {row['max_g_fine']:+.5f} | "
                f"{row['max_g_reduced']:+.5f} | {row['seconds']:.1f} |"
            )
        disagreement = reference["reference_disagreement"]
        ratio = reference["reference_disagreement_over_delta_g"]
        verdict = (
            "可信" if reference["reference_trustworthy"]
            else "不可信 —— A4 结论作废, 只保留 A1--A3"
        )
        print(
            f"\n[验收门] 两条参考彼此的逐单元最大分歧 = {disagreement:.5f} "
            f"({ratio:.1f} x delta_g): {verdict}"
        )
        print("\n各粗离散相对参考的逐单元最大偏差 (参考不可信时仅供存档, 不得引用)")
        print("| 粗离散 | " + " | ".join(f"vs {label}" for label in labels) + " |")
        print("|---|" + "---|" * len(labels))
        for discretization, row in reference["deviation"].items():
            cells = " | ".join(f"{row[label]:.5f}" for label in labels)
            print(f"| {discretization} | {cells} |")


def write_outputs(
    payload: dict[str, Any],
    fields: dict[str, np.ndarray],
    context: dict[str, Any],
) -> Path:
    """写 npz / json / vtu, 返回输出目录."""
    target = OUTPUT_DIR / CASE_ID / "postprocess" / "discretization_probe"
    target.mkdir(parents=True, exist_ok=True)
    label = payload["design"]

    np.savez_compressed(target / f"{label}__fields.npz", **fields)
    (target / f"{label}__probe.json").write_text(
        json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mesh = context["mesh"]
    for discretization in context["labels"]:
        cell_data = {
            "density": fields["density"],
            "pad_mask": fields["pad_mask"].astype(np.float64),
            "accepted_mask": fields["accepted_mask"].astype(np.float64),
            "g": fields[f"g__{discretization}"],
            "solid_stress_ratio": fields[f"s__{discretization}"],
            "apparent_stress_ratio": fields[f"vm_app__{discretization}"],
        }
        for sampling_order in SAMPLING_ORDERS:
            key = f"g_q{sampling_order}__{discretization}"
            if key in fields:
                cell_data[f"g_q{sampling_order}"] = fields[key]
        jump_key = f"jump__{discretization}"
        if jump_key in fields:
            cell_data["relative_traction_jump"] = fields[jump_key]
        write_vtu(
            mesh=mesh,
            filepath=str(target / f"{label}__{discretization}"),
            cell_data=cell_data,
        )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare.py discretization-probe",
        description="实验 A: 冻结构型下的离散敏感性探针 (散布 / 采样 / 牵引跳量).",
    )
    parser.add_argument(
        "--design", action="append", default=None, metavar="<目录名>",
        help=f"outputs/<case>/ 下的运行目录名, 可重复; 默认 {', '.join(DEFAULT_DESIGNS)}.",
    )
    parser.add_argument(
        "--reference", action="store_true",
        help="额外在加密网格上求两条高阶参考解 (分钟级, 默认关).",
    )
    parser.add_argument(
        "--no-jump", dest="jump", action="store_false",
        help="跳过 A3 牵引跳量 (只做散布与采样分析).",
    )
    return parser


def run_discretization_probe(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv or [])
    designs = arguments.design or list(DEFAULT_DESIGNS)
    root = OUTPUT_DIR / CASE_ID

    failed = False
    for run_dir in designs:
        design_dir = root / run_dir
        if not design_dir.is_dir():
            raise SystemExit(f"没有运行目录 {design_dir}.")
        payload, fields, context = probe_design(
            design_dir, arguments.reference, arguments.jump
        )
        payload["provenance"] = provenance.run_stamp()
        print_markdown(payload)
        target = write_outputs(payload, fields, context)
        print(f"\n[probe] {target.relative_to(OUTPUT_DIR.parent)}")

        gate = payload["checks"].get("huzhang_jump_gate")
        if gate is not None and not gate["passed"]:
            failed = True
        gate3 = payload["checks"].get("accepted_max_g_matches_summary")
        if gate3 is not None and not gate3["passed"]:
            failed = True
        if not all(
            value for key, value in payload["checks"].items()
            if key.startswith("resample_q1_matches_fun__")
        ):
            failed = True

    return 1 if failed else 0
