# -*- coding: utf-8 -*-
"""悬臂梁应力算例的冻结构型重分析: 图 5.8、5.10、5.11 的数据源.

每份最终构型冻结后, 在优化所用的离散下重解一次状态方程, 导出逐单元的表观应力比
``vm_app``、约束值 ``g`` 与内边法向牵引绝对跳量 ``A_e`` (见本模块第二节). Hu--Zhang
因 H(div, S) 协调该跳量恒为舍入量级, LFEM 由位移梯度逐单元恢复应力, 跳量非零;
论文 5.2.3 节的牵引连续性对比即取自本模块.

入口由 ``plot.py`` 派发, 本模块不直接执行::

    plot.py discretization-probe [--design <目录名>]...

产出写到 ``results/<case>/postprocess/discretization_probe/``: 逐构型一份 npz (逐单元
场, 供成图模块读取)、一份 json (跳量统计、验收门与 provenance) 与一份 vtu; 终端同时
打印 Markdown 表.

被动实体区 (summary 的 ``load_pad_centers`` / ``load_pad_radius`` 圈出的单元, 优化中
不施加约束且钉为实体) 不进入实体带统计. 两条验收门, 任一不过则返回码为 1:

1. Hu--Zhang 的内边相对跳量必须 < 1e-10 (协调性的直接后果), 否则跳量求值 (第二节) 有错;
2. 重分析在判据集合上的最大 ``g`` 必须复现 summary 的
   ``max_relative_violation_solid_region``, 否则掩码或阈值口径与优化器不同.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm

from soptx.postprocess.vtk_export import read_vtu_cell_data, write_vtu
from soptx.topology.constraints import build_exemption_mask

from config import OUTPUT_DIR, VIEW_DIR, bootstrap_source_path

bootstrap_source_path()

import provenance  # noqa: E402
import run_cantilever_stress  # noqa: E402

CASE_ID = run_cantilever_stress.CASE_ID
# 停止准则的容差 delta_g; 仅作跳量的尺度参照.
DELTA_G = run_cantilever_stress.STRESS_TOLERANCE
# 实体带: 密度高于该值且不在被动实体区的单元, 与 plots/stress_traction_jump 同口径.
SOLID_THRESHOLD = 0.9
# Hu--Zhang 相对跳量的硬门.
HUZHANG_JUMP_GATE = 1e-10

# 论文 5.2.3 节的六份构型 (pad 1.5 mm, 判据集合 rho >= 0.5): LFEM p=2..4 与
# Hu--Zhang k=2..4.
DEFAULT_DESIGNS: tuple[str, ...] = tuple(
    run_cantilever_stress.run_label(method, order)
    for method in run_cantilever_stress.METHODS
    for order in run_cantilever_stress.ORDERS
)


# ================================================================ 一、工具


def load_design(design_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """读取冻结构型及其运行摘要."""
    density_file = design_dir / "density_final.vtu"
    summary_file = design_dir / "summary.json"
    for path in (density_file, summary_file):
        if not path.is_file():
            raise SystemExit(f"缺少 {path}; 先把该组合跑出来, 或换一个运行目录.")
    density = np.asarray(read_vtu_cell_data(density_file, "density"), dtype=np.float64)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return density, summary


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


def design_discretization(run_dir: str) -> tuple[str, int]:
    """产出该构型的离散 ``(method, order)``; 解析不出即报错."""
    method = re.search(r"analyzer-(\w+?)__", run_dir)
    order = re.search(r"order-(\d+)", run_dir)
    if method is None or order is None:
        raise SystemExit(f"无法从运行目录名 {run_dir} 解析离散方法与阶次.")
    return method.group(1), int(order.group(1))


def pad_mask_from_summary(mesh, summary: dict[str, Any], n_cells: int) -> np.ndarray:
    """按运行摘要记录的圆心与半径复原被动实体区的单元掩码.

    半径为 0 或摘要无圆心时返回全 False; 单元数与摘要 ``pad_cells`` 不符即报错.
    """
    mask = np.zeros(n_cells, dtype=bool)
    radius = float(summary.get("load_pad_radius", 0.0) or 0.0)
    centers = summary.get("load_pad_centers") or []
    if radius <= 0.0 or not centers:
        return mask
    exempt = build_exemption_mask(mesh=mesh, centers=centers, radius=radius)
    mask[np.asarray(bm.to_numpy(exempt)).astype(bool)] = True
    expected = summary.get("load_pad_cells", summary.get("pad_cells"))
    if expected is not None and int(mask.sum()) != int(expected):
        raise RuntimeError(
            f"被动实体区单元数 {int(mask.sum())} 与摘要记录的 {expected} 不符."
        )
    return mask


def _per_cell_max(values: Any) -> np.ndarray:
    """把 (NC, ...) 的评价点量约化为逐单元最大值."""
    array = np.asarray(bm.to_numpy(values), dtype=np.float64)
    return array.reshape(array.shape[0], -1).max(axis=1)


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


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def evaluate(method: str, order: int, design: np.ndarray) -> dict[str, Any]:
    """在冻结构型上做一次前向求解, 返回逐单元的表观应力比与约束值.

    分析链由 ``run_cantilever_stress.build`` 组装 (与优化运行同一份代码), 但垫片半径
    取 0: 在不豁免的约束对象上求值, 被动实体区的真实读数一并取回; 构型由
    ``rho[:] = design`` 整体覆写, 不经过过滤链, 实体保留与否不影响前向求解.
    """
    parts = run_cantilever_stress.build(method, order, load_pad_radius=0.0)
    pipeline = SimpleNamespace(**parts)
    rho = parts["density"]
    if rho.shape[0] != design.shape[0]:
        raise SystemExit(
            f"{method}-{order}: 网格单元数 {rho.shape[0]} 与构型 {design.shape[0]} 不符; "
            "冻结评价必须与产出该构型的运行同网格."
        )
    rho[:] = design
    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    constraint = pipeline.stress_constraint
    return {
        "pipeline": pipeline,
        "state": state,
        "constraint_value": _per_cell_max(constraint.fun(rho, state)),
        "apparent_stress_ratio": _per_cell_max(constraint.compute_stress_measure(rho, state)),
        "seconds": time.perf_counter() - started,
    }


# ================================================================ 二、内边法向牵引跳量
#
# 跳量一律测表观应力 sigma^app, 不测实体应力: 连续介质中即便模量随空间变化, 真实
# 牵引 sigma . n 也跨面连续 (跳的是应变), 而 sigma^sol = sigma^app / m_E 因 m_E 逐单元
# 常值必然跳变. 两族的表观应力分别为 LFEM 的 m_E(rho) * D B u 与 Hu--Zhang 原始应力
# 自由度的取值. soptx 中没有可复用的两侧求值设施 (JumpPenaltyIntegrator
# 跳的是位移基函数而非给定场), 故此处自建.
#
# 配序是唯一的正确性陷阱: 同一条边从两侧单元提升到重心坐标后, 求积点在物理空间的
# 走向可能相反. 用两侧物理点直接比对判定, 比对失败即抛错; 边上 Gauss 权重对称, 翻转
# bcs 等价于沿求积点轴反转求值结果, 因此只需一次求值.


# 三角形局部边的顶点编号, 与 soptx.mesh.TriangleMesh.localFace 一致; 仅用于断言.
_TRIANGLE_LOCAL_FACE = ((1, 2), (2, 0), (0, 1))


def _to_numpy(values: Any) -> np.ndarray:
    return np.asarray(bm.to_numpy(values), dtype=np.float64)


def _voigt_traction(stress: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """由 Voigt 应力 ``[xx, yy, xy]`` 与单位法向算法向牵引.

    Parameters
    ----------
    stress : ndarray, shape (NF, NQ, 3)
        Voigt 序的二维应力.
    normal : ndarray, shape (NF, 2)
        逐边单位法向.

    Returns
    -------
    ndarray, shape (NF, NQ, 2)
        牵引向量 ``t_i = sigma_ij n_j``.
    """
    nx = normal[:, 0][:, None]
    ny = normal[:, 1][:, None]
    tx = stress[..., 0] * nx + stress[..., 2] * ny
    ty = stress[..., 2] * nx + stress[..., 1] * ny
    return np.stack([tx, ty], axis=-1)


def _edge_l2(values: np.ndarray, weights: np.ndarray, measure: np.ndarray) -> np.ndarray:
    """逐边的 ``L2(e)`` 范数: ``sqrt(|e| * sum_q w_q |v_q|^2)``."""
    squared = (values ** 2).sum(axis=-1)
    return np.sqrt(np.maximum(measure * (squared * weights[None, :]).sum(axis=1), 0.0))


def _apparent_stress_lfem(pipeline, state, cells, bcs_cell) -> np.ndarray:
    """LFEM 在给定单元与重心坐标上的表观应力, 形状 ``(n, NQ, 3)``."""
    analyzer = pipeline.analyzer
    material = analyzer.material
    gphi = analyzer.scalar_space.grad_basis(bcs_cell, index=cells, variable="x")
    strain_matrix = material.strain_matrix(
        dof_priority=analyzer.tensor_space.dof_priority,
        gphi=gphi,
    )
    cell_to_dof = analyzer.tensor_space.cell_to_dof()[cells]
    displacement = state["displacement"][cell_to_dof]
    solid = _to_numpy(material.calculate_stress_vector(strain_matrix, displacement))

    # 表观应力要逐单元乘该单元的 m_E; state['stiffness_ratio'] 由约束的 fun 写入,
    # 与 LagrangeStressConstraint 读的是同一份缓存, 故此处不另取.
    if "stiffness_ratio" not in state:
        raise RuntimeError(
            "state 缺 stiffness_ratio: 需先调用一次 constraint.fun 填充状态."
        )
    scale = _to_numpy(state["stiffness_ratio"])[_to_numpy(cells).astype(np.int64)]
    return solid * scale[:, None, None]


def _apparent_stress_huzhang(pipeline, state, cells, bcs_cell) -> np.ndarray:
    """Hu--Zhang 在给定单元与重心坐标上的表观应力, 形状 ``(n, NQ, 3)``."""
    space = pipeline.analyzer.huzhang_space
    # 原生 value 内部完成松弛坐标到基函数坐标的变换 (TM), 不可绕过.
    stress = _to_numpy(space.value(state["stress"][:], bcs_cell, index=cells))
    if stress.shape[-1] != 3:
        raise NotImplementedError("仅支持二维问题的应力分量重排.")
    # 原生分量序 [xx, xy, yy] -> Voigt [xx, yy, xy].
    return stress[..., [0, 2, 1]]


_STRESS_EVALUATORS = {
    "lfem": _apparent_stress_lfem,
    "huzhang": _apparent_stress_huzhang,
}


def _side_values(pipeline, state, cells, local_face, bcs_face):
    """把一侧的表观应力与物理点按局部边编号分组求值.

    Parameters
    ----------
    cells : ndarray, shape (NF,)
        该侧的相邻单元编号.
    local_face : ndarray, shape (NF,)
        该边在该侧单元内的局部编号.
    bcs_face : ndarray, shape (NQ, 2)
        边上求积点的重心坐标.

    Returns
    -------
    stress : ndarray, shape (NF, NQ, 3)
    points : ndarray, shape (NF, NQ, 2)
    """
    evaluator = _STRESS_EVALUATORS[pipeline.method]
    mesh = pipeline.mesh
    n_face, n_quadrature = cells.shape[0], bcs_face.shape[0]
    stress = np.empty((n_face, n_quadrature, 3), dtype=np.float64)
    points = np.empty((n_face, n_quadrature, 2), dtype=np.float64)

    for local_index in range(3):
        selected = np.flatnonzero(local_face == local_index)
        if selected.size == 0:
            continue
        # 边重心坐标提升到单元重心坐标: 在缺席顶点处插 0.
        bcs_cell = bm.insert(bm.array(bcs_face), local_index, 0.0, axis=-1)
        group_cells = bm.array(cells[selected])
        stress[selected] = evaluator(pipeline, state, group_cells, bcs_cell)
        points[selected] = _to_numpy(mesh.bc_to_point(bcs_cell, index=group_cells))

    return stress, points


def interior_edge_traction_jump(
    pipeline,
    state: dict,
    integration_order: int | None = None,
) -> dict[str, np.ndarray]:
    """计算内边上表观应力的法向牵引跳量.

    Parameters
    ----------
    pipeline : StressOptimizationPipeline
        已建好的分析流水线, 需带 ``mesh``/``analyzer``/``method``.
    state : dict
        ``analyzer.solve_state`` 的返回, LFEM 需 ``displacement``,
        Hu--Zhang 需 ``stress``.
    integration_order : int, optional
        边上求积阶次; 默认取 ``2 * order + 2``, 与分析器的单元求积同档.

    Returns
    -------
    dict
        ``integration_order`` 边上求积阶次;
        ``interior_faces`` (NF_int,) 全局边编号;
        ``relative_jump`` (NF_int,) ``||[[sigma n]]||_{L2(e)} / ||{sigma n}||_{L2(e)}``
        (分母加地板);
        ``well_scaled`` (NF_int,) 布尔, 分母是否足够大到让比值有意义;
        ``cell_rms_jump`` (NC,) 逐边跳量均方根 ``sqrt(sum_q w_q |[[sigma n]]|^2)``
        散射回单元的逐单元最大值; 与边长无关、与牵引同量纲, 按许用应力归一后即
        ``A_e``. 相对跳量在牵引本身趋零的边上被分母放大, 不能与余量对照, 只用于
        混合法机器零的验收门.

    Raises
    ------
    RuntimeError
        两侧求积点无法配上 (既不同序也不反序), 说明提升或分组有误.
    """
    mesh = pipeline.mesh
    order = int(pipeline.order)
    if integration_order is None:
        integration_order = 2 * order + 2

    local_face = _to_numpy(mesh.localFace).astype(np.int64)
    if local_face.shape != (3, 2) or tuple(map(tuple, local_face)) != _TRIANGLE_LOCAL_FACE:
        raise RuntimeError(f"三角形局部边约定与预期不符: {local_face.tolist()}")

    face_to_cell = _to_numpy(mesh.face_to_cell()).astype(np.int64)
    interior = np.flatnonzero(face_to_cell[:, 0] != face_to_cell[:, 1])
    if interior.size == 0:
        raise RuntimeError("网格没有内边.")

    quadrature = mesh.quadrature_formula(integration_order, "face")
    bcs_face, weights = quadrature.get_quadrature_points_and_weights()
    bcs_face = _to_numpy(bcs_face)
    weights = _to_numpy(weights)
    # 翻转 bcs 等价于反转求值结果, 这一步只在权重对称时成立.
    if not np.allclose(weights, weights[::-1]):
        raise RuntimeError("边求积权重非对称, 不能用反转结果代替翻转 bcs.")

    left_stress, left_points = _side_values(
        pipeline, state, face_to_cell[interior, 0], face_to_cell[interior, 2], bcs_face
    )
    right_stress, right_points = _side_values(
        pipeline, state, face_to_cell[interior, 1], face_to_cell[interior, 3], bcs_face
    )

    # 配序: 逐边判定右侧求积点是同序还是反序, 反序者沿求积点轴翻转求值结果.
    scale = float(np.abs(left_points).max()) + 1.0
    tolerance = 1e-10 * scale
    aligned = np.abs(left_points - right_points).max(axis=(1, 2)) < tolerance
    reversed_match = np.abs(left_points - right_points[:, ::-1]).max(axis=(1, 2)) < tolerance
    if not np.all(aligned | reversed_match):
        bad = int(np.flatnonzero(~(aligned | reversed_match))[0])
        raise RuntimeError(
            f"内边 {int(interior[bad])} 的两侧求积点既不同序也不反序; "
            "重心坐标提升或分组有误."
        )
    flip = reversed_match & ~aligned
    right_stress[flip] = right_stress[flip][:, ::-1]

    normal = _to_numpy(mesh.face_unit_normal())[interior]
    measure = _to_numpy(mesh.entity_measure("face"))[interior]
    left_traction = _voigt_traction(left_stress, normal)
    right_traction = _voigt_traction(right_stress, normal)

    jump_l2 = _edge_l2(left_traction - right_traction, weights, measure)
    mean_l2 = _edge_l2(0.5 * (left_traction + right_traction), weights, measure)

    # 空洞区两侧牵引都趋于 0, 比值无意义; 用全局尺度设地板并标出可用的那批.
    reference = float(mean_l2.max()) if mean_l2.size else 0.0
    floor = max(reference * 1e-12, np.finfo(np.float64).tiny)
    relative_jump = jump_l2 / np.maximum(mean_l2, floor)
    well_scaled = mean_l2 > reference * 1e-6

    cell_to_face = _to_numpy(mesh.cell_to_face()).astype(np.int64)
    rms_jump = jump_l2 / np.sqrt(np.maximum(measure, np.finfo(np.float64).tiny))
    per_face_rms = np.zeros(mesh.number_of_faces(), dtype=np.float64)
    per_face_rms[interior] = rms_jump
    cell_rms_jump = per_face_rms[cell_to_face].max(axis=1)

    return {
        "integration_order": integration_order,
        "interior_faces": interior,
        "relative_jump": relative_jump,
        "well_scaled": well_scaled,
        "cell_rms_jump": cell_rms_jump,
    }


# ================================================================ 三、主流程


def probe_design(design_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Any]:
    """在构型自身的离散下重分析一份冻结构型.

    Returns
    -------
    payload : dict
        聚合结果, 可直接序列化为 json.
    fields : dict of ndarray
        逐单元场, 写入 npz 与 vtu.
    mesh : Mesh
        出 vtu 用的网格.
    """
    design, design_summary = load_design(design_dir)
    method, order = design_discretization(design_dir.name)
    discretization = f"{method}-{order}"
    stress_limit = run_cantilever_stress.STRESS_LIMIT
    solid_threshold = design_summary.get("acceptance_solid_threshold")

    result = evaluate(method, order, design)
    pipeline = result["pipeline"]
    mesh = pipeline.mesh
    g = result["constraint_value"]

    # 被动实体区不施加约束且钉为实体; 判据集合是停止判据的评价集合 (区外且
    # rho >= 阈值), 无阈值记录时即区外全部单元.
    pad = pad_mask_from_summary(mesh, design_summary, design.shape[0])
    accepted = ~pad
    if solid_threshold is not None:
        accepted &= design >= float(solid_threshold)

    # 牵引跳量: 相对量只用于 Hu--Zhang 硬门; 绝对量 RMS / sigma_bar 与 g 同尺度,
    # 逐单元 A_e 取单元各内边的最大值, 统计限于实体带.
    jump = interior_edge_traction_jump(pipeline, result["state"])
    relative = jump["relative_jump"][jump["well_scaled"]]
    abs_cell = jump["cell_rms_jump"] / stress_limit
    solid_cells = (design > SOLID_THRESHOLD) & ~pad
    traction_jump = {
        "integration_order": jump["integration_order"],
        "interior_faces": int(jump["interior_faces"].size),
        "relative_jump_max": float(relative.max()) if relative.size else None,
        "solid_cells": int(solid_cells.sum()),
        **{f"cell_abs_solid_{key}": value for key, value in _stats(abs_cell[solid_cells]).items()},
    }

    checks: dict[str, Any] = {}
    if method == "huzhang":
        value = traction_jump["relative_jump_max"]
        checks["huzhang_jump_gate"] = {
            "value": value,
            "threshold": HUZHANG_JUMP_GATE,
            "passed": value is not None and value < HUZHANG_JUMP_GATE,
        }
    recorded = design_summary.get("max_relative_violation_solid_region")
    if recorded is not None:
        reproduced = float(g[accepted].max())
        checks["accepted_max_g_matches_summary"] = {
            "reproduced": reproduced,
            "recorded": float(recorded),
            "passed": bool(np.isclose(reproduced, float(recorded), rtol=1e-6, atol=1e-9)),
        }

    fields = {
        "density": design,
        "pad_mask": pad,
        "accepted_mask": accepted,
        f"g__{discretization}": g,
        f"vm_app__{discretization}": result["apparent_stress_ratio"],
        f"absjump__{discretization}": abs_cell,
    }
    payload: dict[str, Any] = {
        "case_id": CASE_ID,
        "design": design_label(design_dir.name),
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
        "discretization": discretization,
        "delta_g": DELTA_G,
        "pad_cells": int(pad.sum()),
        "accepted_cells": int(accepted.sum()),
        "max_g_accepted": float(g[accepted].max()),
        "seconds": float(result["seconds"]),
        "traction_jump": traction_jump,
        "checks": checks,
    }
    return payload, fields, mesh


def print_markdown(payload: dict[str, Any]) -> None:
    """终端打印重分析结果与验收门."""
    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.3e}"

    jump = payload["traction_jump"]
    print(
        f"\n{payload['case_id']}: 冻结构型 {payload['design']} ({payload['design_run_dir']}), "
        f"离散 {payload['discretization']}, 被动实体区 {payload['pad_cells']} 单元, "
        f"判据集合 {payload['accepted_cells']} 单元"
    )
    print("| 离散 | 判据集合 max g | 实体带单元数 | A_e 中位 | A_e P95 | A_e 最大 | 相对跳量最大 |")
    print("|---|---|---|---|---|---|---|")
    print(
        f"| {payload['discretization']} | {payload['max_g_accepted']:+.5f} | "
        f"{jump['solid_cells']} | {fmt(jump['cell_abs_solid_median'])} | "
        f"{fmt(jump['cell_abs_solid_p95'])} | {fmt(jump['cell_abs_solid_max'])} | "
        f"{fmt(jump['relative_jump_max'])} |"
    )
    gate = payload["checks"].get("huzhang_jump_gate")
    if gate is not None:
        verdict = "通过" if gate["passed"] else "未通过 —— 跳量求值有错"
        print(f"[验收 1] Hu--Zhang 相对跳量 < {gate['threshold']:g}: {verdict}")
    gate = payload["checks"].get("accepted_max_g_matches_summary")
    if gate is not None:
        verdict = "通过" if gate["passed"] else "未通过 —— 掩码口径与优化器不符"
        print(
            f"[验收 2] 判据集合 max g 复现 = {gate['reproduced']:+.6e}, "
            f"summary 记录 = {gate['recorded']:+.6e}: {verdict}"
        )


def write_outputs(payload: dict[str, Any], fields: dict[str, np.ndarray], mesh) -> Path:
    """写 npz / json / vtu, 返回输出目录.

    json (跳量统计、验收门与 provenance) 入库; npz (逐单元场, 每构型约 12 MB) 留在
    results/ 但不入库, 缺失时重跑本探针即可; vtu 只供 ParaView 查看, 写到 Windows
    查看目录 ``VIEW_DIR`` 下的同名路径, 该盘不可用时退回 results/ (同样不入库).
    """
    target = OUTPUT_DIR / CASE_ID / "postprocess" / "discretization_probe"
    target.mkdir(parents=True, exist_ok=True)
    view = (VIEW_DIR / CASE_ID / "postprocess" / "discretization_probe"
            if VIEW_DIR.parent.is_dir() else target)
    view.mkdir(parents=True, exist_ok=True)
    label = payload["design"]
    discretization = payload["discretization"]

    np.savez_compressed(target / f"{label}__fields.npz", **fields)
    (target / f"{label}__probe.json").write_text(
        json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_vtu(
        mesh=mesh,
        filepath=str(view / f"{label}__{discretization}"),
        cell_data={
            "density": fields["density"],
            "pad_mask": fields["pad_mask"].astype(np.float64),
            "accepted_mask": fields["accepted_mask"].astype(np.float64),
            "g": fields[f"g__{discretization}"],
            "apparent_stress_ratio": fields[f"vm_app__{discretization}"],
            "traction_jump": fields[f"absjump__{discretization}"],
        },
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot.py discretization-probe",
        description="应力算例: 冻结构型在自身离散下重分析, 导出应力比与牵引跳量.",
    )
    parser.add_argument(
        "--design", action="append", default=None, metavar="<目录名>",
        help="results/<case>/ 下的运行目录名, 可重复; 默认论文 5.2.3 节的六份构型.",
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
        payload, fields, mesh = probe_design(design_dir)
        payload["provenance"] = provenance.run_stamp()
        print_markdown(payload)
        target = write_outputs(payload, fields, mesh)
        print(f"[probe] {target.relative_to(OUTPUT_DIR.parent)}")
        if not all(check["passed"] for check in payload["checks"].values()):
            failed = True

    return 1 if failed else 0
