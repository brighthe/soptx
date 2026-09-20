# -*- coding: utf-8 -*-
"""内边法向牵引跳量 ``[[sigma . n]]``: 逐单元应力约束是否良定义的度量.

局部应力约束逐单元读一个 ``sigma_vM`` 值. 这个读数要有意义, 前提是"该单元的应力"
本身良定义. 对 Hu--Zhang 元, 应力是 ``H(div, S)`` 协调的原始变量, 法向牵引跨边连续,
``[[sigma . n]] = 0`` 成立到舍入; 对 LFEM, 应力由位移求导得到, 只在 ``L2`` 意义下
收敛, 跨边法向牵引有跳跃, 于是"这个单元的应力是多少"依赖于在哪里取值、怎么恢复.
若该歧义量与实体带的可行余量 (1%--4%) 同量级, 则约束判定本身不可靠.

跳量一律测**表观应力** ``sigma^app``, 不测实体应力: 连续介质中即便模量随空间变化,
真实牵引 ``sigma . n`` 也是跨面连续的 (跳的是应变), 而实体应力
``sigma^sol = sigma^app / m_E`` 因 ``m_E`` 逐单元常值必然跳变, 测它没有意义.
两族的表观应力分别为::

    LFEM       sigma^app = m_E(rho) * D B u        (逐单元乘该单元的 m_E)
    Hu--Zhang  sigma^app = 原始应力自由度的取值     (无需缩放)

soptx 与 fealpy 中没有可复用的两侧求值设施: ``JumpPenaltyIntegrator`` 跳的是位移
基函数而非给定场, 且未处理两侧求积点的配序; ``LagrangeFESpace.cell_grad_basis_on_face``
依赖当前 mesh 已移除的 ``update_bcs``/``face_to_cell(index)`` 签名. 故本模块自建.

配序是唯一的正确性陷阱: 同一条边从两侧单元提升到重心坐标后, 求积点在物理空间的
走向可能相反. 本模块用两侧的物理点直接比对来判定, 并在比对失败时抛错, 不做静默
猜测; 由于边上 Gauss 规则的权重对称, 翻转 bcs 等价于沿求积点轴反转求值结果, 因此
只需一次求值.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm

# 三角形局部边的顶点编号, 与 fealpy TriangleMesh.localFace 一致; 仅用于断言.
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
        ``interior_faces`` (NF_int,) 全局边编号;
        ``cells`` (NF_int, 2) 两侧单元;
        ``jump_l2`` (NF_int,) ``||[[sigma n]]||_{L2(e)}``;
        ``mean_l2`` (NF_int,) ``||{sigma n}||_{L2(e)}``;
        ``relative_jump`` (NF_int,) 二者之比 (分母加地板);
        ``well_scaled`` (NF_int,) 布尔, 分母是否足够大到让比值有意义;
        ``cell_relative_jump`` (NC,) 散射回单元的逐单元最大相对跳量;
        ``left_traction`` / ``right_traction`` (NF_int, NQ, 2) 两侧在同序求积点上
        的牵引 ``sigma n`` (已按左侧配序), ``points`` (NF_int, NQ, 2) 求积点物理坐标,
        ``weights`` (NQ,) 求积权重, ``normal`` (NF_int, 2) 与 ``measure`` (NF_int,)
        为逐边单位法向与边长; 供剖面图直接读两侧读数, 不必再求值;
        ``rms_jump`` (NF_int,) 逐边跳量的均方根 ``sqrt(sum_q w_q |[[sigma n]]|^2)``,
        与边长无关、与牵引同量纲, 是能按许用应力归一后直接与 g 对照的绝对跳量;
        ``cell_rms_jump`` (NC,) 散射回单元的逐单元最大 RMS 跳量. 相对跳量在
        牵引本身趋零的边 (杆件内的水平边, sigma_yy ~ 0) 上被分母放大, 不能与
        余量对照, 只用于混合法机器零的验收门.

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
    per_face = np.zeros(mesh.number_of_faces(), dtype=np.float64)
    per_face[interior] = relative_jump
    cell_relative_jump = per_face[cell_to_face].max(axis=1)

    rms_jump = jump_l2 / np.sqrt(np.maximum(measure, np.finfo(np.float64).tiny))
    per_face_rms = np.zeros(mesh.number_of_faces(), dtype=np.float64)
    per_face_rms[interior] = rms_jump
    cell_rms_jump = per_face_rms[cell_to_face].max(axis=1)

    return {
        "integration_order": integration_order,
        "interior_faces": interior,
        "cells": face_to_cell[interior, :2],
        "jump_l2": jump_l2,
        "mean_l2": mean_l2,
        "relative_jump": relative_jump,
        "well_scaled": well_scaled,
        "cell_relative_jump": cell_relative_jump,
        "rms_jump": rms_jump,
        "cell_rms_jump": cell_rms_jump,
        "left_traction": left_traction,
        "right_traction": right_traction,
        "points": left_points,
        "weights": weights,
        "normal": normal,
        "measure": measure,
    }
