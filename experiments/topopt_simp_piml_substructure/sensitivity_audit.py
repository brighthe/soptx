"""PIML Route A 拓扑优化的诊断性局部能量与灵敏度审计。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


AUDIT_SCHEMA_VERSION = "piml-sensitivity-audit-v1"


def _relative_l2(approximate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(approximate - reference))
    denominator = max(float(np.linalg.norm(reference)), 1.0e-14)
    return numerator / denominator


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-28:
        return 1.0 if np.allclose(left, right, atol=1.0e-14, rtol=0.0) else 0.0
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def _json_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            result[str(name)] = array.item()
        else:
            result[str(name)] = array.tolist()
    return result


def select_audit_positions(count: int, limit: int) -> np.ndarray:
    """在保持确定性和空间顺序覆盖的前提下选择 audit 位置。"""
    if count < 0:
        raise ValueError("count 不能为负数。")
    if limit <= 0:
        raise ValueError("limit 必须为正整数。")
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    positions = np.linspace(0, count - 1, num=limit, dtype=np.int64)
    if len(np.unique(positions)) != limit:
        raise RuntimeError("确定性 audit 采样产生了重复位置。")
    return positions


def audit_accepted_local_responses(
    *,
    substructure_ids: Sequence[int],
    boundary_displacement: np.ndarray,
    predicted_interior_displacement: np.ndarray,
    exact_interior_displacement: np.ndarray,
    density_cell: np.ndarray,
    interior_dofs: np.ndarray,
    boundary_dofs: np.ndarray,
    cell_to_dof: np.ndarray,
    unit_cell_stiffness: np.ndarray,
    simp_penalty: float,
    rho_min: float,
    gate_metrics: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """比较 accepted PIML 恢复与同一边界迹下的 Exact Schur 恢复。

    返回 Exact 单元能量以及逐子结构诊断记录。这里不重新求解全局接口系统，
    因而隔离的是局部恢复关系对能量与 SIMP 灵敏度的影响。
    """
    ids = tuple(int(index) for index in substructure_ids)
    boundary = np.asarray(boundary_displacement, dtype=np.float64)
    predicted_interior = np.asarray(
        predicted_interior_displacement, dtype=np.float64
    )
    exact_interior = np.asarray(exact_interior_displacement, dtype=np.float64)
    density = np.asarray(density_cell, dtype=np.float64)
    i_dofs = np.asarray(interior_dofs, dtype=np.int64)
    b_dofs = np.asarray(boundary_dofs, dtype=np.int64)
    cell2dof = np.asarray(cell_to_dof, dtype=np.int64)
    K0 = np.asarray(unit_cell_stiffness, dtype=np.float64)

    batch_size = len(ids)
    if not (
        len(boundary)
        == len(predicted_interior)
        == len(exact_interior)
        == len(density)
        == len(gate_metrics)
        == batch_size
    ):
        raise ValueError("audit 输入的 batch 长度不一致。")
    if density.ndim != 2:
        raise ValueError("density_cell 必须为 (batch, n_cell)。")
    if cell2dof.ndim != 2 or density.shape[1] != cell2dof.shape[0]:
        raise ValueError("density_cell 与 cell_to_dof 的单元数不一致。")
    all_dofs = np.concatenate((i_dofs, b_dofs))
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1] or K0.shape[0] != cell2dof.shape[1]:
        raise ValueError(
            "unit_cell_stiffness 必须为与 cell_to_dof 局部列数一致的方阵。"
        )
    n_dofs = int(np.max(cell2dof)) + 1
    if not np.array_equal(
        np.sort(all_dofs),
        np.arange(n_dofs, dtype=np.int64),
    ):
        raise ValueError("interior_dofs 与 boundary_dofs 必须构成完整不重叠分区。")
    if boundary.shape[1] != len(b_dofs):
        raise ValueError("boundary_displacement 与 boundary_dofs 不匹配。")
    if predicted_interior.shape != exact_interior.shape:
        raise ValueError("PIML 与 Exact 内部位移形状不一致。")
    if predicted_interior.shape[1] != len(i_dofs):
        raise ValueError("interior_displacement 与 interior_dofs 不匹配。")

    predicted_local = np.zeros((batch_size, n_dofs), dtype=np.float64)
    exact_local = np.zeros_like(predicted_local)
    predicted_local[:, b_dofs] = boundary
    exact_local[:, b_dofs] = boundary
    predicted_local[:, i_dofs] = predicted_interior
    exact_local[:, i_dofs] = exact_interior

    predicted_element = predicted_local[:, cell2dof]
    exact_element = exact_local[:, cell2dof]
    predicted_energy = np.sum((predicted_element @ K0) * predicted_element, axis=-1)
    exact_energy = np.sum((exact_element @ K0) * exact_element, axis=-1)

    coefficient_derivative = simp_penalty * np.power(
        density,
        simp_penalty - 1.0,
    )
    if rho_min != 0.0:
        coefficient_derivative *= 1.0 - rho_min
    predicted_sensitivity = -coefficient_derivative * predicted_energy
    exact_sensitivity = -coefficient_derivative * exact_energy

    records: list[dict[str, Any]] = []
    for local_index, substructure_id in enumerate(ids):
        predicted_energy_row = predicted_energy[local_index]
        exact_energy_row = exact_energy[local_index]
        predicted_sensitivity_row = predicted_sensitivity[local_index]
        exact_sensitivity_row = exact_sensitivity[local_index]
        records.append(
            {
                "substructure_id": substructure_id,
                "gate_metrics": _json_metrics(gate_metrics[local_index]),
                "density_min": float(np.min(density[local_index])),
                "density_mean": float(np.mean(density[local_index])),
                "density_max": float(np.max(density[local_index])),
                "density_std": float(np.std(density[local_index])),
                "predicted_interior_norm": float(
                    np.linalg.norm(predicted_interior[local_index])
                ),
                "exact_interior_norm": float(
                    np.linalg.norm(exact_interior[local_index])
                ),
                "interior_displacement_relative_l2": _relative_l2(
                    predicted_interior[local_index],
                    exact_interior[local_index],
                ),
                "cell_energy_relative_l2": _relative_l2(
                    predicted_energy_row,
                    exact_energy_row,
                ),
                "sensitivity_relative_l2": _relative_l2(
                    predicted_sensitivity_row,
                    exact_sensitivity_row,
                ),
                "sensitivity_cosine_similarity": _cosine_similarity(
                    predicted_sensitivity_row,
                    exact_sensitivity_row,
                ),
                "sensitivity_max_abs_error": float(
                    np.max(
                        np.abs(predicted_sensitivity_row - exact_sensitivity_row)
                    )
                ),
            }
        )
    return exact_energy, records


def filtered_sensitivity_metrics(
    approximate: np.ndarray,
    hybrid_exact_reference: np.ndarray,
) -> dict[str, float]:
    """比较正式 PIML 过滤梯度与 accepted 局部 Exact 替换后的混合参考。"""
    approximate_array = np.asarray(approximate, dtype=np.float64).reshape(-1)
    reference_array = np.asarray(
        hybrid_exact_reference, dtype=np.float64
    ).reshape(-1)
    if approximate_array.shape != reference_array.shape:
        raise ValueError("过滤灵敏度形状不一致。")
    return {
        "relative_l2": _relative_l2(approximate_array, reference_array),
        "cosine_similarity": _cosine_similarity(
            approximate_array,
            reference_array,
        ),
        "max_abs_error": float(
            np.max(np.abs(approximate_array - reference_array))
        ),
    }


def summarize_local_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总逐子结构 audit 记录，同时保留原始记录供定位。"""
    keys = (
        "interior_displacement_relative_l2",
        "cell_energy_relative_l2",
        "sensitivity_relative_l2",
    )
    summary: dict[str, Any] = {"accepted_count": len(records)}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        if len(values) == 0:
            summary[key] = {"median": None, "p95": None, "max": None}
        else:
            summary[key] = {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }
    cosine = np.asarray(
        [record["sensitivity_cosine_similarity"] for record in records],
        dtype=np.float64,
    )
    summary["sensitivity_cosine_similarity"] = (
        {"median": None, "p05": None, "min": None}
        if len(cosine) == 0
        else {
            "median": float(np.median(cosine)),
            "p05": float(np.quantile(cosine, 0.05)),
            "min": float(np.min(cosine)),
        }
    )
    return summary


def summarize_audit_iterations(
    iterations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总全部迭代的局部误差及混合参考过滤梯度误差。"""
    records = [
        record
        for iteration in iterations
        for record in iteration.get("substructures", [])
    ]
    local = summarize_local_records(records)
    filtered = [
        iteration["filtered_sensitivity"]
        for iteration in iterations
        if iteration.get("filtered_sensitivity") is not None
    ]
    filtered_relative = np.asarray(
        [item["relative_l2"] for item in filtered], dtype=np.float64
    )
    filtered_cosine = np.asarray(
        [item["cosine_similarity"] for item in filtered], dtype=np.float64
    )
    local["iteration_count"] = len(iterations)
    local["filtered_full_reference_iteration_count"] = len(filtered)
    local["filtered_sensitivity_relative_l2"] = (
        {
            "median": float(np.median(filtered_relative)),
            "p95": float(np.quantile(filtered_relative, 0.95)),
            "max": float(np.max(filtered_relative)),
        }
        if len(filtered_relative)
        else {"median": None, "p95": None, "max": None}
    )
    local["filtered_sensitivity_cosine_similarity"] = (
        {
            "median": float(np.median(filtered_cosine)),
            "p05": float(np.quantile(filtered_cosine, 0.05)),
            "min": float(np.min(filtered_cosine)),
        }
        if len(filtered_cosine)
        else {"median": None, "p05": None, "min": None}
    )
    return local
