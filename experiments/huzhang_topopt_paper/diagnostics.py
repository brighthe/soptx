"""Hu--Zhang 拓扑优化论文实验能量诊断、残差计算与结果输出模块."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm


def relative_residual(pipeline: Any, state: dict[str, Any]) -> float:
    """计算状态解的真相对平衡残差.

    参数:
        pipeline: 优化/分析管线对象.
        state: 求解器返回的状态字典.

    返回:
        相对平衡残差标量.
    """
    if pipeline.method == "huzhang":
        return float(pipeline.analyzer.relative_state_residual())

    analyzer = pipeline.analyzer
    displacement = state["displacement"]
    residual = analyzer.stiffness_matrix.matmul(displacement[:]) - analyzer.force_vector
    is_boundary = analyzer.tensor_space.is_boundary_dof(
        threshold=pipeline.problem.is_dirichlet_boundary(), method="interp"
    )
    numerator = float(bm.linalg.norm(residual[~is_boundary]))
    denominator = max(float(bm.linalg.norm(analyzer.force_vector[~is_boundary])), 1.0e-30)
    return numerator / denominator


def energy_identity_diagnostics(
    pipeline: Any,
    state: dict[str, Any],
) -> dict[str, float | str]:
    """计算各离散方法在给定密度场下可直接验证的能量恒等式.

    参数:
        pipeline: 组装好的分析管线对象.
        state: 前向求解状态字典.

    返回:
        包含能量分量与相对对偶一致性缺陷的字典.
    """
    if pipeline.method == "lfem":
        displacement = state["displacement"][:]
        force = pipeline.analyzer.force_vector
        stiffness = pipeline.analyzer.stiffness_matrix
        external_work = float(bm.einsum("i, i ->", displacement, force[:]))
        strain_energy = float(
            bm.einsum("i, i ->", displacement, stiffness.matmul(displacement))
        )
        return {
            "identity": "fTu_equals_uKu",
            "external_work": external_work,
            "internal_energy": strain_energy,
            "relative_defect": abs(external_work - strain_energy)
            / max(abs(external_work), 1.0e-30),
        }

    stress = state["stress"][:]
    displacement = state["displacement"][:]
    stress_matrix = pipeline.analyzer.get_stress_matrix(
        rho_val=pipeline.density_distribution
    )
    mix_matrix = pipeline.analyzer.mix_matrix
    complementary_energy = float(
        bm.einsum("i, i ->", stress, stress_matrix.matmul(stress))
    )
    coupling_work = float(
        bm.einsum("i, i ->", stress, mix_matrix.matmul(displacement))
    )
    traction_dual_work = complementary_energy + coupling_work
    return {
        "identity": "sigmaAsigma_plus_sigmaBu_equals_traction_dual_work",
        "complementary_energy": complementary_energy,
        "coupling_work": coupling_work,
        "traction_dual_work": traction_dual_work,
        "relative_coupling_ratio": abs(coupling_work)
        / max(abs(complementary_energy), 1.0e-30),
    }


def write_optimization_result(
    output: Path,
    pipeline: Any,
    density: Any,
    history: Any,
    summary: dict[str, Any],
) -> None:
    """保存最终密度场 VTU、标量收敛历史与运行摘要 JSON.

    参数:
        output: 输出目标文件夹路径.
        pipeline: 优化管线对象.
        density: 最终单元密度场.
        history: 优化迭代历史记录对象.
        summary: 汇总指标字典.
    """
    output.mkdir(parents=True, exist_ok=True)
    from soptx.visualization.vtk_export import write_vtu
    from fealpy.backend import backend_manager as bm

    # 1. 最终密度场便捷文件
    density_np = np.asarray(bm.to_numpy(density[:]), dtype=np.float64).flatten()
    write_vtu(
        mesh=pipeline.mesh,
        filepath=str(output / "density_final"),
        cell_data={"density": density_np},
    )

    # 2. 完整迭代演化序列 (供 ParaView 作为动画时间序列直接加载)
    if hasattr(history, "physical_densities") and history.physical_densities:
        vtu_dir = output / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)
        for iter_idx, rho_i in enumerate(history.physical_densities, start=1):
            rho_i_np = np.asarray(bm.to_numpy(rho_i), dtype=np.float64).flatten()
            write_vtu(
                mesh=pipeline.mesh,
                filepath=str(vtu_dir / f"density_iter_{iter_idx:03d}"),
                cell_data={"density": rho_i_np},
            )
    history_payload = {
        "iter_indices": history.iter_indices,
        "changes": history.changes,
        "iteration_times": history.iteration_times,
        "scalar_histories": history.scalar_histories,
    }
    (output / "history.json").write_text(
        json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_state_comparison(payload: dict[str, Any]) -> None:
    """在终端格式化打印单次状态分析的关键比较指标与能量诊断.

    参数:
        payload: 状态对比数据字典.
    """
    print()
    print("固定初始密度状态对比")
    print(
        f"  case={payload['case_id']}, model={payload['model']}, "
        f"rho0={payload['initial_density']:.3f}"
    )
    print("  协议: 给定阶次 k, LFEM 采用位移阶 p=k; Hu--Zhang 采用应力阶 k (对应位移阶 k-1); 统一高斯积分阶 q=2k+2.")
    print()
    print("  method   k   q       compliance          volfrac       residual")
    print("  ------- --- --- ------------------ ------------- ----------------")
    for row in payload["rows"]:
        print(
            f"  {row['method']:<7} {row['order']:>3} {row['integration_order']:>3} "
            f"{row['compliance']:>18.8e} {row['volume_fraction']:>13.6f} "
            f"{row['relative_equilibrium_residual']:>16.3e}"
        )
        energy = row["energy_diagnostics"]
        if row["method"] == "lfem":
            print(
                f"           能量: fTu={energy['external_work']:.8e}, "
                f"uKu={energy['internal_energy']:.8e}, "
                f"相对缺陷={energy['relative_defect']:.3e}"
            )
        else:
            print(
                f"           能量: sigmaAsigma={energy['complementary_energy']:.8e}, "
                f"sigmaBu={energy['coupling_work']:.8e}, "
                f"牵引对偶功={energy['traction_dual_work']:.8e}"
            )

    rows_by_order: dict[int, dict[str, dict[str, Any]]] = {}
    for row in payload["rows"]:
        rows_by_order.setdefault(row["order"], {})[row["method"]] = row
    print()
    for order, rows in sorted(rows_by_order.items()):
        lfem = rows.get("lfem")
        huzhang = rows.get("huzhang")
        if lfem is None or huzhang is None:
            continue
        diff = abs(huzhang["compliance"] - lfem["compliance"]) / max(
            abs(lfem["compliance"]), 1.0e-30
        )
        print(f"  k={order}: |C_HZ - C_LFEM| / |C_LFEM| = {diff:.2%}")
    print(
        "  注: 残差小说明各自线性系统已解收敛. 结构合力守恒由 "
        "examples/huzhang_elasticity/concentrated_load_demo.py 承担核查."
    )
