# -*- coding: utf-8 -*-
"""PIML 子结构拓扑优化完整迭代执行脚本 (支持 2D 与 3D 算例, 严格对齐 Huang 2023).

本脚本实现基于 PIML 路线 A (形函数预测 + 式 17) 及精确 FEA 基线的 2D/3D MBB 梁完整拓扑优化闭环。
支持 ParaView 逐代演化动画 VTU 文件序列导出、收敛历程记录及性能评估。
特性：
1. 包含 Huang 2023 式 (330) 同质子结构复用与分块流式缩聚机制，杜绝超大规模下的 OOM 内存溢出；
2. 只加载并校验离线训练产生的带签名 checkpoint；
3. 集成 soptx.topology.filters 官方原生 3D 卷积滤波。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from fealpy.backend import backend_manager as bm

# 将 soptx 源码目录加入路径
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soptx.problems.elasticity import FullMBBBeam2d, FullMBBBeam3d
from soptx.topology.filters import apply_structured_sensitivity_filter
from soptx.topology.optimizers import OCOptimizer
from soptx.ml import ShapeFunctionSurrogateNet
from soptx.ml.substructure import (
    ArtifactCompatibilityError,
    ModelSignature,
    load_checkpoint,
    load_legacy_state_dict,
)
from soptx.ml.substructure.sampling import SAMPLER_VERSION
from soptx.fem.substructure import (
    ExactSchurReduction,
    GlobalAssembler,
    PIMLShapeReduction,
    LinearCornerTraceBasis,
    StreamingShapeFunctionCondensation,
    SubstructureMesh,
    SubstructurePrototype,
    build_substructures,
    project_problem_conditions_to_macro_system,
    solve_interface_system,
)
from soptx.postprocess.vtk_export import write_vtu
from config import CASES_FILE, OUTPUT_DIR, TopOptCase, load
import provenance
from sensitivity_audit import (
    AUDIT_SCHEMA_VERSION,
    audit_accepted_local_responses,
    filtered_sensitivity_metrics,
    select_audit_positions,
    summarize_audit_iterations,
    summarize_local_records,
)


def _shape_function_signature(
    prototype: SubstructurePrototype,
) -> ModelSignature:
    """根据当前子结构原型构造 Route A checkpoint 的物理签名。"""
    n_interior = int(len(prototype.i_dofs))
    n_reduced = int(prototype.deformation_basis.shape[1])
    n_fine = tuple(int(value) for value in prototype.n_fine)
    return ModelSignature(
        n_fine=n_fine,
        input_dim=int(np.prod(n_fine)),
        output_dim=n_interior * n_reduced,
        n_interior_dofs=n_interior,
        n_reduced=n_reduced,
        sampler_version=SAMPLER_VERSION,
    )


def load_route_a_model(
    prototype: SubstructurePrototype,
    checkpoint_path: Path,
    *,
    allow_legacy_weight: bool = False,
) -> ShapeFunctionSurrogateNet:
    """加载并校验 Route A 模型；旧裸权重只能显式启用。"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Route A checkpoint 不存在: {checkpoint_path}. "
            "请先运行 train_route_a.py 生成带签名的模型。"
        )
    signature = _shape_function_signature(prototype)

    def model_factory() -> ShapeFunctionSurrogateNet:
        return ShapeFunctionSurrogateNet(
            signature.input_dim,
            signature.output_dim,
        )

    try:
        model, summary = load_checkpoint(
            checkpoint_path,
            model_factory,
            signature,
        )
    except ArtifactCompatibilityError:
        if not allow_legacy_weight:
            raise
        model = load_legacy_state_dict(checkpoint_path, model_factory())
        print(
            f"[!] 显式加载旧裸权重: {checkpoint_path}; "
            "该文件不含采样与物理签名，仅用于过渡回归。"
        )
        return model
    print(f"[+] 已加载并校验 Route A checkpoint: {checkpoint_path}")
    if summary:
        print(f"    training_summary keys: {sorted(summary)}")
    return model




def run_topopt(
    case: TopOptCase,
    *,
    output_dir: Path = OUTPUT_DIR,
    weight_path: Optional[Path] = None,
    allow_legacy_weight: bool = False,
    save_density_trajectory: bool = False,
    audit_piml_sensitivity: bool = False,
    audit_max_substructures: int = 256,
) -> Dict[str, Any]:
    """执行单个工况的完整拓扑优化流程 (通用 2D/3D, Huang 2023 方案)."""
    if save_density_trajectory and case.solver_mode != "fea_baseline":
        raise ValueError("训练轨迹只能由 fea_baseline 工况生成。")
    print(f"\n=======================================================")
    if audit_piml_sensitivity and case.solver_mode != "piml_route_a":
        raise ValueError("灵敏度 audit 只允许 piml_route_a 工况。")
    if audit_piml_sensitivity and audit_max_substructures <= 0:
        raise ValueError("audit_max_substructures 必须为正整数。")
    print(f" 开始执行拓扑优化工况: [{case.id}] ({case.summary})")
    print(f"=======================================================")

    np.random.seed(case.seed)
    torch.manual_seed(case.seed)

    # 1. 物理问题与有限元分析器
    if case.dim == 2:
        problem = FullMBBBeam2d(domain=case.domain, P=case.p_load, E=case.emax, nu=case.nu)
        domain_size = (case.domain[1] - case.domain[0], case.domain[3] - case.domain[2])
    elif case.dim == 3:
        problem = FullMBBBeam3d(domain=case.domain, P=case.p_load, E=case.emax, nu=case.nu)
        domain_size = (
            case.domain[1] - case.domain[0],
            case.domain[3] - case.domain[2],
            case.domain[5] - case.domain[4],
        )
    else:
        raise ValueError(f"不支持的维数: {case.dim}")

    assembler = GlobalAssembler(
        domain_size, case.n_sub, case.n_fine, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, positions = build_substructures(assembler)
    prototype.rho_min = case.emin
    prototype.penal = case.simp_penalty

    n_elem_grid = tuple(case.n_sub[d] * case.n_fine[d] for d in range(case.dim))
    n_elem_total = int(np.prod(n_elem_grid))
    h_grid = tuple(domain_size[d] / n_elem_grid[d] for d in range(case.dim))
    n_sub_total = len(sub_meshes)

    # 1. 线性插值矩阵 L 与宏观粗网格自由度解析构造 (Huang 2023 式 16)
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)
    L = trace_basis.matrix
    total_macro_dofs = assembler.total_macro_dofs
    c_macro = assembler.macro_corner_indices(sub_meshes)

    # FullMBBBeam2d/3d 是载荷与边界条件的唯一事实源.
    f_macro, fixed_macro_dofs = project_problem_conditions_to_macro_system(
        problem, assembler
    )

    # 2. 预计算标准纯实体子结构缩聚基准 (Huang 2023 Line 330)
    rho_solid_grid = bm.ones((1, *case.n_fine), dtype=bm.float64)
    K_solid_single = prototype.assemble_local_stiffness_batch(
        prototype.grid_to_cell_field(rho_solid_grid)
    )
    exact_reduction = ExactSchurReduction(
        prototype.i_dofs, prototype.b_dofs
    )
    solid_result = exact_reduction.reduce(K_solid_single[0], rho_solid_grid[0])
    Ks_solid = solid_result.stiffness
    N_solid = solid_result.recovery

    # 3. 求解器与代理网络配置
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_weight_path = (
        weight_path or output_dir / f"{case.id}_checkpoint.pt"
    )
    audit_checkpoint_sha256 = None
    if audit_piml_sensitivity:
        audit_checkpoint_sha256 = hashlib.sha256(
            effective_weight_path.read_bytes()
        ).hexdigest()
    if case.solver_mode == "piml_route_a":
        net = load_route_a_model(
            prototype,
            effective_weight_path,
            allow_legacy_weight=allow_legacy_weight,
        )
    elif case.solver_mode == "fea_baseline":
        net = None
    else:
        raise ValueError(f"未知的 solver_mode: {case.solver_mode}")

    piml_reduction = None
    if net is not None:
        piml_reduction = PIMLShapeReduction(
            prototype.i_dofs,
            prototype.b_dofs,
            model=net,
            rigid_basis=prototype.rigid_basis,
            deformation_basis=prototype.deformation_basis,
            rigid_interior=prototype.rigid_interior_modes,
        )

    # 4. 初始化设计变量 (统一使用 bm 数组)
    rho = bm.full((n_elem_total,), case.volfrac, dtype=bm.float64)
    K0 = prototype.KE_unit[0]

    history: List[Dict[str, Any]] = []
    density_trajectory: List[np.ndarray] = []
    t_start_total = time.perf_counter()

    sensitivity_audit_iterations: List[Dict[str, Any]] = []
    vtu_dir = output_dir / f"{case.id}_vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] 进入拓扑优化迭代循环 (Max Iter: {case.max_iter}, Target Vol: {case.volfrac}, Mesh: {n_elem_grid})...")
    print(f"{'Iter':>5} | {'Compliance':>12} | {'VolFrac':>8} | {'MaxChange':>10} | {'SolveTime(ms)':>14} | {'StepTime(ms)':>13}")
    print("-" * 75)

    recent_comp_changes: List[float] = []

    for it in range(1, case.max_iter + 1):
        t_step_start = time.perf_counter()
        piml_fallback_count = 0
        piml_evaluated_count = 0
        piml_fallback_reasons: Counter[str] = Counter()
        piml_gate_rejections: Counter[str] = Counter()

        # (a) 将全场密度在 bm 空间分解为子结构局部密度
        piml_accepted_indices: List[int] = []
        piml_accepted_gate_metrics: List[Dict[str, Any]] = []
        audit_local_records: List[Dict[str, Any]] = []
        audit_hybrid_dc_np: Optional[np.ndarray] = None
        rho_subs_grid = assembler.split_global_cell_field(rho)
        if save_density_trajectory:
            # 保存本轮实际参与 FE 装配的 pre-OC 局部密度，而不是更新后的 rho_new。
            density_trajectory.append(bm.to_numpy(rho_subs_grid).astype(np.float32, copy=True))
        rho_subs_cell = prototype.grid_to_cell_field(rho_subs_grid)

        # (b) 正问题求解: Huang 2023 同质复用与分块流式缩聚 (全 bm 算子)
        t_solve_start = time.perf_counter()

        if case.solver_mode == "piml_route_a":
            rho_max_sub = bm.max(rho_subs_grid, axis=tuple(range(1, case.dim + 1)))
            rho_mean_sub = bm.mean(rho_subs_grid, axis=tuple(range(1, case.dim + 1)))
            is_homo = (rho_max_sub - rho_mean_sub) < 1e-4

            coef_homo = bm.zeros((n_sub_total,), dtype=bm.float64)
            homo_idx = bm.nonzero(is_homo)[0]
            if len(homo_idx) > 0:
                coef_val = prototype.rho_min + (1.0 - prototype.rho_min) * (rho_mean_sub[homo_idx] ** prototype.penal)
                coef_homo = bm.set_at(coef_homo, homo_idx, coef_val)

            N_hetero_dict: Dict[int, Any] = {}
            Ks_hetero_dict: Dict[int, Any] = {}
            hetero_idx = bm.nonzero(~is_homo)[0]
            hetero_idx_list = bm.to_numpy(hetero_idx).tolist() if len(hetero_idx) > 0 else []
            chunk_size = 200

            for c_start in range(0, len(hetero_idx_list), chunk_size):
                c_sel_list = hetero_idx_list[c_start : c_start + chunk_size]
                c_sel = bm.asarray(c_sel_list, dtype=bm.int64)
                rho_chunk_grid = rho_subs_grid[c_sel]
                rho_chunk_cell = rho_subs_cell[c_sel]

                # 分批装配局部刚度矩阵 (保持 bm 张量)
                K_local_chunk = prototype.assemble_local_stiffness_batch(rho_chunk_cell)
                chunk_result = piml_reduction.reduce_many(
                    K_local_chunk, rho_chunk_grid
                )
                N_chunk = chunk_result.recovery
                Ks_chunk = chunk_result.stiffness
                piml_evaluated_count += len(chunk_result)
                for k, diagnostics in enumerate(chunk_result.diagnostics):
                    if not diagnostics.used_fallback:
                        if audit_piml_sensitivity:
                            piml_accepted_indices.append(int(c_sel_list[k]))
                            piml_accepted_gate_metrics.append(
                                dict(diagnostics.metrics)
                            )
                        continue
                    piml_fallback_count += 1
                    reason = diagnostics.fallback_reason or "unspecified"
                    piml_fallback_reasons[reason] += 1
                    failed_gate = diagnostics.metrics.get("failed_gate")
                    if reason == "gate_rejected" and failed_gate is not None:
                        piml_gate_rejections[str(failed_gate)] += 1

                for k, g_idx in enumerate(c_sel_list):
                    N_hetero_dict[g_idx] = N_chunk[k]
                    Ks_hetero_dict[g_idx] = Ks_chunk[k]

            condensors = StreamingShapeFunctionCondensation(
                prototype.i_dofs,
                prototype.b_dofs,
                Ks_batch=None,
                N_homo=N_solid,
                N_hetero_dict=N_hetero_dict,
                is_homo=is_homo,
                Ks_solid=Ks_solid,
                coef_homo=coef_homo,
                Ks_hetero_dict=Ks_hetero_dict,
                n_sub_total=n_sub_total,
            )

        else:
            # 精确 FEA 基线
            K_local_batch = prototype.assemble_local_stiffness_batch(rho_subs_cell)
            exact_result = exact_reduction.reduce_many(
                K_local_batch, rho_subs_grid
            )
            Ks_exact, N_exact = exact_result.stiffness, exact_result.recovery
            # 统一适配后续宏观装配与内部位移恢复接口. 精确基线没有同质旁路,
            # 因此把每个子结构的精确 Schur 补与恢复矩阵作为异质项交给流式容器.
            exact_indices = range(n_sub_total)
            condensors = StreamingShapeFunctionCondensation(
                prototype.i_dofs,
                prototype.b_dofs,
                N_hetero_dict={idx: N_exact[idx] for idx in exact_indices},
                is_homo=bm.zeros((n_sub_total,), dtype=bm.bool),
                Ks_hetero_dict={idx: Ks_exact[idx] for idx in exact_indices},
                n_sub_total=n_sub_total,
            )
        # (c) 宏观降维系统装配与求解 (Huang 2023 式 16)
        Ks_macro = condensors.get_projected_stiffness(trace_basis)
        system_macro = assembler.assemble_macro_system(sub_meshes, Ks_macro)
        u_macro = solve_interface_system(system_macro, f_macro, fixed_macro_dofs)
        t_solve = (time.perf_counter() - t_solve_start) * 1000.0

        # (d) 细尺度位移流式恢复与单元应变能 (基于单单元刚度模板 prototype.KE_unit[0])
        u_c_batch = u_macro[c_macro]
        u_b_batch, u_i_batch = condensors.recover_from_trace(
            u_c_batch,
            trace_basis,
        )

        u_local_batch = bm.zeros((n_sub_total, prototype.n_total_dofs), dtype=bm.float64)
        u_local_batch = bm.set_at(u_local_batch, (slice(None), prototype.b_dofs), u_b_batch)
        u_local_batch = bm.set_at(u_local_batch, (slice(None), prototype.i_dofs), u_i_batch)

        u_elem = u_local_batch[:, prototype.cell2dof]
        energy_sub_cell = bm.sum((u_elem @ K0) * u_elem, axis=-1)
        energy_sub_grid = prototype.cell_to_grid_field(energy_sub_cell)
        energy_global_grid = assembler.merge_substructure_cell_field(energy_sub_grid)
        energy_flat = bm.reshape(energy_global_grid, (-1,))

        compliance = float(bm.dot(f_macro, u_macro))
        dcoef = case.simp_penalty * (rho ** (case.simp_penalty - 1.0))
        if prototype.rho_min != 0.0:
            dcoef = (1.0 - prototype.rho_min) * dcoef
        dc = - dcoef * energy_flat

        # (e) 空间灵敏度卷积滤波与状态暂存
        rho_np = bm.to_numpy(rho)
        if audit_piml_sensitivity and piml_accepted_indices:
            audit_start = time.perf_counter()
            audit_positions = select_audit_positions(
                len(piml_accepted_indices),
                audit_max_substructures,
            )
            audited_substructure_ids = [
                piml_accepted_indices[int(position)]
                for position in audit_positions
            ]
            audited_gate_metrics = [
                piml_accepted_gate_metrics[int(position)]
                for position in audit_positions
            ]
            audit_full_coverage = (
                len(audited_substructure_ids) == len(piml_accepted_indices)
            )
            accepted_index = bm.asarray(
                audited_substructure_ids,
                dtype=bm.int64,
            )
            accepted_stiffness = prototype.assemble_local_stiffness_batch(
                rho_subs_cell[accepted_index]
            )
            accepted_exact = exact_reduction.reduce_many(
                accepted_stiffness,
                rho_subs_grid[accepted_index],
            )
            exact_interior = accepted_exact.recover(
                u_b_batch[accepted_index]
            )
            exact_energy_np, audit_local_records = (
                audit_accepted_local_responses(
                    substructure_ids=audited_substructure_ids,
                    boundary_displacement=bm.to_numpy(u_b_batch[accepted_index]),
                    predicted_interior_displacement=bm.to_numpy(
                        u_i_batch[accepted_index]
                    ),
                    exact_interior_displacement=bm.to_numpy(exact_interior),
                    density_cell=bm.to_numpy(rho_subs_cell[accepted_index]),
                    interior_dofs=bm.to_numpy(prototype.i_dofs),
                    boundary_dofs=bm.to_numpy(prototype.b_dofs),
                    cell_to_dof=bm.to_numpy(prototype.cell2dof),
                    unit_cell_stiffness=bm.to_numpy(K0),
                    simp_penalty=case.simp_penalty,
                    rho_min=prototype.rho_min,
                    gate_metrics=audited_gate_metrics,
                )
            )
            if audit_full_coverage:
                hybrid_energy_sub_cell = bm.copy(energy_sub_cell)
                hybrid_energy_sub_cell = bm.set_at(
                    hybrid_energy_sub_cell,
                    accepted_index,
                    bm.asarray(exact_energy_np, dtype=bm.float64),
                )
                hybrid_energy_grid = prototype.cell_to_grid_field(
                    hybrid_energy_sub_cell
                )
                hybrid_energy_global = assembler.merge_substructure_cell_field(
                    hybrid_energy_grid
                )
                audit_hybrid_dc_np = bm.to_numpy(
                    -dcoef * bm.reshape(hybrid_energy_global, (-1,))
                )
            audit_time_ms = (time.perf_counter() - audit_start) * 1000.0
        else:
            audit_full_coverage = True
            audit_time_ms = 0.0
        dc_np = bm.to_numpy(dc)

        # 保存关键迭代步的 VTU 场数据 (小规模每步写, 165 万大规模每 10 步及末步写)
        audit_dc_filtered_np = None
        is_save_step = (n_elem_total <= 100000) or (it % 10 == 0) or (it == case.max_iter)
        if is_save_step:
            iter_vtu_stem = vtu_dir / f"iter_{it:03d}"
            write_vtu(
                assembler.full_mesh,
                cell_data={"density": rho_np},
                filepath=str(iter_vtu_stem),
            )
        if case.dim == 2:
            dc_grid = dc_np.reshape((*n_elem_grid, 1))
            rho_grid_f = rho_np.reshape((*n_elem_grid, 1))
            spacing = (*h_grid, max(domain_size) * 10.0)
            dc_filtered_grid = apply_structured_sensitivity_filter(
                sensitivity=dc_grid,
                density=rho_grid_f,
                rmin=case.filter_radius,
                spacing=spacing,
                kind="cone",
            )
            dc_filtered_np = dc_filtered_grid.flatten()
            if audit_hybrid_dc_np is not None:
                audit_dc_grid = audit_hybrid_dc_np.reshape((*n_elem_grid, 1))
                audit_dc_filtered_grid = apply_structured_sensitivity_filter(
                    sensitivity=audit_dc_grid,
                    density=rho_grid_f,
                    rmin=case.filter_radius,
                    spacing=spacing,
                    kind="cone",
                )
                audit_dc_filtered_np = audit_dc_filtered_grid.flatten()
        elif case.dim == 3:
            dc_grid = dc_np.reshape(n_elem_grid)
            rho_grid_f = rho_np.reshape(n_elem_grid)
            dc_filtered_grid = apply_structured_sensitivity_filter(
                sensitivity=dc_grid,
                density=rho_grid_f,
                rmin=case.filter_radius,
                spacing=h_grid,
                kind="cone",
            )
            dc_filtered_np = dc_filtered_grid.flatten()
            if audit_hybrid_dc_np is not None:
                audit_dc_grid = audit_hybrid_dc_np.reshape(n_elem_grid)
                audit_dc_filtered_grid = apply_structured_sensitivity_filter(
                    sensitivity=audit_dc_grid,
                    density=rho_grid_f,
                    rmin=case.filter_radius,
                    spacing=h_grid,
                    kind="cone",
                )
                audit_dc_filtered_np = audit_dc_filtered_grid.flatten()

        if audit_piml_sensitivity:
            accepted_count = len(piml_accepted_indices)
            homogeneous_bypass_count = n_sub_total - piml_evaluated_count
            if accepted_count + piml_fallback_count != piml_evaluated_count:
                raise RuntimeError("audit 计数不满足 accepted + fallback = evaluated。")
            if piml_evaluated_count + homogeneous_bypass_count != n_sub_total:
                raise RuntimeError("audit 计数不满足 evaluated + homogeneous = total。")
            sensitivity_audit_iterations.append(
                {
                    "iter": it,
                    "evaluated_count": int(piml_evaluated_count),
                    "accepted_count": accepted_count,
                    "audited_count": len(audit_local_records),
                    "full_accepted_coverage": audit_full_coverage,
                    "fallback_count": int(piml_fallback_count),
                    "homogeneous_bypass_count": int(homogeneous_bypass_count),
                    "audit_time_ms": float(audit_time_ms),
                    "local_summary": summarize_local_records(
                        audit_local_records
                    ),
                    "filtered_sensitivity": (
                        None
                        if audit_dc_filtered_np is None
                        else filtered_sensitivity_metrics(
                            dc_filtered_np,
                            audit_dc_filtered_np,
                        )
                    ),
                    "substructures": audit_local_records,
                }
            )

        dc_filtered = bm.asarray(dc_filtered_np, dtype=bm.float64)

        # (e) 直接复用 soptx 的 OC optimizer 完成二分搜索与设计变量更新
        rho_new = OCOptimizer.update_design_variable(
            design_variable=rho,
            objective_gradient=dc_filtered,
            constraint_gradient=bm.ones(rho.shape, dtype=bm.float64),
            constraint_function=lambda candidate: bm.mean(candidate) - case.volfrac,
            move_limit=0.2,
            damping_coef=0.5,
            initial_lambda=1.0e9,
            bisection_tol=1.0e-4,
            design_variable_min=1.0e-3,
        )

        change = float(bm.max(bm.abs(rho_new - rho)))
        vol_curr = float(bm.mean(rho_new))
        t_step = (time.perf_counter() - t_step_start) * 1000.0

        history.append({
            "iter": it,
            "compliance": float(compliance),
            "volfrac": float(vol_curr),
            "max_change": float(change),
            "solve_time_ms": float(t_solve),
            "step_time_ms": float(t_step),
            "piml_sensitivity_audit_time_ms": float(audit_time_ms),
            "piml_evaluated_count": int(piml_evaluated_count),
            "piml_fallback_count": int(piml_fallback_count),
            "piml_fallback_reasons": dict(piml_fallback_reasons),
            "piml_gate_rejections": dict(piml_gate_rejections),
        })

        if len(history) >= 2:
            c_prev = history[-2]["compliance"]
            rel_c_change = abs(compliance - c_prev) / c_prev
            recent_comp_changes.append(rel_c_change)
            if len(recent_comp_changes) > 5:
                recent_comp_changes.pop(0)

        if it % 5 == 0 or it == 1 or change < case.tol_change or it == case.max_iter:
            print(f"{it:5d} | {compliance:12.4f} | {vol_curr:8.4f} | {change:10.5f} | {t_solve:14.2f} | {t_step:13.2f}")

        rho = bm.copy(rho_new)

        # 检查 Huang 2023 连续 5 步收敛判定
        if len(recent_comp_changes) == 5 and all(rc < case.tol_change for rc in recent_comp_changes) and it >= 10:
            print(f"[+] 满足 Huang 2023 论文收敛判据 (连续 5 步 |dC|/C < {case.tol_change}), 优化完成!")
            break

    t_total = time.perf_counter() - t_start_total
    print("-" * 75)
    print(f"[+] 优化完成! 总耗时: {t_total:.2f} s, 最终迭代步: {len(history)}, 最终柔度: {history[-1]['compliance']:.4f}")

    # 二维保存云图 (IO 边界显式转为 NumPy)
    rho_final_np = bm.to_numpy(rho)
    if case.dim == 2:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{case.id}_topology.png"
        rho_plot = rho_final_np.reshape(n_elem_grid).T  # 转置以符合 2D (y, x) 图像显示
        plt.figure(figsize=(10, 2.5), dpi=300)
        plt.imshow(1.0 - rho_plot, cmap="gray", origin="lower", extent=[case.domain[0], case.domain[1], case.domain[2], case.domain[3]])
        plt.title(f"Optimized Topology: {case.id} (Iter: {len(history)}, C: {history[-1]['compliance']:.4f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"[+] 拓扑云图已保存至: {fig_path}")

    # 保存最终 VTU 三维/二维非结构网格文件 (支持 ParaView 可视化)
    vtu_stem = output_dir / f"{case.id}_final"
    write_vtu(
        assembler.full_mesh,
        cell_data={"density": rho_final_np},
        filepath=str(vtu_stem),
    )
    print(f"[+] 最终 VTU 文件已保存至: {vtu_stem}.vtu")
    print(f"[+] 完整迭代步 VTU 序列已保存至: {vtu_dir}/")

    # 保存轨迹与数据
    npz_path = output_dir / f"{case.id}_trajectory.npz"
    trajectory_payload: Dict[str, Any] = {
        "schema_version": np.asarray("substructure-density-trajectory-v2"),
        "case_id": np.asarray(case.id),
        "solver_mode": np.asarray(case.solver_mode),
        "trajectory_id": np.asarray(
            f"{case.id}:vf{case.volfrac:g}:seed{case.seed}"
        ),
        "density_role": np.asarray(
            "training_candidate" if save_density_trajectory else "result_only"
        ),
        "volfrac": np.asarray(case.volfrac, dtype=np.float64),
        "n_sub": np.asarray(case.n_sub, dtype=np.int64),
        "n_fine": np.asarray(case.n_fine, dtype=np.int64),
        "filter_radius": np.asarray(case.filter_radius, dtype=np.float64),
        "filter_type": np.asarray(case.filter_type),
        "simp_penalty": np.asarray(case.simp_penalty, dtype=np.float64),
        "design_variable_min": np.asarray(1.0e-3, dtype=np.float64),
        "iter_indices": np.asarray([h["iter"] for h in history], dtype=np.int64),
        "final_density": rho_final_np,
        "history_compliance": np.asarray([h["compliance"] for h in history]),
        "history_change": np.asarray([h["max_change"] for h in history]),
        "history_solve_time": np.asarray([h["solve_time_ms"] for h in history]),
    }
    if save_density_trajectory:
        trajectory_payload["rho_sub"] = np.stack(density_trajectory, axis=0)
    np.savez_compressed(npz_path, **trajectory_payload)

    sensitivity_audit_meta = None
    if audit_piml_sensitivity:
        audit_payload = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "diagnostic_only": True,
            "case_id": case.id,
            "checkpoint": {
                "path": str(effective_weight_path),
                "sha256": audit_checkpoint_sha256,
            },
            "case_config": {
                "n_sub": list(case.n_sub),
                "n_fine": list(case.n_fine),
                "volfrac": case.volfrac,
                "simp_penalty": case.simp_penalty,
                "rho_min": prototype.rho_min,
                "filter_type": case.filter_type,
                "filter_radius": case.filter_radius,
            },
            "sampling": {
                "policy": "deterministic_evenly_spaced_accepted_indices",
                "max_substructures_per_iteration": audit_max_substructures,
                "filtered_reference_requires_full_accepted_coverage": True,
            },
            "comparison_scope": (
                "Only gate-accepted heterogeneous PIML substructures are "
                "recovered by Exact Schur under the same PIML global boundary "
                "trace. The global interface problem is not re-solved."
            ),
            "does_not_modify_optimization": True,
            "summary": summarize_audit_iterations(
                sensitivity_audit_iterations
            ),
            "iterations": sensitivity_audit_iterations,
        }
        audit_path = output_dir / f"{case.id}_sensitivity_audit.json"
        audit_path.write_text(
            json.dumps(audit_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sensitivity_audit_meta = {
            "path": str(audit_path),
            "summary": audit_payload["summary"],
        }
        print(f"[+] PIML 灵敏度诊断报告已保存至: {audit_path}")

    total_evaluated = sum(h["piml_evaluated_count"] for h in history)
    total_fallback = sum(h["piml_fallback_count"] for h in history)
    total_fallback_reasons: Counter[str] = Counter()
    total_gate_rejections: Counter[str] = Counter()
    for item in history:
        total_fallback_reasons.update(item["piml_fallback_reasons"])
        total_gate_rejections.update(item["piml_gate_rejections"])
    piml_diagnostics = {
        "evaluated_count": int(total_evaluated),
        "fallback_count": int(total_fallback),
        "fallback_rate": float(total_fallback / total_evaluated) if total_evaluated else 0.0,
        "fallback_reasons": dict(total_fallback_reasons),
        "gate_rejections": dict(total_gate_rejections),
    }

    result_meta = {
        "case_id": case.id,
        "dim": case.dim,
        "solver_mode": case.solver_mode,
        "summary": case.summary,
        "total_time_s": t_total,
        "final_compliance": history[-1]["compliance"],
        "final_volfrac": history[-1]["volfrac"],
        "converged_iters": len(history),
        "mean_solve_time_ms": float(np.mean([h["solve_time_ms"] for h in history])),
        "mean_step_time_ms": float(np.mean([h["step_time_ms"] for h in history])),
        "provenance": provenance.capture(),
        "piml_diagnostics": piml_diagnostics,
        "sensitivity_audit": sensitivity_audit_meta,
        "history": history,
    }
    json_path = output_dir / f"{case.id}_result.json"
    json_path.write_text(json.dumps(result_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] 详细指标 JSON 已落盘至: {json_path}")
    return result_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="PIML 子结构拓扑优化执行器 (2D/3D)")
    parser.add_argument("--case", type=str, default="all", help="执行指定工况 ID 或 'all'")
    parser.add_argument("--max-iter", type=int, default=None, help="覆盖最大迭代步数")
    parser.add_argument(
        "--audit-piml-sensitivity",
        action="store_true",
        help="诊断 accepted PIML 局部恢复的能量/灵敏度误差；不改变优化更新",
    )
    parser.add_argument(
        "--audit-max-substructures",
        type=int,
        default=256,
        help="每轮最多审计的 accepted 子结构数；采用确定性均匀位置采样",
    )
    parser.add_argument("--volfrac", type=float, default=None, help="覆盖目标体积分数")
    parser.add_argument(
        "--save-density-trajectory",
        action="store_true",
        help="显式保存每轮求解前的局部密度；仅允许 fea_baseline 工况",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="结果输出目录；可用于隔离 smoke run，默认沿用实验 outputs 目录",
    )
    parser.add_argument(
        "--weight-path", type=Path, default=None,
        help="指定单个 PIML 工况的已有权重；省略时从 output-dir 按工况名读取",
    )
    parser.add_argument(
        "--allow-legacy-weight",
        action="store_true",
        help="显式允许加载无签名的旧裸 state_dict；仅用于过渡回归",
    )
    args = parser.parse_args()
    if args.volfrac is not None and not 0.0 < args.volfrac <= 1.0:
        parser.error("--volfrac 必须位于 (0, 1]")
    if args.save_density_trajectory and args.case == "all":
        parser.error("--save-density-trajectory 必须与单个 --case 同时使用")
    if args.audit_piml_sensitivity and args.case == "all":
        parser.error("--audit-piml-sensitivity 必须与单个 --case 同时使用")
    if args.audit_max_substructures <= 0:
        parser.error("--audit-max-substructures 必须为正整数")

    meta, cases = load()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_list = [c for c in cases if args.case == "all" or c.id == args.case]
    if not run_list:
        print(f"[!] 未找到匹配的工况: {args.case}")
        sys.exit(1)
    if args.weight_path is not None and len(run_list) != 1:
        parser.error("--weight-path 只能与单个 --case 工况同时使用")

    all_results = []
    for c in run_list:
        overrides = {}
        if args.max_iter is not None:
            overrides["max_iter"] = args.max_iter
        if args.volfrac is not None:
            overrides["volfrac"] = args.volfrac
        if overrides:
            c = TopOptCase(**{**c.__dict__, **overrides})
        res = run_topopt(
            c,
            output_dir=output_dir,
            weight_path=args.weight_path,
            allow_legacy_weight=args.allow_legacy_weight,
            save_density_trajectory=args.save_density_trajectory,
            audit_piml_sensitivity=args.audit_piml_sensitivity,
            audit_max_substructures=args.audit_max_substructures,
        )
        all_results.append(res)

    print(f"\n[+] 全部工况执行完毕. 产物存放于 {output_dir}")


if __name__ == "__main__":
    main()
