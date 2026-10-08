"""二维悬臂梁局部应力约束体积最小化 (论文 5.2.3 节 / 图 5.8~5.11).

设计域 80 mm x 40 mm, 平面应力, 左端固支, 右端中点竖直合力 P 化为宽 LOAD_WIDTH 的
均布牵引. 以表观应力松弛约束 (式 (4.7)) 下的体积最小化为目标, 增广拉格朗日 + MMA
(AL-MMA) 求解; LFEM p = 2, 3, 4 与 HZMFEM k = 2, 3, 4 各做一次优化. 牵引贴片两端点的
应力奇异性由载荷侧垫片处置: 半径 LOAD_PAD_RADIUS 内的单元既豁免应力约束又钉为实体.

    python run_cantilever_stress.py                                # 六组
    python run_cantilever_stress.py --analyzer huzhang --order 2   # 只跑其中几组
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.fem import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import create_huzhang_checkerboard_mesh
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import CantileverMiddle2d
from soptx.topology.constraints import (
    EpsilonRelaxedStressFormulation,
    HuZhangStressConstraint,
    LagrangeStressConstraint,
    build_exemption_mask,
)
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import AugmentedLagrangianObjective, VolumeObjective
from soptx.topology.optimizers import ALMMMAOptimizer, ALMMMAOptions

CASE_ID = "cantilever-middle-2d-stress"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / CASE_ID
# ParaView 查看副本 (Windows 本地盘; 经 \\wsl.localhost 读大批帧很慢): 逐步帧 vtu/ 与
# evolution.pvd 写到这里 (最终密度只在 results/), 目录结构与 results/ 一一对应; 该盘不可用时退回 results/
VIEW_ROOT = Path("/mnt/c/workspace/soptx-results/paper_topopt_huzhang")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 模型参数. 杨氏模量取 1: 带跳量稳定化的 HZMFEM 柔度块 O(1/E) 与惩罚块 O(E) 使
# 条件数达 O(E^2), 真实值 70000 MPa 下 MUMPS 报 -9; 纯牵引 + 齐次 Dirichlet 下应力
# 与 E 无关, 应力约束不受影响
LOAD = -400.0                # N, 右端中点竖直合力
LOAD_WIDTH = 6.0             # mm, 化为均布牵引的接触宽度
E0 = 1.0                     # MPa
NU = 0.25
PLANE_TYPE = "plane_stress"
STRESS_LIMIT = 180.0         # MPa, 许用应力
EPSILON = 1.0e-3             # 式 (4.7) 松弛参数; 与过渡区单元 m_E 同量级才起松弛作用
# 载荷侧垫片半径 (mm), 取 l / 4: 贴片端点的牵引间断使应力随阶次单调升而不收敛, 其
# 邻域既豁免应力约束又钉为实体 (两者成对施加, 只豁免会被优化器减料换体积)
LOAD_PAD_RADIUS = 1.5
SUPPORT_PAD_RADIUS = 0.0     # 固支角点垫片半径; 0 表示不处置

# 离散: 棋盘格交替对角三角网格 (nx, ny 须为偶数), 四个角点满足角点松弛的拓扑要求
NX, NY = 80, 40
METHODS = ("lfem", "huzhang")
ORDERS = (2, 3, 4)
USE_RELAXATION = True        # HZMFEM 角点松弛
ASSEMBLY_METHOD = "standard"  # LFEM 单元刚度装配方法
SOLVER = "mumps"

# 优化: MSIMP 只插值 Young 模量; 密度过滤 + tanh 投影 (beta 1 -> 10, 每 5 个外层步 +1)
FILTER_RADIUS = 6.0          # mm
PENALTY = 3.5
EMIN = 1.0e-9
INITIAL_DENSITY = 0.5        # 满密度启动会导致 ALM 发散
PROJECTION = {"continuation_strategy": "additive", "projection_type": "tanh", "beta": 1.0,
              "beta_max": 10.0, "continuation_iter": 5, "beta_increment": 1.0}
# AL-MMA (论文第 4.6 节); 停止准则 C0 连续化终止 / C1 设计稳定 / C2 松弛可行 /
# C3 连续 HOLD_STEPS 个外层步同时满足
MAX_AL_ITERATIONS = 200
MMA_ITERS_PER_AL = 5
CHANGE_TOLERANCE = 2.0e-3
CHANGE_MEASURE = "mean"      # C1: 外层步首末设计变量的平均绝对变化 (PolyStress 口径)
STRESS_TOLERANCE = 5.0e-3    # C2: delta_g
HOLD_STEPS = 3
INNER_STOP_RULE = "legacy"
INNER_RELATIVE_TOLERANCE = 0.1
INNER_ABSOLUTE_TOLERANCE = 1.0e-6
MU_0 = 50.0                  # 出图旧程序实取 50 (论文 4.6 节正文写 10, 应以 50 为准)
MU_MAX = 1.0e4
ALPHA = 1.1
LAMBDA_0 = 0.0
LAMBDA_MAX = 3000.0          # 乘子安全阈 (safeguarded AL)
MOVE_LIMIT = 0.15
ASYMPTOTE_MIN_DISTANCE = 1.0e-4
ACCEPTANCE_SOLID_THRESHOLD = 0.5  # C2 只在 rho_phys >= 0.5 的未豁免单元上验收


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析要跑的分析链与阶次; 缺省为全部六组."""
    parser = argparse.ArgumentParser(description="二维悬臂梁局部应力约束体积最小化 (论文 5.2.3 节)")
    parser.add_argument("--analyzer", choices=METHODS, action="append",
                        help="只跑指定分析链 (可重复); 缺省为两条链")
    parser.add_argument("--order", type=int, choices=ORDERS, action="append",
                        help="只跑指定阶次 (可重复); 缺省为 2, 3, 4")
    return parser.parse_args(argv)


def run_label(method: str, order: int) -> str:
    """运行目录名: 约束协议、垫片半径与 C2 验收子集恒进目录名 (沿用 driver 的标签规则)."""
    return (f"analyzer-{method}__lfem_constraint-apparent__load_pad_radius-{LOAD_PAD_RADIUS}"
            f"__order-{order}__solid_thr-{ACCEPTANCE_SOLID_THRESHOLD}")


def al_options() -> ALMMMAOptions:
    """AL-MMA 选项; 增广拉格朗日目标与优化器共用同一组."""
    return ALMMMAOptions(
        change_tolerance=CHANGE_TOLERANCE, stress_tolerance=STRESS_TOLERANCE,
        hold_steps=HOLD_STEPS, inner_stop_rule=INNER_STOP_RULE,
        inner_relative_tolerance=INNER_RELATIVE_TOLERANCE,
        inner_absolute_tolerance=INNER_ABSOLUTE_TOLERANCE,
        max_al_iterations=MAX_AL_ITERATIONS, mma_iters_per_al=MMA_ITERS_PER_AL,
        mu_0=MU_0, mu_max=MU_MAX, alpha=ALPHA, lambda_0_init_val=LAMBDA_0,
        move_limit=MOVE_LIMIT, asymptote_min_distance=ASYMPTOTE_MIN_DISTANCE,
        change_measure=CHANGE_MEASURE, lambda_max=LAMBDA_MAX,
        acceptance_solid_threshold=ACCEPTANCE_SOLID_THRESHOLD,
    )


def build(method: str, order: int, load_pad_radius: float = LOAD_PAD_RADIUS) -> dict[str, Any]:
    """按受控比较协议组装一条分析链: 问题、网格、分析器、垫片、应力约束与 AL 目标.

    Parameters
    ----------
    method : {'lfem', 'huzhang'}
        分析链.
    order : int
        阶次 k (LFEM 位移阶 p = k, HZMFEM 应力阶 k).
    load_pad_radius : float, optional
        载荷侧垫片半径; 冻结构型探针取 0, 在不豁免的约束上取回被动实体区的真实读数.

    Returns
    -------
    dict[str, Any]
        组装好的对象; 冻结构型再分析只用其中的分析链与应力约束, 不挂优化器.
    """
    problem = CantileverMiddle2d(P=LOAD, load_width=LOAD_WIDTH, E=E0, nu=NU, plane_type=PLANE_TYPE)
    xmin, xmax, ymin, ymax = problem.domain
    mesh: Any = create_huzhang_checkerboard_mesh(box=problem.domain, nx=NX, ny=NY)
    # 过滤矩阵等依赖网格上的 meshdata 元数据
    mesh.meshdata = {"domain": list(problem.domain), "mesh_type": "triangle-checkerboard",
                     "nx": NX, "ny": NY, "hx": (xmax - xmin) / NX, "hy": (ymax - ymin) / NY}

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E, poisson_ratio=problem.nu,
        hypothesis=problem.plane_type, enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp",
        options={"penalty_factor": PENALTY, "void_youngs_modulus": EMIN, "target_variables": ["E"]},
        enable_logging=False,
    )
    common = dict(disp_mesh=mesh, pde=problem, material=material, space_degree=order,
                  integration_order=2 * order + 2, solve_method=SOLVER,
                  topopt_algorithm="density_based", interpolation_scheme=interpolation)
    if method == "lfem":
        analyzer = LagrangeFEMAnalyzer(assembly_method=ASSEMBLY_METHOD, **common)
        constraint_type = LagrangeStressConstraint
    else:
        analyzer = HuZhangMFEMAnalyzer(use_relaxation=USE_RELAXATION, **common)
        constraint_type = HuZhangStressConstraint

    # 两处几何应力奇点按各自的固定物理半径处置, 两条分析链用同一组掩码: 剔除其应力
    # 评价点, 并经 problem 与过滤器把这些单元钉为实体
    load_pad_mask = build_exemption_mask(
        mesh=mesh, centers=problem.traction_patch_endpoints, radius=load_pad_radius)
    support_pad_mask = build_exemption_mask(
        mesh=mesh, centers=problem.clamped_corner_points, radius=SUPPORT_PAD_RADIUS)
    pad_mask = bm.logical_or(load_pad_mask, support_pad_mask)
    problem.set_passive_element_mask(pad_mask)
    stress_constraint = constraint_type(
        analyzer=analyzer, stress_limit=STRESS_LIMIT,
        formulation=EpsilonRelaxedStressFormulation(epsilon=EPSILON),
        exemption_mask=pad_mask, enable_logging=False)

    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh, displacement_mesh=mesh, relative_density=INITIAL_DENSITY)
    volume_objective = VolumeObjective(analyzer=analyzer, enable_logging=False)
    return {
        "method": method, "order": order, "problem": problem, "mesh": mesh, "analyzer": analyzer,
        "design_variable": design_variable, "density": density,
        "volume_objective": volume_objective, "stress_constraint": stress_constraint,
        "al_objective": AugmentedLagrangianObjective(
            volume_objective=volume_objective, stress_constraint=stress_constraint,
            options=al_options(), enable_logging=False),
        "load_pad_mask": load_pad_mask, "support_pad_mask": support_pad_mask, "pad_mask": pad_mask,
    }


def main(argv: list[str] | None = None) -> int:
    """逐组: 组装 -> 投影过滤 + AL-MMA 优化 -> 终态求解与验收 -> 落盘."""
    args = parse_args(argv)
    bm.set_backend("numpy")

    # 溯源戳记在开始时盖: 记录的是本进程实际加载的代码, 不受运行期间仓库变动影响
    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments], capture_output=True, text=True)
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain")
    dirty = None if status is None else bool(status)
    stamp = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "reproducible": dirty is False,
    }
    to_np = lambda value: np.asarray(bm.to_numpy(value), dtype=np.float64)
    # 积分点场 (NC, NQ...) 压成单元最大值 (NC,), 与 max_apparent_stress_ratio 同口径
    per_cell_max = lambda value: to_np(value).reshape(to_np(value).shape[0], -1).max(axis=1)

    for method in args.analyzer or METHODS:
        for order in args.order or ORDERS:
            label = run_label(method, order)
            output = OUTPUT_DIR / label
            view = VIEW_ROOT / CASE_ID / label if VIEW_ROOT.parent.is_dir() else output
            print(f"\n[run] {CASE_ID}: analyzer={method}, order={order} -> {output} (帧: {view})",
                  flush=True)
            parts = build(method, order)
            analyzer, problem, constraint = parts["analyzer"], parts["problem"], parts["stress_constraint"]

            # 1. 投影过滤 (实体保留施加在过滤/投影之后) + AL-MMA 优化
            density_filter = Filter(design_mesh=parts["mesh"], filter_type="projection",
                                    rmin=FILTER_RADIUS, density_location="element", filter_q=3,
                                    projection_params=dict(PROJECTION), passive_mask=parts["pad_mask"],
                                    enable_logging=False)
            optimizer = ALMMMAOptimizer(al_objective=parts["al_objective"], filter=density_filter,
                                        options=al_options(), enable_logging=True)
            density, history = optimizer.optimize(
                design_variable=parts["design_variable"], density_distribution=parts["density"])

            # 2. 在最终物理密度上重新求解并验收; 优化器判据基于迭代末态, 这里再核一次 C2
            state = analyzer.solve_state(rho_val=density)
            if method == "huzhang":
                residual = float(analyzer.relative_state_residual())
            else:
                # LFEM 只计非 Dirichlet 自由度
                r = analyzer.stiffness_matrix.matmul(state["displacement"][:]) - analyzer.force_vector
                free = ~analyzer.tensor_space.is_boundary_dof(
                    threshold=problem.is_dirichlet_boundary(), method="interp")
                residual = float(bm.linalg.norm(r[free])) / max(
                    float(bm.linalg.norm(analyzer.force_vector[free])), 1.0e-30)
            constraint_values = constraint.fun(density, state)
            stress_measure = constraint.compute_stress_measure(density, state)
            relative_violation = constraint.compute_relative_violation(density, state)
            # 未加权实体应力比: 只报告不判据, 反映实体材料真实的应力水平
            solid_stress_ratio = to_np(constraint.compute_solid_stress_ratio(density, state))
            rho = to_np(density[:]).reshape(-1)
            solid = rho >= ACCEPTANCE_SOLID_THRESHOLD
            violation_cell = to_np(relative_violation).reshape(rho.shape[0], -1).max(axis=1)
            g_solid = float(violation_cell[solid].max()) if solid.any() else None
            feasible = bool((g_solid if g_solid is not None else float(to_np(relative_violation).max()))
                            <= STRESS_TOLERANCE)
            # 垫片内的约束值按未豁免口径重算 (约束对象在豁免单元上返回哨兵 -1)
            pad, load_pad, support_pad = (to_np(parts[k]).astype(bool)
                                          for k in ("pad_mask", "load_pad_mask", "support_pad_mask"))
            pad_constraint = to_np(constraint.compute_unexempted_constraint(density, state))
            masked_max = lambda values, mask: float(values[mask].max()) if mask.any() else None
            al = parts["al_objective"]
            multiplier = to_np(al.lamb)
            change = optimizer.last_multiplier_change
            outer = {key.replace("last_", ""): getattr(optimizer, key, None)
                     for key in ("last_change_outer_mean", "last_change_outer_max")}

            summary = {
                "case_id": CASE_ID,
                "model": "CantileverMiddle2d",
                "method": method,
                "order": order,
                "mesh_type": "triangle-checkerboard",
                "nx": NX,
                "ny": NY,
                "optimizer": "al_mma",
                "load_discretization": "patch",
                "integration_order": 2 * order + 2,
                "stabilization_coefficient": getattr(analyzer, "stabilization_coefficient", None),
                "volume_fraction": float(parts["volume_objective"].fun(density)),
                "relative_equilibrium_residual": residual,
                "optimization_iterations": len(history.iter_indices),
                "converged": bool(optimizer.converged and feasible),
                "termination_reason": optimizer.termination_reason,
                "solver": SOLVER,
                "interpolation": {"variables": "E"},
                "provenance": stamp,
                "stress_constraint_type": type(constraint).__name__,
                "lfem_stress_constraint_formulation": "apparent",
                "stress_constraint_formulation": "apparent",
                # 表观应力比用于展示, 约束对象定义的相对超限量 g 用于验收
                "max_constraint": float(to_np(constraint_values).max()),
                "max_von_mises": float(to_np(stress_measure).max()),
                "max_apparent_stress_ratio": float(to_np(stress_measure).max()),
                "max_relative_violation": float(to_np(relative_violation).max()),
                "relative_stress_tolerance": STRESS_TOLERANCE,
                "acceptance_solid_threshold": ACCEPTANCE_SOLID_THRESHOLD,
                "max_relative_violation_solid_region": g_solid,
                "relative_stress_feasible": feasible,
                "stress_feasible": feasible,
                "max_solid_stress_ratio": float(solid_stress_ratio.max()),
                "solid_region_threshold": ACCEPTANCE_SOLID_THRESHOLD,
                "max_solid_stress_ratio_solid_region": masked_max(solid_stress_ratio, solid),
                # 垫片诊断: 两侧分列, 使"垫片掩盖了多大的应力"可直接读出
                "load_pad_radius": LOAD_PAD_RADIUS,
                "support_pad_radius": SUPPORT_PAD_RADIUS,
                "load_pad_centers": [[float(v) for v in c] for c in problem.traction_patch_endpoints] or None,
                "support_pad_centers": [[float(v) for v in c] for c in problem.clamped_corner_points] or None,
                "pad_cells": int(pad.sum()),
                "load_pad_cells": int(load_pad.sum()),
                "support_pad_cells": int(support_pad.sum()),
                "max_solid_stress_ratio_constrained": masked_max(solid_stress_ratio, ~pad),
                "max_solid_stress_ratio_pad": masked_max(solid_stress_ratio, load_pad),
                "max_constraint_pad": masked_max(pad_constraint, load_pad),
                "max_solid_stress_ratio_support_pad": masked_max(solid_stress_ratio, support_pad),
                "max_constraint_support_pad": masked_max(pad_constraint, support_pad),
                # ALM 内部状态: mu 为全局标量, 乘子相对变化只有优化器能算
                "penalty_parameter": float(al.mu),
                "max_multiplier": float(multiplier.max()),
                "complementarity_residual": float(np.abs(multiplier * to_np(constraint_values)).max()),
                "multiplier_relative_change": (None if change is None or not np.isfinite(change)
                                               else float(change)),
                "change_measure": CHANGE_MEASURE,
                **{key: (None if value is None or not np.isfinite(value) else float(value))
                   for key, value in outer.items()},
                "lambda_max": LAMBDA_MAX,
                "multiplier_capped_count": int(getattr(al, "last_capped_count", 0)),
                "move_limit_base": MOVE_LIMIT,
                "asymptote_min_distance": ASYMPTOTE_MIN_DISTANCE,
            }

            # 3. AL 终态 (供冻结复算与内层诊断, 不支持精确断点续算)
            design = optimizer.final_design_variable
            state_arrays = {"design": to_np(design[:]), "density": rho,
                            "lamb": multiplier, "mu": np.asarray(float(al.mu))}
            if any(not np.all(np.isfinite(value)) for value in state_arrays.values()):
                raise ValueError("AL 终态包含非有限值, 不能作为诊断状态保存.")
            options = optimizer.options
            summary["optimizer_state"] = {
                "file": "final_optimizer_state.npz",
                "purpose": "frozen-analysis-and-inner-diagnostics",
                "exact_restart_supported": False,
                "beta": getattr(density_filter, "beta", None),
                "inner_stop_rule": options.inner_stop_rule,
                "inner_relative_tolerance": options.inner_relative_tolerance,
                "inner_absolute_tolerance": options.inner_absolute_tolerance,
                "max_inner_iterations": options.mma_iters_per_al,
                "last_inner_diagnostics": getattr(optimizer, "last_inner_diagnostics", None),
            }

            # 4. 落盘: 摘要、收敛历史、最终密度 (带终态应力比) 与 AL 终态写 output (入库);
            #    逐步帧 vtu/ (带当步应力比) 与 evolution.pvd 写 view (ParaView 查看副本, 不入库)
            mesh = parts["mesh"]
            output.mkdir(parents=True, exist_ok=True)
            write_vtu(mesh=mesh, filepath=str(output / "density_final"),
                      cell_data={"density": rho, "von_mises_normalized": per_cell_max(stress_measure)})
            np.savez_compressed(output / "final_optimizer_state.npz", **state_arrays)
            (output / "outer_history.json").write_text(
                json.dumps(getattr(optimizer, "outer_history", []), ensure_ascii=False, indent=2),
                encoding="utf-8")
            frames = view / "vtu"
            frames.mkdir(parents=True, exist_ok=True)
            stress_frames = (getattr(history, "field_histories", None) or {}).get("von_mises_stress") or []
            # 第 0 帧是优化前的初始构型 (优化器记录了才写, 只带密度), 其后每帧对应一次迭代
            if history.initial_physical_density is not None:
                write_vtu(mesh=mesh, filepath=str(frames / "density_iter_000"),
                          cell_data={"density": to_np(history.initial_physical_density[:]).reshape(-1)})
            for index, rho_i in enumerate(history.physical_densities, start=1):
                cells = {"density": to_np(rho_i).reshape(-1)}
                if len(stress_frames) >= index:
                    cells["von_mises_normalized"] = per_cell_max(stress_frames[index - 1])
                write_vtu(mesh=mesh, filepath=str(frames / f"density_iter_{index:03d}"), cell_data=cells)
            # ParaView 时间序列集合文件, 与 vtu/ 同级; 时间步即迭代步号
            steps = ([0] if history.initial_physical_density is not None else []) + list(
                range(1, len(history.physical_densities) + 1))
            (view / "evolution.pvd").write_text(
                '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                '<Collection>\n' + "".join(
                    f'<DataSet timestep="{i}" group="" part="0" file="vtu/density_iter_{i:03d}.vtu"/>\n'
                    for i in steps) + '</Collection>\n</VTKFile>\n', encoding="utf-8")
            payload = {"iter_indices": history.iter_indices, "changes": history.changes,
                       "iteration_times": history.iteration_times,
                       "scalar_histories": history.scalar_histories}
            (output / "history.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[done] {method} k={order}: {summary['optimization_iterations']} 步, "
                  f"volfrac={summary['volume_fraction']:.6f}, g_max(实体)={g_solid}, "
                  f"converged={summary['converged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
