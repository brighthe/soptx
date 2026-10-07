"""两端固支梁柔顺度拓扑优化 (论文 5.2.1 节 / 图 5.2~5.3, 表 5.3).

完整梁 160 mm x 20 mm, 下边界中点受集中力; 利用几何与受载对称性只算左半域
80 mm x 20 mm, 对称面 x = 80 上施加 u_x = 0 与 sigma_xy = 0, 柔顺度按完整结构
(半域值 x 2) 报告. LFEM p = 2, 3, 4 与 HZMFEM k = 2, 3, 4 各做一次 MMA 优化.
    python run_fixed_fixed.py                                # 六组
    python run_fixed_fixed.py --analyzer huzhang --order 2   # 只跑其中几组
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
from soptx.fem import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer, project_patch_traction_to_p1_trace
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import create_huzhang_checkerboard_mesh
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import FixedFixedBeamHalfDomain2d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import MMAOptimizer

CASE_ID = "compliance-fixed-fixed-half"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / CASE_ID
# ParaView 查看副本 (Windows 本地盘; 经 \\wsl.localhost 读大批帧很慢): 逐步帧 vtu/ 与
# evolution.pvd 写到这里 (最终密度只在 results/), 目录结构与 results/ 一一对应; 该盘不可用时退回 results/
VIEW_ROOT = Path("/mnt/c/workspace/soptx-results/paper_topopt_huzhang")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 模型参数 (完整结构的载荷 P; 左半域承担 P / 2, 由模型类与 P1 迹投影自动给出)
LOAD = -3.0                  # N
LOAD_WIDTH = 1.0             # mm, 集中力化为均布牵引的接触宽度
E0 = 30.0                    # MPa
NU = 0.4
PLANE_TYPE = "plane_stress"
FULL_STRUCTURE_FACTOR = 2.0  # 半域柔顺度 x 2 = 完整结构柔顺度

# 离散: 棋盘格交替对角三角网格 (nx, ny 须为偶数), 四个角点满足角点松弛的拓扑要求
NX, NY = 80, 20
METHODS = ("lfem", "huzhang")
ORDERS = (2, 3, 4)
USE_RELAXATION = True        # HZMFEM 角点松弛
ASSEMBLY_METHOD = "standard"  # LFEM 单元刚度装配方法
SOLVER = "mumps"

# 优化: MSIMP 只插值 Young 模量 (nu = 0.4 可压缩); 密度过滤 + MMA
VOLFRAC = 0.4
FILTER_RADIUS = 2.4          # mm
PENALTY = 3.0
EMIN = 1.0e-9
MOVE_LIMIT = 0.2
ASYMP_INIT = 0.5
MAX_ITERATIONS = 500
CHANGE_TOLERANCE = 1.0e-2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析要跑的分析链与阶次; 缺省为全部六组."""
    parser = argparse.ArgumentParser(description="两端固支梁柔顺度拓扑优化 (论文 5.2.1 节)")
    parser.add_argument("--analyzer", choices=METHODS, action="append",
                        help="只跑指定分析链 (可重复); 缺省为两条链")
    parser.add_argument("--order", type=int, choices=ORDERS, action="append",
                        help="只跑指定阶次 (可重复); 缺省为 2, 3, 4")
    return parser.parse_args(argv)


def build(method: str, order: int) -> dict[str, Any]:
    """按受控比较协议组装一条分析链: 问题、网格、分析器、初始密度、目标与约束.

    Parameters
    ----------
    method : {'lfem', 'huzhang'}
        分析链.
    order : int
        阶次 k (LFEM 位移阶 p = k, HZMFEM 应力阶 k).

    Returns
    -------
    dict[str, Any]
        组装好的对象; 冻结设计再分析只用其中的分析链, 不挂优化器.
    """
    # 先按原始物理问题取载荷区几何, 再把牵引换成它在底边 P1 迹空间上的 L2 投影
    problem = FixedFixedBeamHalfDomain2d(
        P=LOAD, E=E0, nu=NU, load_width=LOAD_WIDTH, plane_type=PLANE_TYPE)
    xmin, xmax, ymin, ymax = problem.domain
    traction = project_patch_traction_to_p1_trace(
        line=(xmin, xmax), n_cells=NX, level=problem.traction_level,
        patch=problem.traction_patch, intensity=problem.traction_intensity,
    )
    problem = FixedFixedBeamHalfDomain2d(
        P=LOAD, E=E0, nu=NU, load_width=LOAD_WIDTH, plane_type=PLANE_TYPE, traction=traction)

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
        state_variable = "u"
    else:
        analyzer = HuZhangMFEMAnalyzer(use_relaxation=USE_RELAXATION, **common)
        state_variable = "sigma"

    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh, displacement_mesh=mesh, relative_density=VOLFRAC)
    return {
        "method": method, "order": order, "problem": problem, "mesh": mesh,
        "analyzer": analyzer, "design_variable": design_variable, "density": density,
        "objective": ComplianceObjective(analyzer=analyzer, state_variable=state_variable,
                                         diff_mode="manual", enable_logging=False),
        "constraint": VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC,
                                       diff_mode="manual", enable_logging=False),
    }


def main(argv: list[str] | None = None) -> int:
    """逐组: 组装 -> 密度过滤 + MMA 优化 -> 终态求解 -> 落盘."""
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
    to_cells = lambda value: np.asarray(bm.to_numpy(value), dtype=np.float64).flatten()

    for method in args.analyzer or METHODS:
        for order in args.order or ORDERS:
            label = f"analyzer-{method}__order-{order}"
            output = OUTPUT_DIR / label
            view = VIEW_ROOT / CASE_ID / label if VIEW_ROOT.parent.is_dir() else output
            print(f"\n[run] {CASE_ID}: analyzer={method}, order={order} -> {output} (帧: {view})",
                  flush=True)
            parts = build(method, order)
            analyzer, problem = parts["analyzer"], parts["problem"]

            # 1. 密度过滤 + MMA 优化
            density_filter = Filter(design_mesh=parts["mesh"], filter_type="density",
                                    rmin=FILTER_RADIUS, density_location="element", filter_q=3,
                                    enable_logging=False)
            optimizer = MMAOptimizer(
                objective=parts["objective"], constraint=parts["constraint"], filter=density_filter,
                options={"max_iterations": MAX_ITERATIONS, "change_tolerance": CHANGE_TOLERANCE},
                enable_logging=True,
            )
            optimizer.options.move_limit = MOVE_LIMIT
            optimizer.options.asymp_init = ASYMP_INIT
            density, history = optimizer.optimize(
                design_variable=parts["design_variable"], density_distribution=parts["density"])

            # 2. 在最终物理密度上重新求解, 柔顺度与平衡残差都取自这次求解
            state = analyzer.solve_state(rho_val=density)
            compliance = float(parts["objective"].fun(density=density, state=state))
            if method == "huzhang":
                residual = float(analyzer.relative_state_residual())
            else:
                # LFEM 只计非 Dirichlet 自由度
                r = analyzer.stiffness_matrix.matmul(state["displacement"][:]) - analyzer.force_vector
                free = ~analyzer.tensor_space.is_boundary_dof(
                    threshold=problem.is_dirichlet_boundary(), method="interp")
                residual = float(bm.linalg.norm(r[free])) / max(
                    float(bm.linalg.norm(analyzer.force_vector[free])), 1.0e-30)

            summary = {
                "case_id": CASE_ID,
                "model": "FixedFixedBeamHalfDomain2d",
                "method": method,
                "order": order,
                "mesh_type": "triangle-checkerboard",
                "nx": NX,
                "ny": NY,
                "optimizer": "mma",
                "load_discretization": "p1_trace_l2_projection",
                "integration_order": 2 * order + 2,
                "stabilization_coefficient": getattr(analyzer, "stabilization_coefficient", None),
                "volume_fraction": VOLFRAC + float(parts["constraint"].fun(density)),
                "relative_equilibrium_residual": residual,
                "optimization_iterations": len(history.iter_indices),
                "converged": bool(history.changes and history.changes[-1] <= CHANGE_TOLERANCE),
                "termination_reason": None,
                "solver": SOLVER,
                "interpolation": {"variables": "E"},
                "provenance": stamp,
                # 原始产物保存计算域 (半域) 柔顺度, 完整结构换算只在展示层进行
                "compliance": compliance,
                "compliance_domain": "half",
                "full_structure_factor": FULL_STRUCTURE_FACTOR,
            }

            # 3. 落盘: 摘要、收敛历史与最终密度写 output (入库, 计算依据); 逐步帧 vtu/ 与
            #    evolution.pvd 写 view (ParaView 查看副本, 不入库), 每个文件只存一处
            mesh = parts["mesh"]
            output.mkdir(parents=True, exist_ok=True)
            write_vtu(mesh=mesh, filepath=str(output / "density_final"),
                      cell_data={"density": to_cells(density[:])})
            frames = view / "vtu"
            frames.mkdir(parents=True, exist_ok=True)
            # 第 0 帧是优化前的初始构型 (优化器记录了才写), 其后每帧对应一次迭代
            if history.initial_physical_density is not None:
                write_vtu(mesh=mesh, filepath=str(frames / "density_iter_000"),
                          cell_data={"density": to_cells(history.initial_physical_density[:])})
            for index, rho in enumerate(history.physical_densities, start=1):
                write_vtu(mesh=mesh, filepath=str(frames / f"density_iter_{index:03d}"),
                          cell_data={"density": to_cells(rho)})
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
                  f"C_full={FULL_STRUCTURE_FACTOR * compliance:.6f}, "
                  f"volfrac={summary['volume_fraction']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
