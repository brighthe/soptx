"""二维轴承装置近不可压缩拓扑优化 (论文 5.2.2 节 / 图 5.5~5.6, 表 5.4).

设计域 120 mm x 40 mm, 平面应变, 底边固支, 顶边竖直向下均布牵引. 两组材料:
可压缩基准组 nu = 0.3 (只插值 Young 模量) 与近不可压缩组 nu = 0.4999 (按式 (4.4)
同时插值 Young 模量与 Poisson 比). 每组以 LFEM p = 1, 2 与 HZMFEM k = 2 (跳量
稳定化) 各做一次 OC 优化, 共六次; LFEM p = 1 用于对照线性位移元的体积自锁.

网格为对称单向对角三角网格 (左半 "/", 右半 "\", 与问题左右对称), 低阶位移元在该
剖分上出现经典体积自锁. 每组写入 ``results/<case>/analyzer-<lfem|huzhang>__order-<k>/``.
用法::

    python run_bearing.py                                          # 两组材料 x 三种离散
    python run_bearing.py --case bearing-incompressible --analyzer lfem --order 1
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
from soptx.mesh import create_huzhang_symmetric_single_diagonal_mesh
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import BearingDevice2d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import OCOptimizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 模型参数
TRACTION = -0.08             # N/mm, 顶边均布牵引
E0 = 1.0                     # MPa
PLANE_TYPE = "plane_strain"
# 两组材料: 算例 id -> (实体 Poisson 比, 材料插值对象)
CASES = {
    "bearing-compressible": (0.3, "E"),
    "bearing-incompressible": (0.4999, "E+nu"),
}
NU_VOID = 0.3                # E+nu 插值: 空材料 Poisson 比
NU_PENALTY = 1.0             # E+nu 插值: Poisson 比惩罚指数

# 离散: 对称单向对角三角网格 (nx 须为偶数)
NX, NY = 120, 40
MESH_TYPE = "triangle-single-diagonal-symmetric"
RUNS = (("lfem", 1), ("lfem", 2), ("huzhang", 2))   # 每组材料的三种离散
USE_RELAXATION = True        # HZMFEM 角点松弛
ASSEMBLY_METHOD = "standard"  # LFEM 单元刚度装配方法
SOLVER = "mumps"

# 优化: 密度过滤 + OC (MMA 及其渐近线扫描不能复现三拱构型)
VOLFRAC = 0.35
FILTER_RADIUS = 2.0          # mm
PENALTY = 3.0
EMIN = 1.0e-9
MOVE_LIMIT = 0.2
DAMPING = 0.5                # OC 阻尼指数
MAX_ITERATIONS = 1000
CHANGE_TOLERANCE = 1.0e-2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析要跑的材料组、分析链与阶次; 缺省为两组材料 x 三种离散."""
    parser = argparse.ArgumentParser(description="二维轴承装置近不可压缩拓扑优化 (论文 5.2.2 节)")
    parser.add_argument("--case", choices=tuple(CASES), action="append",
                        help="只跑指定材料组 (可重复); 缺省为两组")
    parser.add_argument("--analyzer", choices=("lfem", "huzhang"), action="append",
                        help="只跑指定分析链 (可重复)")
    parser.add_argument("--order", type=int, choices=(1, 2), action="append",
                        help="只跑指定阶次 (可重复)")
    return parser.parse_args(argv)


def build(case_id: str, method: str, order: int, nx: int = NX, ny: int = NY) -> dict[str, Any]:
    """按受控比较协议组装一条分析链: 问题、网格、分析器、初始密度、目标与约束.

    Parameters
    ----------
    case_id : str
        材料组, ``CASES`` 的键.
    method : {'lfem', 'huzhang'}
        分析链.
    order : int
        阶次 k (LFEM 位移阶 p = k, HZMFEM 应力阶 k); 再分析可取 ``RUNS`` 之外的阶次
        (如表 5.4 的参考列 HZMFEM k = 4).
    nx, ny : int
        网格剖分; 图 5.5 的全实体 h 收敛考察逐级加密.

    Returns
    -------
    dict[str, Any]
        组装好的对象; 冻结设计再分析只用其中的分析链, 不挂优化器.
    """
    nu, variables = CASES[case_id]
    problem = BearingDevice2d(t=TRACTION, E=E0, nu=nu, plane_type=PLANE_TYPE)
    xmin, xmax, ymin, ymax = problem.domain
    mesh: Any = create_huzhang_symmetric_single_diagonal_mesh(box=problem.domain, nx=nx, ny=ny)
    # 过滤矩阵等依赖网格上的 meshdata 元数据
    mesh.meshdata = {"domain": list(problem.domain), "mesh_type": MESH_TYPE,
                     "nx": nx, "ny": ny, "hx": (xmax - xmin) / nx, "hy": (ymax - ymin) / ny}

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E, poisson_ratio=problem.nu,
        hypothesis=problem.plane_type, enable_logging=False)
    options: dict[str, Any] = {"penalty_factor": PENALTY, "void_youngs_modulus": EMIN,
                               "target_variables": ["E"]}
    if variables == "E+nu":
        # 空区域退化为可压缩弱材料, 避免空单元的体积锁定污染实体区域
        options.update(target_variables=["E", "nu"], nu_penalty_factor=NU_PENALTY,
                       void_poisson_ratio=NU_VOID)
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp", options=options,
        enable_logging=False)
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
        "case_id": case_id, "method": method, "order": order, "problem": problem, "mesh": mesh,
        "analyzer": analyzer, "design_variable": design_variable, "density": density,
        "objective": ComplianceObjective(analyzer=analyzer, state_variable=state_variable,
                                         diff_mode="manual", enable_logging=False),
        "constraint": VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC,
                                       diff_mode="manual", enable_logging=False),
    }


def main(argv: list[str] | None = None) -> int:
    """逐组: 组装 -> 密度过滤 + OC 优化 -> 终态求解 -> 落盘."""
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

    runs = [(m, o) for m, o in RUNS
            if (not args.analyzer or m in args.analyzer) and (not args.order or o in args.order)]
    for case_id in args.case or CASES:
        for method, order in runs:
            output = RESULTS_DIR / case_id / f"analyzer-{method}__order-{order}"
            print(f"\n[run] {case_id}: analyzer={method}, order={order} -> {output}", flush=True)
            parts = build(case_id, method, order)
            analyzer, problem = parts["analyzer"], parts["problem"]

            # 1. 密度过滤 + OC 优化
            density_filter = Filter(design_mesh=parts["mesh"], filter_type="density",
                                    rmin=FILTER_RADIUS, density_location="element", filter_q=3,
                                    enable_logging=False)
            optimizer = OCOptimizer(
                objective=parts["objective"], constraint=parts["constraint"], filter=density_filter,
                options={"max_iterations": MAX_ITERATIONS, "change_tolerance": CHANGE_TOLERANCE},
                enable_logging=True,
            )
            optimizer.options.set_advanced_options(
                move_limit=MOVE_LIMIT, damping_coef=DAMPING, initial_lambda=1.0e9, bisection_tol=1.0e-3)
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

            nu, variables = CASES[case_id]
            interpolation = {"variables": variables}
            if variables == "E+nu":
                interpolation.update(nu_penalty_factor=NU_PENALTY, void_poisson_ratio=NU_VOID)
            summary = {
                "case_id": case_id,
                "model": "BearingDevice2d",
                "method": method,
                "order": order,
                "mesh_type": MESH_TYPE,
                "nx": NX,
                "ny": NY,
                "optimizer": "oc",
                "load_discretization": None,
                "integration_order": 2 * order + 2,
                "stabilization_coefficient": getattr(analyzer, "stabilization_coefficient", None),
                "volume_fraction": VOLFRAC + float(parts["constraint"].fun(density)),
                "relative_equilibrium_residual": residual,
                "optimization_iterations": len(history.iter_indices),
                "converged": bool(history.changes and history.changes[-1] <= CHANGE_TOLERANCE),
                "termination_reason": None,
                "solver": SOLVER,
                "interpolation": interpolation,
                "provenance": stamp,
                "compliance": compliance,
                "compliance_domain": "full",
                "full_structure_factor": 1.0,
            }

            # 3. 落盘: 最终密度、逐步帧 vtu/ (不入库)、收敛历史与摘要
            output.mkdir(parents=True, exist_ok=True)
            mesh = parts["mesh"]
            write_vtu(mesh=mesh, filepath=str(output / "density_final"),
                      cell_data={"density": to_cells(density[:])})
            frames = output / "vtu"
            frames.mkdir(exist_ok=True)
            # 第 0 帧是优化前的初始构型 (优化器记录了才写), 其后每帧对应一次迭代
            if history.initial_physical_density is not None:
                write_vtu(mesh=mesh, filepath=str(frames / "density_iter_000"),
                          cell_data={"density": to_cells(history.initial_physical_density[:])})
            for index, rho in enumerate(history.physical_densities, start=1):
                write_vtu(mesh=mesh, filepath=str(frames / f"density_iter_{index:03d}"),
                          cell_data={"density": to_cells(rho)})
            payload = {"iter_indices": history.iter_indices, "changes": history.changes,
                       "iteration_times": history.iteration_times,
                       "scalar_histories": history.scalar_histories}
            (output / "history.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[done] {case_id} {method} k={order}: {summary['optimization_iterations']} 步, "
                  f"C={compliance:.6f}, volfrac={summary['volume_fraction']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
