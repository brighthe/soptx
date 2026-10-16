"""两端固支梁柔顺度拓扑优化 (论文 5.2.1 节 / 图 5.2~5.3, 表 5.3).

完整梁 160 mm x 20 mm, 下边界中点受集中力; 利用几何与受载对称性只算左半域
80 mm x 20 mm, 对称面 x = 80 上施加 u_x = 0 与 sigma_xy = 0, 柔顺度按完整结构
(半域值 x 2) 报告. LFEM p = 2, 3, 4 与 HZMFEM k = 2, 3, 4 各做一次 MMA 优化.

结果写入运行目录 ``outputs/<case>/<label>/`` (不入库, 已存在则拒绝覆盖); 确认后经
``sync_results.py`` 固化到入库的 ``results/``. 逐步帧 ``iterations/`` 与 ``evolution.pvd`` 留在运行目录,
需要时手动迁移到 Windows 查看目录 ``config.VIEW_DIR``.

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
from soptx.mesh import TriangleMesh
from soptx.postprocess.iteration_frames import IterationFrameWriter
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import HalfFixedFixedBeamPatch2d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import MMAOptimizer

CASE_ID = "compliance-fixed-fixed-half"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / CASE_ID   # 运行产物, 不入库
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 模型参数 (完整结构的载荷 P; 左半域承担 P / 2, 由模型类与 P1 迹投影自动给出)
LOAD = -3.0                  # N
LOAD_WIDTH = 1.0             # mm, 集中力化为均布牵引的接触宽度
E0 = 30.0                    # MPa
NU = 0.4
PLANE_TYPE = "plane_stress"
FULL_STRUCTURE_FACTOR = 2.0  # 半域柔顺度 x 2 = 完整结构柔顺度

# 离散: 棋盘格交替对角三角网格 (nx, ny 须为偶数)
NX, NY = 80, 20
METHODS = ("lfem", "huzhang")
ORDERS = (2, 3, 4)
# HZMFEM 角点松弛: 不松弛. Hu--Ma (2021) 4.1 节只在两条牵引边的角点数据不一致时需要松弛;
# 本模型对称面角点 (80, 0)、(80, 20) 两边对 sigma_xy 的要求都是 0, 一致; 固支角点为位移--牵引角点
USE_RELAXATION = False
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


def run_config(method: str, order: int, label: str, stamp: dict[str, Any]) -> dict[str, Any]:
    """汇总一组运行的全部设置, 开跑前写入 ``config.json``.

    Parameters
    ----------
    method : {'lfem', 'huzhang'}
        分析链.
    order : int
        阶次 k (LFEM 位移阶 p = k, HZMFEM 应力阶 k).
    label : str
        运行名, 即运行目录名.
    stamp : dict
        代码版本戳记.

    Returns
    -------
    config : dict
        按模型、离散、优化分组的设置, 取值全部来自模块顶部常量; 计算域取模型类的默认值.
    """
    domain = HalfFixedFixedBeamPatch2d(
        P=LOAD, E=E0, nu=NU, load_width=LOAD_WIDTH, plane_type=PLANE_TYPE).domain
    return {
        "case_id": CASE_ID, "method": method, "order": order, "label": label,
        "model": {"domain": [float(v) for v in domain], "load": LOAD, "load_width": LOAD_WIDTH,
                  "E0": E0, "nu": NU, "plane_type": PLANE_TYPE,
                  "full_structure_factor": FULL_STRUCTURE_FACTOR},
        "discretization": {"mesh_type": "triangle-checkerboard", "nx": NX, "ny": NY,
                           "integration_order": 2 * order + 2, "use_relaxation": USE_RELAXATION,
                           "load_discretization": "p1_trace_l2_projection",
                           "assembly_method": ASSEMBLY_METHOD, "solver": SOLVER},
        "optimization": {"volume_fraction": VOLFRAC, "initial_density": VOLFRAC,
                         "filter_type": "density", "filter_radius": FILTER_RADIUS,
                         "penalty": PENALTY, "emin": EMIN, "max_iterations": MAX_ITERATIONS,
                         "change_tolerance": CHANGE_TOLERANCE, "move_limit": MOVE_LIMIT,
                         "asymp_init": ASYMP_INIT},
        "provenance": stamp,
    }


def format_run_header(config: dict[str, Any], output: Path) -> str:
    """把一组运行的关键设置排成启动时打印的摘要.

    Parameters
    ----------
    config : dict
        ``run_config`` 构造的完整设置, 与落盘的 ``config.json`` 同源.
    output : Path
        运行目录.

    Returns
    -------
    header : str
        首行为运行名与运行目录, 其后按模型、离散、优化分行; 完整设置 (含代码版本戳记) 以
        ``config.json`` 为准.
    """
    model, disc, opt = config["model"], config["discretization"], config["optimization"]
    x0, x1, y0, y1 = model["domain"]
    if config["method"] == "huzhang":
        chain = f"HZMFEM k={config['order']}"
        variant = f"角点松弛={'开' if disc['use_relaxation'] else '关'}"
    else:
        chain = f"LFEM p={config['order']}"
        variant = f"装配={disc['assembly_method']}"
    return "\n".join([
        f"[run] {config['label']} -> {output}",
        f"  模型  左半域 {x1 - x0:g} x {y1 - y0:g} mm (完整结构 x{model['full_structure_factor']:g}), "
        f"{model['plane_type']}, E0={model['E0']:g}, nu={model['nu']:g}, "
        f"P={model['load']:g} N (完整结构) 均布于 l={model['load_width']:g} mm",
        f"  离散  {chain}, 棋盘三角 {disc['nx']}x{disc['ny']}, 积分 {disc['integration_order']} 阶, "
        f"{variant}, {disc['solver']}",
        f"  优化  MMA, 体积分数 {opt['volume_fraction']:g}, 密度过滤 r={opt['filter_radius']:g}, "
        f"MSIMP p={opt['penalty']:g}, 迭代 <= {opt['max_iterations']}, "
        f"变化容差 {opt['change_tolerance']:g}, 移动限 {opt['move_limit']:g}",
    ])


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
    problem = HalfFixedFixedBeamPatch2d(
        P=LOAD, E=E0, nu=NU, load_width=LOAD_WIDTH, plane_type=PLANE_TYPE)
    xmin, xmax, ymin, ymax = problem.domain
    traction = project_patch_traction_to_p1_trace(
        line=(xmin, xmax), n_cells=NX, level=problem.traction_level,
        patch=problem.traction_patch, intensity=problem.traction_intensity,
        axis=problem.traction_axis,
    )
    problem = HalfFixedFixedBeamPatch2d(
        P=LOAD, E=E0, nu=NU, load_width=LOAD_WIDTH, plane_type=PLANE_TYPE, traction=traction)

    mesh: Any = TriangleMesh.from_checkerboard_box(box=problem.domain, nx=NX, ny=NY)
    # 过滤矩阵等依赖网格上的 meshdata 元数据
    mesh.meshdata = {"domain": list(problem.domain), "mesh_type": "triangle-checkerboard",
                     "nx": NX, "ny": NY, "hx": (xmax - xmin) / NX, "hy": (ymax - ymin) / NY}

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E, poisson_ratio=problem.nu,
        hypothesis=problem.plane_type)
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp",
        options={"penalty_factor": PENALTY, "void_youngs_modulus": EMIN, "target_variables": ["E"]},
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
                                         diff_mode="manual"),
        "constraint": VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC,
                                       diff_mode="manual"),
    }


def main(argv: list[str] | None = None) -> int:
    """逐组: 记录配置 -> 组装 -> 密度过滤 + MMA 优化 -> 终态求解 -> 落盘."""
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

    # 开跑前统一检查全部运行目录, 不在跑完前几组后才因目录已存在而中断
    runs = [(method, order, OUTPUT_DIR / f"analyzer-{method}__order-{order}")
            for method in args.analyzer or METHODS for order in args.order or ORDERS]
    existing = [str(output) for _, _, output in runs if output.exists()]
    if existing:
        raise SystemExit(f"运行目录已存在, 拒绝覆盖: {existing}; 不再需要时先删除 (要保留先经 sync_results.py 固化)")

    for method, order, output in runs:
        # 0. 开跑前记录本组的全部设置, 中途中断也能复原运行条件
        config = run_config(method, order, output.name, stamp)
        print("\n" + format_run_header(config, output), flush=True)
        output.mkdir(parents=True)
        (output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        parts = build(method, order)
        analyzer, problem = parts["analyzer"], parts["problem"]

        # 1. 密度过滤 + MMA 优化
        density_filter = Filter(design_mesh=parts["mesh"], filter_type="density",
                                rmin=FILTER_RADIUS, density_location="element", filter_q=3)
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
            "model": "HalfFixedFixedBeamPatch2d",
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

        # 3. 落盘 (全部写 output, 不入库, 确认后经 sync_results.py 固化到 results/):
        #    初始与最终密度 vtu、终态位移/应力 npy、两份 json. 初始构型是优化器记录的优化前
        #    物理密度; 没记录即报错, 不静默缺文件
        mesh = parts["mesh"]
        if history.initial_physical_density is None:
            raise RuntimeError("优化器没有记录初始物理密度, 无法写出 density_initial.vtu.")
        initial_cells = {"density": to_cells(history.initial_physical_density[:])}
        write_vtu(mesh=mesh, filepath=str(output / "density_initial"), cell_data=initial_cells)
        write_vtu(mesh=mesh, filepath=str(output / "density_final"),
                  cell_data={"density": to_cells(density[:])})
        # 终态求解 (第 2 步) 的位移自由度向量; Hu--Zhang 另存应力自由度向量 (松弛坐标, 按 huzhang_space.value 取值)
        np.save(output / "displacement_final.npy", np.asarray(bm.to_numpy(state["displacement"][:])))
        if "stress" in state:
            np.save(output / "stress_final.npy", np.asarray(bm.to_numpy(state["stress"][:])))
        payload = {"iter_indices": history.iter_indices, "changes": history.changes,
                   "iteration_times": history.iteration_times,
                   "scalar_histories": history.scalar_histories}
        (output / "history.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # 4. 逐步帧: 第 0 帧是初始构型, 其后每帧对应一次迭代 (帧由优化器记录的历史在结束后
        #    一次写出); IterationFrameWriter 写 iterations/iter_NNNN.vtu 与 evolution.pvd
        #    到运行目录, 留在原处, 需要时手动迁移到 Windows 查看目录
        frames = IterationFrameWriter(output, mesh)
        frames.write(0, cell_data=initial_cells)
        for index, rho in enumerate(history.physical_densities, start=1):
            frames.write(index, cell_data={"density": to_cells(rho)})
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {method} k={order}: {summary['optimization_iterations']} 步, "
              f"C_full={FULL_STRUCTURE_FACTOR * compliance:.6f}, "
              f"volfrac={summary['volume_fraction']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
