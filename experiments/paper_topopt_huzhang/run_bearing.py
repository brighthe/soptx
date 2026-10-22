"""二维轴承装置近不可压缩拓扑优化 (论文 5.2.2 节 / 图 5.5~5.6, 表 5.4).

设计域 120 mm x 40 mm, 平面应变, 底边固支, 顶边竖直向下均布牵引. 两组材料:
可压缩基准组 nu = 0.3 (只插值 Young 模量) 与近不可压缩组 nu = 0.4999 (按式 (4.4)
同时插值 Young 模量与 Poisson 比). 每组以 LFEM p = 1, 2 与 HZMFEM k = 2 (跳量
稳定化), k = 3 (无需稳定化) 各做一次 OC 优化, 共八次; LFEM p = 1 用于对照线性位移元的体积自锁.

结果写入运行目录 ``outputs/<case>/<group>/<label>/`` (不入库, 已存在则拒绝覆盖); 确认后经
``sync_results.py`` (算例名取 ``<case>/<group>``) 固化到入库的 ``results/``. 逐步帧 ``iterations/`` 与
``evolution.pvd`` 留在运行目录, 需要时手动迁移到 Windows 查看目录 ``config.VIEW_DIR``.

    python run_bearing.py                                          # 两组材料 x 四种离散
    python run_bearing.py --group nu-0.4999 --analyzer lfem --order 1
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
from soptx.mesh import TriangleMesh
from soptx.postprocess.iteration_frames import IterationFrameWriter
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import BearingTraction2d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import OCOptimizer

CASE_ID = "bearing"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / CASE_ID   # 运行产物, 不入库
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 模型参数
DOMAIN = (0.0, 120.0, 0.0, 40.0)
TRACTION = -0.08             # N/mm, 顶边均布牵引强度; 问题类收合力 P = t * 跨度
E0 = 1.0                     # MPa
PLANE_TYPE = "plane_strain"
# 两组材料: 组名 (即 outputs/bearing/ 与 results/bearing/ 下的子目录名) -> (实体 Poisson 比, 材料插值对象)
GROUPS = {
    "nu-0.3": (0.3, "E"),
    "nu-0.4999": (0.4999, "E+nu"),
}
NU_VOID = 0.3                # E+nu 插值: 空材料 Poisson 比
NU_PENALTY = 1.0             # E+nu 插值: Poisson 比惩罚指数

# 离散: 对称单向对角三角网格 (nx 须为偶数)
NX, NY = 120, 40
MESH_TYPE = "triangle-single-diagonal-symmetric"
RUNS = (("lfem", 1), ("huzhang", 2), ("lfem", 2), ("huzhang", 3))   # 按位移逼近次数配对
# HZMFEM 角点松弛: 不松弛. Hu--Ma (2021) 4.1 节只在两条牵引边的角点数据不一致时需要松弛;
# 本模型顶边角点受法向压载 (0, t), 与零牵引侧边对 sigma_xy 的要求都是 0, 一致; 底边角点为位移--牵引角点
USE_RELAXATION = False
ASSEMBLY_METHOD = "standard"  # LFEM 单元刚度装配方法
SOLVER = "mumps"

# 优化: 密度过滤 + OC (MMA 及其渐近线扫描不能复现三拱构型)
VOLFRAC = 0.35
FILTER_RADIUS = 2.0          # mm
PENALTY = 3.0
EMIN = 1.0e-9
MOVE_LIMIT = 0.2
DAMPING = 0.5                # OC 阻尼指数
INITIAL_LAMBDA = 1.0e9       # OC 体积约束乘子二分的初始上界
BISECTION_TOL = 1.0e-3       # OC 乘子二分的相对收敛容差 (l2 - l1) / (l2 + l1)
MAX_ITERATIONS = 1000
CHANGE_TOLERANCE = 1.0e-2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析要跑的材料组、分析链与阶次; 缺省为两组材料 x 四种离散."""
    parser = argparse.ArgumentParser(description="二维轴承装置近不可压缩拓扑优化 (论文 5.2.2 节)")
    parser.add_argument("--group", choices=tuple(GROUPS), action="append",
                        help="只跑指定材料组 (可重复); 缺省为两组")
    parser.add_argument("--analyzer", choices=("lfem", "huzhang"), action="append",
                        help="只跑指定分析链 (可重复)")
    parser.add_argument("--order", type=int, choices=(1, 2, 3), action="append",
                        help="只跑指定阶次 (可重复)")
    return parser.parse_args(argv)


def run_config(group: str, method: str, order: int, label: str,
               stamp: dict[str, Any]) -> dict[str, Any]:
    """汇总一组运行的全部设置, 开跑前写入 ``config.json``.

    Parameters
    ----------
    group : str
        材料组, ``GROUPS`` 的键.
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
        按模型、离散、优化分组的设置, 取值全部来自模块顶部常量.
    """
    nu, variables = GROUPS[group]
    interpolation: dict[str, Any] = {"method": "msimp", "variables": variables, "penalty": PENALTY,
                                     "emin": EMIN}
    if variables == "E+nu":
        interpolation.update(nu_penalty_factor=NU_PENALTY, void_poisson_ratio=NU_VOID)
    return {
        "case_id": CASE_ID, "group": group, "method": method, "order": order, "label": label,
        "model": {"domain": list(DOMAIN), "traction": TRACTION, "E0": E0, "nu": nu,
                  "plane_type": PLANE_TYPE},
        "discretization": {"mesh_type": MESH_TYPE, "nx": NX, "ny": NY,
                           "integration_order": 2 * order + 2, "use_relaxation": USE_RELAXATION,
                           "assembly_method": ASSEMBLY_METHOD, "solver": SOLVER},
        "optimization": {"optimizer": "oc", "volume_fraction": VOLFRAC, "initial_density": VOLFRAC,
                         "filter_type": "density", "filter_radius": FILTER_RADIUS,
                         "interpolation": interpolation, "max_iterations": MAX_ITERATIONS,
                         "change_tolerance": CHANGE_TOLERANCE, "move_limit": MOVE_LIMIT,
                         "damping": DAMPING, "initial_lambda": INITIAL_LAMBDA,
                         "bisection_tol": BISECTION_TOL},
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
        首行为材料组/运行名与运行目录, 其后按模型、离散、优化分行; 完整设置 (含代码版本
        戳记) 以 ``config.json`` 为准.
    """
    model, disc, opt = config["model"], config["discretization"], config["optimization"]
    interp = opt["interpolation"]
    x0, x1, y0, y1 = model["domain"]
    material = f"nu={model['nu']:g} (插值 {interp['variables']}"
    if interp["variables"] == "E+nu":
        material += f", nu_void={interp['void_poisson_ratio']:g}, q={interp['nu_penalty_factor']:g}"
    material += ")"
    if config["method"] == "huzhang":
        chain = f"HZMFEM k={config['order']}"
        variant = f"角点松弛={'开' if disc['use_relaxation'] else '关'}"
    else:
        chain = f"LFEM p={config['order']}"
        variant = f"装配={disc['assembly_method']}"
    return "\n".join([
        f"[run] {config['group']}/{config['label']} -> {output}",
        f"  模型  {x1 - x0:g} x {y1 - y0:g} mm, {model['plane_type']}, E0={model['E0']:g}, {material}, "
        f"顶边均布牵引 t={model['traction']:g} N/mm",
        f"  离散  {chain}, 对称单向对角三角 {disc['nx']}x{disc['ny']}, 积分 {disc['integration_order']} 阶, "
        f"{variant}, {disc['solver']}",
        f"  优化  OC, 体积分数 {opt['volume_fraction']:g}, 密度过滤 r={opt['filter_radius']:g}, "
        f"MSIMP p={interp['penalty']:g}, 阻尼 {opt['damping']:g}, 迭代 <= {opt['max_iterations']}, "
        f"变化容差 {opt['change_tolerance']:g}, 移动限 {opt['move_limit']:g}",
    ])


def build(group: str, method: str, order: int, nx: int = NX, ny: int = NY) -> dict[str, Any]:
    """按受控比较协议组装一条分析链: 问题、网格、分析器、初始密度、目标与约束.

    Parameters
    ----------
    group : str
        材料组, ``GROUPS`` 的键.
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
    nu, variables = GROUPS[group]
    problem = BearingTraction2d(domain=DOMAIN, P=TRACTION * (DOMAIN[1] - DOMAIN[0]),
                              E=E0, nu=nu, plane_type=PLANE_TYPE)
    xmin, xmax, ymin, ymax = problem.domain
    mesh: Any = TriangleMesh.from_mirror_box(box=problem.domain, nx=nx, ny=ny)
    # 过滤矩阵等依赖网格上的 meshdata 元数据
    mesh.meshdata = {"domain": list(problem.domain), "mesh_type": MESH_TYPE,
                     "nx": nx, "ny": ny, "hx": (xmax - xmin) / nx, "hy": (ymax - ymin) / ny}

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E, poisson_ratio=problem.nu,
        hypothesis=problem.plane_type)
    options: dict[str, Any] = {"penalty_factor": PENALTY, "void_youngs_modulus": EMIN,
                               "target_variables": ["E"]}
    if variables == "E+nu":
        # 空区域退化为可压缩弱材料, 避免空单元的体积锁定污染实体区域
        options.update(target_variables=["E", "nu"], nu_penalty_factor=NU_PENALTY,
                       void_poisson_ratio=NU_VOID)
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp", options=options)
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
        "group": group, "method": method, "order": order, "problem": problem, "mesh": mesh,
        "analyzer": analyzer, "design_variable": design_variable, "density": density,
        "objective": ComplianceObjective(analyzer=analyzer, state_variable=state_variable,
                                         diff_mode="manual"),
        "constraint": VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC,
                                       diff_mode="manual"),
    }


def main(argv: list[str] | None = None) -> int:
    """逐组: 记录配置 -> 组装 -> 密度过滤 + OC 优化 -> 终态求解 -> 落盘."""
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
    # 开跑前统一检查全部运行目录, 不在跑完前几组后才因目录已存在而中断
    targets = [(group, method, order, OUTPUT_DIR / group / f"analyzer-{method}__order-{order}")
               for group in args.group or GROUPS for method, order in runs]
    existing = [str(output) for *_, output in targets if output.exists()]
    if existing:
        raise SystemExit(f"运行目录已存在, 拒绝覆盖: {existing}; 不再需要时先删除 (要保留先经 sync_results.py 固化)")

    for group, method, order, output in targets:
        # 0. 开跑前记录本组的全部设置, 中途中断也能复原运行条件
        config = run_config(group, method, order, output.name, stamp)
        print("\n" + format_run_header(config, output), flush=True)
        output.mkdir(parents=True)
        (output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        parts = build(group, method, order)
        analyzer, problem = parts["analyzer"], parts["problem"]

        # 1. 密度过滤 + OC 优化
        density_filter = Filter(design_mesh=parts["mesh"], filter_type="density",
                                rmin=FILTER_RADIUS, density_location="element", filter_q=3)
        optimizer = OCOptimizer(
            objective=parts["objective"], constraint=parts["constraint"], filter=density_filter,
            options={"max_iterations": MAX_ITERATIONS, "change_tolerance": CHANGE_TOLERANCE},
            enable_logging=True,
        )
        optimizer.options.set_advanced_options(
            move_limit=MOVE_LIMIT, damping_coef=DAMPING, initial_lambda=INITIAL_LAMBDA,
            bisection_tol=BISECTION_TOL)
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

        nu, variables = GROUPS[group]
        interpolation: dict[str, Any] = {"variables": variables}
        if variables == "E+nu":
            interpolation.update(nu_penalty_factor=NU_PENALTY, void_poisson_ratio=NU_VOID)
        summary = {
            "case_id": CASE_ID,
            "group": group,
            "poisson_ratio": nu,
            "model": "BearingTraction2d",
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
        print(f"[done] {CASE_ID}/{group} {method} k={order}: {summary['optimization_iterations']} 步, "
              f"C={compliance:.6f}, volfrac={summary['volume_fraction']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
