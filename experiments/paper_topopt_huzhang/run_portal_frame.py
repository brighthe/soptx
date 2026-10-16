"""门式框架局部应力约束体积最小化 (论文 5.2.4 节 / 图 5.11~5.13).

完整域 L x H 的矩形去掉底部中央三角形缺口, 平面应力, 左支座段固定铰、右支座段滚动铰
(PolyStress 口径), 顶边中点竖直合力 P 化为宽 l 的均布牵引, 按论文口径投影到顶边连续 P1 迹
空间后施加. 支座不对称, 状态场不严格对称, 因此按全域求解; 设计对称性由过滤器对 x = L/2 做
对称化 (symmetry_planes). 设计域含斜边, 在 gmsh 生成左半域再镜像到右半的非结构三角形网格
(严格关于 x = L/2 对称) 上求解. 以表观应力松弛约束下的体积最小化为目标, 增广拉格朗日 + MMA
(AL-MMA) 求解. 顶边载荷是法向区段牵引, 端点处应力只间断不奇异 (全实体加密实测有界, 约 0.70),
故不设被动实体区, 全部单元受应力约束 (与 PolyStress 一致); 凹角顶点同样不设.

结果写入运行目录 ``outputs/<case>/<label>/`` (不入库, 已存在则拒绝覆盖); 确认后经
``sync_results.py`` 固化到入库的 ``results/``. 逐步帧 ``iterations/`` 与 ``evolution.pvd`` 留在运行目录,
需要时手动迁移到 Windows 查看目录 ``config.VIEW_DIR``.

    python run_portal_frame.py --analyzer lfem --order 3   # 单组
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gmsh
import numpy as np

from soptx.backend import backend_manager as bm
from soptx.fem import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer, project_patch_traction_to_p1_trace
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TriangleMesh
from soptx.postprocess.iteration_frames import IterationFrameWriter
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import PortalFramePatch2d
from soptx.sparse import COOTensor, CSRTensor
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

from analysis import provenance

CASE_ID = "portal-frame-2d-stress"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / CASE_ID   # 运行产物, 不入库

# 模型参数.
L, H = 12.0, 6.0             # mm, 域宽与高
LOAD_WIDTH = 1.0             # mm, 顶边载荷区宽度 l
SUPPORT_WIDTH = 0.55         # mm, 底边支座段宽度 b
CORNER_HEIGHT = 3.5          # mm, 凹角顶点高度 h_c
LOAD = -300.0                # N, 顶边竖直合力
E0 = 1.0                     # MPa (参考值 100000)
NU = 0.25
PLANE_TYPE = "plane_stress"
STRESS_LIMIT = 1000.0        # MPa, 材料许用应力

# 应力约束参数
EPSILON = 1.0e-3             # 松弛参数
LOAD_PAD_RADIUS = 0.0        # 载荷端点被动实体区半径 (mm); 0 表示不处置. 法向区段载荷端点无奇异
                             # (2026-10-16 全实体 h 加密: 端点邻域实体应力比收敛到约 0.70), 设 0.25
                             # 时被动区内应力比达 1.17 (HZ k=3), 是豁免被优化器利用而非奇异
SUPPORT_PAD_RADIUS = 0.0     # 支座端点被动实体区半径 (mm); 0 表示不处置

# 有限元离散参数: gmsh 在左半域生成非结构三角形网格, 再镜像到右半
MESH_SIZE = 0.0625           # mm, 目标单元尺寸 h (R/h=4)
MESH_ALGORITHM = 6           # gmsh Mesh.Algorithm: 6 = Frontal-Delaunay
METHODS = ("lfem", "huzhang")
ORDERS = (1, 2, 3, 4)        # 1 只供 LFEM (p=1 对照 PolyStress 的线性单元); HZMFEM 要求 k >= 2
# HZMFEM 不做角点松弛: 各几何角点处牵引为零, 两侧数据相容, 未松弛的角点由分析器按 Hu & Ma (2021)
# 式 (4.1) 联立两侧强约束施加; 凹角与支座段内端通常不满足两单元松弛拓扑, 无法统一松弛
USE_RELAXATION = False
ASSEMBLY_METHOD = "standard"  # LFEM 单元刚度装配方法
SOLVER = "mumps"

# 优化参数: MSIMP 只插值 Young 模量; 密度过滤 + tanh 投影 (beta 1 -> 10, 每 5 个外层步 +1)
FILTER_RADIUS = 0.25         # mm
PENALTY = 3.5
EMIN = 1.0e-9
INITIAL_DENSITY = 0.5        # 满密度启动会导致 ALM 发散
PROJECTION = {"continuation_strategy": "additive", "projection_type": "tanh", "beta": 1.0,
              "beta_max": 10.0, "continuation_iter": 5, "beta_increment": 1.0}
# AL-MMA; 停止准则 C0 连续化终止 / C1 设计稳定 / C2 松弛可行 / C3 连续 HOLD_STEPS 个外层步同时满足
MAX_AL_ITERATIONS = 200
MMA_ITERS_PER_AL = 5
CHANGE_TOLERANCE = 2.0e-3
CHANGE_MEASURE = "mean"      # C1: 外层步首末设计变量的平均绝对变化 (PolyStress 口径)
STRESS_TOLERANCE = 5.0e-3    # C2: delta_g
HOLD_STEPS = 10               # C3 只在 C0 (beta 到顶) 后计数: 到顶后至少再跑 10 个外层步; 悬臂梁为 3,
                              # 门式框架中部灰度区演化慢, 3 步时平均变化判据在其未定型时即满足
INNER_STOP_RULE = "legacy"
INNER_RELATIVE_TOLERANCE = 0.1
INNER_ABSOLUTE_TOLERANCE = 1.0e-6
MU_0 = 50.0                  # 与悬臂梁同值; PolyStress Table 3 的 10 使连续化前期约束严重违反,
                             # beta 到顶时拓扑未成形即被锁定 (体积 0.345, 乘子触顶)
MU_MAX = 1.0e4
ALPHA = 1.1
LAMBDA_0 = 0.0
LAMBDA_MAX = 3000.0          # 乘子安全阈 (safeguarded AL)
MOVE_LIMIT = 0.15
ASYMPTOTE_MIN_DISTANCE = 1.0e-4
ACCEPTANCE_SOLID_THRESHOLD = 0.5  # C2 只在 rho_phys >= 0.5 的未豁免单元上验收

# 运行目录名模板 (填 method, order)
RUN_LABEL = "analyzer-{method}__order-{order}"
# density_final.vtu 与逐步帧写出的单元场: 'density' 物理密度, 'von_mises_normalized' 单元最大应力比;
# density_initial.vtu (第 0 帧的副本) 只有 'density'
VTU_FIELDS = ("density", "von_mises_normalized")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析要跑的分析链与阶次, 并据此确定运行目录名与输出目录; 其余设置均为模块顶部常量."""
    parser = argparse.ArgumentParser(description="门式框架局部应力约束体积最小化 (论文 5.2.4 节)")
    parser.add_argument("--analyzer", choices=METHODS, required=True, help="分析链")
    parser.add_argument("--order", type=int, choices=ORDERS, required=True,
                        help="阶次 k (LFEM 位移阶 p = k, HZMFEM 应力阶 k)")
    args = parser.parse_args(argv)
    if args.analyzer == "huzhang" and args.order < 2:
        parser.error("HZMFEM 要求 --order >= 2")
    args.label = RUN_LABEL.format(method=args.analyzer, order=args.order)
    args.output = OUTPUT_DIR / args.label
    return args


def format_run_header(config: dict[str, Any], output: Path, dofs: str) -> str:
    """把本组的关键设置排成启动时打印的摘要.

    Parameters
    ----------
    config : dict
        ``main`` 第 0 步构造的完整设置, 与落盘的 ``config.json`` 同源.
    output : Path
        运行目录.
    dofs : str
        自由度规模的描述, 如 ``'285,818 自由度'``; 第 1 步组装分析链后才知道.

    Returns
    -------
    header : str
        首行为运行名与运行目录, 其后按模型、离散、约束、优化分行; 完整设置 (含代码版本戳记) 以
        ``config.json`` 为准.
    """
    model, stress = config["model"], config["stress_constraint"]
    disc, opt = config["discretization"], config["optimization"]
    proj = opt["projection"]
    if config["method"] == "huzhang":
        chain = f"HZMFEM k={config['order']}"
        variant = f"角点松弛={'开' if disc['use_relaxation'] else '关'}"
    else:
        chain = f"LFEM p={config['order']}"
        variant = f"装配={disc['assembly_method']}"
    return "\n".join([
        f"[run] {config['label']} -> {output}",
        f"  模型  {model['L']:g} x {model['H']:g} mm, {model['plane_type']}, E0={model['E0']:g}, "
        f"nu={model['nu']:g}, P={model['load']:g} N 均布于 l={model['load_width']:g} mm, "
        f"sigma_lim={model['stress_limit']:g}",
        f"  离散  {chain}, gmsh 镜像三角 h={disc['mesh_size']:g}, {dofs}, 积分 {disc['integration_order']} 阶, "
        f"{variant}, {disc['solver']}",
        f"  约束  epsilon={stress['epsilon']:g}, 被动实体区 r: 载荷端点 {stress['load_pad_radius']:g} / "
        f"支座端点 {stress['support_pad_radius']:g}",
        f"  优化  AL-MMA, rho0={opt['initial_density']:g}, 过滤 r={opt['filter_radius']:g}, "
        f"MSIMP p={opt['penalty']:g}, {proj['projection_type']} beta {proj['beta']:g}->{proj['beta_max']:g} "
        f"(每 {proj['continuation_iter']} 外层步 +{proj['beta_increment']:g}), "
        f"外层 <= {opt['max_al_iterations']} x 内层 {opt['mma_iters_per_al']}, mu0={opt['mu_0']:g}",
    ])


def main(argv: list[str] | None = None) -> int:
    """单组: 记录配置 -> 组装 -> 投影过滤 + AL-MMA 优化 -> 终态求解与验收 -> 落盘."""
    args = parse_args(argv)
    bm.set_backend("numpy")

    # 记下当前代码的 git 版本, 用于追溯结果由哪版代码算出
    stamp = provenance.run_stamp()

    method, order, label, output = args.analyzer, args.order, args.label, args.output

    # 0. 开跑前记录本组的全部设置, 中途中断也能复原运行条件; summary.json 中的同名参数保留
    config = {
        "case_id": CASE_ID, "method": method, "order": order, "label": label,
        "model": {"L": L, "H": H, "load_width": LOAD_WIDTH, "support_width": SUPPORT_WIDTH,
                  "corner_height": CORNER_HEIGHT, "load": LOAD, "E0": E0, "nu": NU,
                  "plane_type": PLANE_TYPE, "stress_limit": STRESS_LIMIT},
        "stress_constraint": {"epsilon": EPSILON, "load_pad_radius": LOAD_PAD_RADIUS,
                              "support_pad_radius": SUPPORT_PAD_RADIUS},
        "discretization": {"mesh_type": "triangle-gmsh-mirrored", "mesh_size": MESH_SIZE,
                           "mesh_algorithm": MESH_ALGORITHM,
                           "integration_order": 2 * order + 2, "use_relaxation": USE_RELAXATION,
                           "assembly_method": ASSEMBLY_METHOD, "solver": SOLVER},
        "optimization": {
            "filter_radius": FILTER_RADIUS, "design_symmetry_axis": L / 2.0, "penalty": PENALTY, "emin": EMIN,
            "initial_density": INITIAL_DENSITY, "projection": PROJECTION,
            "max_al_iterations": MAX_AL_ITERATIONS, "mma_iters_per_al": MMA_ITERS_PER_AL,
            "change_tolerance": CHANGE_TOLERANCE, "change_measure": CHANGE_MEASURE,
            "stress_tolerance": STRESS_TOLERANCE, "hold_steps": HOLD_STEPS,
            "inner_stop_rule": INNER_STOP_RULE, "inner_relative_tolerance": INNER_RELATIVE_TOLERANCE,
            "inner_absolute_tolerance": INNER_ABSOLUTE_TOLERANCE,
            "mu_0": MU_0, "mu_max": MU_MAX, "alpha": ALPHA, "lambda_0": LAMBDA_0,
            "lambda_max": LAMBDA_MAX, "move_limit": MOVE_LIMIT,
            "asymptote_min_distance": ASYMPTOTE_MIN_DISTANCE,
            "acceptance_solid_threshold": ACCEPTANCE_SOLID_THRESHOLD,
        },
        "output": {"vtu_fields": list(VTU_FIELDS)},
        "provenance": stamp,
    }
    unknown = set(VTU_FIELDS) - {"density", "von_mises_normalized"}
    if unknown:
        raise ValueError(f"VTU_FIELDS 含未知场 {sorted(unknown)}; 可选 'density', 'von_mises_normalized'")
    if output.exists():
        raise SystemExit(f"运行目录已存在, 拒绝覆盖: {output}; 不再需要时先删除 (要保留先经 sync_results.py 固化)")
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1. 组装分析链: 问题、网格、分析器、被动实体区、应力约束与 AL 目标.
    # 问题: 先按缺省阶跃牵引构造, 只为取载荷区几何; 再把牵引换成它在顶边 P1 迹空间上的
    # L2 投影 (顶边均匀剖分) 重新构造, 几何与边界标记不变
    h = MESH_SIZE
    problem = PortalFramePatch2d(L=L, H=H, load_width=LOAD_WIDTH, b=SUPPORT_WIDTH, h_c=CORNER_HEIGHT, P=LOAD,
                                 E=E0, nu=NU, plane_type=PLANE_TYPE)
    xmin, xmax, _, _ = problem.domain
    traction = project_patch_traction_to_p1_trace(
        line=(xmin, xmax), n_cells=int(round((xmax - xmin) / h)), level=problem.traction_level,
        patch=problem.traction_patch, intensity=problem.traction_intensity, axis=problem.traction_axis)
    problem = PortalFramePatch2d(L=L, H=H, load_width=LOAD_WIDTH, b=SUPPORT_WIDTH, h_c=CORNER_HEIGHT, P=LOAD,
                                 E=E0, nu=NU, plane_type=PLANE_TYPE, traction=traction)

    # 网格: gmsh 在左半域五边形上生成三角形网格, 再镜像到右半, 得到关于 x = L/2 严格对称的全域网格
    x_mid, x_patch = problem.symmetry_axis, problem.traction_patch[0]
    for name, length in (("l/2", x_mid - x_patch), ("L/2 - l/2", x_patch)):
        ratio = length / h
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"h={h} 不能整除顶边分段 {name}={length}; 请取 0.5/k 形式的 h, 如 0.25, 0.125, 0.1.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("portal_half")
        geo = gmsh.model.geo
        points = [geo.addPoint(x, y, 0.0, h) for x, y in
                  ((0.0, 0.0), (problem.b, 0.0), (x_mid, problem.h_c),
                   (x_mid, problem.H), (x_patch, problem.H), (0.0, problem.H))]
        # 边依次为: 支座段, 斜边, 对称面, 顶边载荷区, 顶边其余, 左边
        lines = [geo.addLine(points[i], points[(i + 1) % 6]) for i in range(6)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.mesh.setTransfiniteCurve(lines[3], int(round((x_mid - x_patch) / h)) + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[4], int(round(x_patch / h)) + 1)
        gmsh.option.setNumber("Mesh.Algorithm", MESH_ALGORITHM)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[:, :2]
        types, _, node_tags = gmsh.model.mesh.getElements(2)
        if list(types) != [2]:
            raise RuntimeError(f"期望只有线性三角形 (gmsh type 2), 实际 {list(types)}")
        tri = np.asarray(node_tags[0], dtype=np.int64).reshape(-1, 3)
    finally:
        gmsh.finalize()

    # gmsh 节点标签不保证连续: 映射到 0 基编号并去掉未被单元引用的节点
    index = {int(tag): i for i, tag in enumerate(tags)}
    cell_half = np.vectorize(index.__getitem__)(tri)
    used = np.unique(cell_half)
    remap = -np.ones(coords.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    node_half, cell_half = coords[used], remap[cell_half]

    # 对称面上的节点横坐标钉到精确值, 镜像合并时按精确相等判定
    node_half[np.abs(node_half[:, 0] - x_mid) < 1e-8 * problem.L, 0] = x_mid

    # 半域单元统一为逆时针
    p0, p1, p2 = node_half[cell_half[:, 0]], node_half[cell_half[:, 1]], node_half[cell_half[:, 2]]
    flip = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]) < 0
    cell_half[flip] = cell_half[flip][:, [0, 2, 1]]

    # 镜像: 对称面上的节点被两半共用, 镜像单元翻转顶点顺序以保持逆时针; 左半编号在前, 右半在后
    off_axis = node_half[:, 0] != x_mid
    new_index = np.arange(node_half.shape[0])
    new_index[off_axis] = node_half.shape[0] + np.arange(int(off_axis.sum()))
    node_right = np.column_stack((2.0 * x_mid - node_half[off_axis, 0], node_half[off_axis, 1]))
    mesh: Any = TriangleMesh(bm.array(np.vstack((node_half, node_right))),
                             bm.array(np.vstack((cell_half, new_index[cell_half][:, [0, 2, 1]]))))
    mesh.meshdata = {"domain": list(problem.domain), "mesh_type": "triangle-gmsh-mirrored", "h": h,
                     "symmetry_axis": x_mid}


    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E, poisson_ratio=problem.nu,
        hypothesis=problem.plane_type)
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp",
        options={"penalty_factor": PENALTY, "void_youngs_modulus": EMIN, "target_variables": ["E"]},
    )
    # 被动实体区: 载荷端点与支座角点各自半径内的单元, 半径为 0 即不处理 (本算例两者均为 0);
    # 凹角顶点不设: 优化从凹角应力集中区去料正是本算例的预期结果.
    # 区内单元不施加应力约束, 并经过滤器 (passive_mask) 固定为实体
    load_pad_mask = build_exemption_mask(
        mesh=mesh, centers=problem.traction_patch_endpoints, radius=LOAD_PAD_RADIUS)
    support_pad_mask = build_exemption_mask(
        mesh=mesh, centers=problem.support_corner_points, radius=SUPPORT_PAD_RADIUS)
    pad_mask = bm.logical_or(load_pad_mask, support_pad_mask)

    common: dict[str, Any] = dict(
        disp_mesh=mesh, pde=problem, material=material, space_degree=order,
        integration_order=2 * order + 2, solve_method=SOLVER,
        topopt_algorithm="density_based", interpolation_scheme=interpolation)
    constraint_args: dict[str, Any] = dict(
        stress_limit=STRESS_LIMIT, formulation=EpsilonRelaxedStressFormulation(epsilon=EPSILON),
        exemption_mask=pad_mask, enable_logging=False)
    if method == "lfem":
        analyzer = LagrangeFEMAnalyzer(assembly_method=ASSEMBLY_METHOD, **common)
        constraint = LagrangeStressConstraint(analyzer=analyzer, **constraint_args)
    else:
        analyzer = HuZhangMFEMAnalyzer(use_relaxation=USE_RELAXATION, **common)
        constraint = HuZhangStressConstraint(analyzer=analyzer, **constraint_args)
    # 启动摘要: 自由度规模要等分析链建好才知道, 故在此打印 (config.json 已在第 0 步写出)
    n_u = int(analyzer.tensor_space.number_of_global_dofs())
    if isinstance(analyzer, HuZhangMFEMAnalyzer):
        n_sigma = int(analyzer.huzhang_space.number_of_global_dofs())
        dofs = f"应力 {n_sigma:,} + 位移 {n_u:,} = {n_sigma + n_u:,} 自由度"
    else:
        dofs = f"{n_u:,} 自由度"
    print(format_run_header(config, output, dofs), flush=True)

    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh, displacement_mesh=mesh, relative_density=INITIAL_DENSITY)
    volume_objective = VolumeObjective(analyzer=analyzer, enable_logging=False)
    # AL-MMA 选项; 增广拉格朗日目标与优化器共用同一实例
    options = ALMMMAOptions(
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
    al = AugmentedLagrangianObjective(
        volume_objective=volume_objective, stress_constraint=constraint,
        options=options, enable_logging=False)

    # 2. 投影过滤 (实体保留施加在过滤/投影之后) + AL-MMA 优化
    density_filter = Filter(design_mesh=mesh, filter_type="projection",
                            rmin=FILTER_RADIUS, density_location="element", filter_q=3,
                            projection_params=dict(PROJECTION), passive_mask=pad_mask,
                            symmetry_planes={0: problem.symmetry_axis})
    optimizer = ALMMMAOptimizer(al_objective=al, filter=density_filter,
                                options=options, enable_logging=True)
    density, history = optimizer.optimize(
        design_variable=design_variable, density_distribution=density)

    # 3. 在最终物理密度上重新求解并验收; 优化器判据基于迭代末态
    state = analyzer.solve_state(rho_val=density)
    if isinstance(analyzer, HuZhangMFEMAnalyzer):
        residual = float(analyzer.relative_state_residual())
    else:
        # LFEM 只计非 Dirichlet 自由度; 需要已装配的全局刚度矩阵与稠密载荷向量
        K, F = analyzer.stiffness_matrix, analyzer.force_vector
        if not isinstance(K, (CSRTensor, COOTensor)) or isinstance(F, COOTensor):
            raise TypeError(f"残差核算需要全局稀疏刚度矩阵与稠密载荷, 实为 {type(K).__name__}, {type(F).__name__}")
        r = K.matmul(state["displacement"][:]) - F
        free = ~analyzer.tensor_space.is_boundary_dof(
            threshold=problem.is_dirichlet_boundary(), method="interp")
        residual = float(bm.linalg.norm(r[free])) / max(float(bm.linalg.norm(F[free])), 1.0e-30)
    rho_phys = density[:]
    constraint_values = np.asarray(constraint.fun(rho_phys, state))
    stress_measure = np.asarray(constraint.compute_stress_measure(rho_phys, state))
    relative_violation = np.asarray(constraint.compute_relative_violation(rho_phys, state))
    # 未加权实体应力比: 只报告不判据, 反映实体材料真实的应力水平
    solid_stress_ratio = np.asarray(constraint.compute_solid_stress_ratio(rho_phys, state))
    rho = np.asarray(rho_phys).reshape(-1)
    solid = rho >= ACCEPTANCE_SOLID_THRESHOLD
    per_cell_max = lambda value: np.asarray(value).reshape(len(value), -1).max(axis=1)  # 积分点场取单元最大值
    violation_cell = per_cell_max(relative_violation)
    g_solid = float(violation_cell[solid].max()) if solid.any() else None
    feasible = bool((g_solid if g_solid is not None else float(relative_violation.max()))
                    <= STRESS_TOLERANCE)
    # 被动实体区内的约束值按未豁免口径重算 (约束对象在豁免单元上返回哨兵 -1)
    pad_constraint = np.asarray(constraint.compute_unexempted_constraint(rho_phys, state))
    masked_max = lambda values, mask: float(values[mask].max()) if mask.any() else None
    multiplier = np.asarray(al.lamb)
    change = optimizer.last_multiplier_change
    outer = {key.replace("last_", ""): getattr(optimizer, key, None)
             for key in ("last_change_outer_mean", "last_change_outer_max")}

    summary = {
        "case_id": CASE_ID,
        "model": "PortalFramePatch2d",
        "method": method,
        "order": order,
        "mesh_type": "triangle-gmsh-mirrored",
        "mesh_size": MESH_SIZE,
        "optimizer": "al_mma",
        "load_discretization": "p1_trace_l2_projection",
        "integration_order": 2 * order + 2,
        "stabilization_coefficient": getattr(analyzer, "stabilization_coefficient", None),
        "volume_fraction": float(volume_objective.fun(density)),
        "relative_equilibrium_residual": residual,
        "optimization_iterations": len(history.iter_indices),
        "converged": bool(optimizer.converged and feasible),
        "termination_reason": optimizer.termination_reason,
        "solver": SOLVER,
        "interpolation": {"variables": "E"},
        "provenance": stamp,
        "stress_constraint_formulation": "apparent",
        # 表观应力比用于展示, 约束对象定义的相对超限量 g 用于验收
        "max_constraint": float(constraint_values.max()),
        "max_apparent_stress_ratio": float(stress_measure.max()),
        "max_relative_violation": float(relative_violation.max()),
        "relative_stress_tolerance": STRESS_TOLERANCE,
        "acceptance_solid_threshold": ACCEPTANCE_SOLID_THRESHOLD,
        "max_relative_violation_solid_region": g_solid,
        "relative_stress_feasible": feasible,
        "max_solid_stress_ratio_solid_region": masked_max(solid_stress_ratio, solid),
        # 被动实体区诊断: 两侧分列, 使"被动实体区掩盖了多大的应力"可直接读出
        "load_pad_radius": LOAD_PAD_RADIUS,
        "load_pad_centers": [[float(v) for v in c] for c in problem.traction_patch_endpoints] or None,
        "pad_cells": int(bm.sum(pad_mask)),
        "load_pad_cells": int(bm.sum(load_pad_mask)),
        "max_solid_stress_ratio_constrained": masked_max(solid_stress_ratio, ~pad_mask),
        "max_solid_stress_ratio_pad": masked_max(solid_stress_ratio, load_pad_mask),
        "max_constraint_pad": masked_max(pad_constraint, load_pad_mask),
        # ALM 收敛诊断: 乘子相对变化只有优化器能算; mu 与最大乘子的逐步值见 history.json
        "complementarity_residual": float(np.abs(multiplier * constraint_values).max()),
        "multiplier_relative_change": (None if change is None or not np.isfinite(change)
                                       else float(change)),
        **{key: (None if value is None or not np.isfinite(value) else float(value))
           for key, value in outer.items()},
        "lambda_max": LAMBDA_MAX,
        "multiplier_capped_count": int(getattr(al, "last_capped_count", 0)),
    }

    # 4. AL 终态 (供冻结复算与内层诊断, 不支持精确断点续算)
    design = optimizer.final_design_variable
    state_arrays = {"design": np.asarray(design[:]), "density": rho,
                    "lamb": multiplier, "mu": np.asarray(float(al.mu))}
    if any(not np.all(np.isfinite(value)) for value in state_arrays.values()):
        raise ValueError("AL 终态包含非有限值, 不能作为诊断状态保存.")
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

    # 5. 落盘 (全部写 output, 不入库, 确认后经 sync_results.py 固化到 results/):
    #    初始与最终构型 vtu、AL 终态 npz、终态位移/应力 npy、四份 json; 两类 vtu 只写 VTU_FIELDS 中的单元场.
    #    初始构型是优化器记录的优化前物理密度 (过滤与实体保留处理之后), 只带密度; 没记录即报错,
    #    不静默缺文件
    if history.initial_physical_density is None:
        raise RuntimeError("优化器没有记录初始物理密度, 无法写出 density_initial.vtu.")
    initial_cells = {"density": np.asarray(history.initial_physical_density[:]).reshape(-1)}
    write_vtu(mesh=mesh, filepath=str(output / "density_initial"), cell_data=initial_cells)
    final_fields = {"density": rho, "von_mises_normalized": per_cell_max(stress_measure)}
    write_vtu(mesh=mesh, filepath=str(output / "density_final"),
              cell_data={name: final_fields[name] for name in VTU_FIELDS})
    np.savez_compressed(output / "final_optimizer_state.npz", **state_arrays)
    # 终态求解 (第 3 步) 的位移自由度向量; Hu--Zhang 另存应力自由度向量 (松弛坐标, 按 huzhang_space.value 取值)
    np.save(output / "displacement_final.npy", np.asarray(state["displacement"][:]))
    if "stress" in state:
        np.save(output / "stress_final.npy", np.asarray(state["stress"][:]))
    (output / "outer_history.json").write_text(
        json.dumps(getattr(optimizer, "outer_history", []), ensure_ascii=False, indent=2),
        encoding="utf-8")
    payload = {"iter_indices": history.iter_indices, "changes": history.changes,
               "iteration_times": history.iteration_times,
               "scalar_histories": history.scalar_histories}
    (output / "history.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. 逐步帧: 第 0 帧是初始构型, 其后每帧对应一次迭代 (AL-MMA 无逐步回调, 帧由优化器记录的
    #    历史在结束后一次写出); IterationFrameWriter 写 iterations/iter_NNNN.vtu 与 evolution.pvd
    #    到运行目录, 留在原处, 需要时手动迁移到 Windows 查看目录
    frames = IterationFrameWriter(output, mesh)
    frames.write(0, cell_data=initial_cells)
    stress_frames = (getattr(history, "field_histories", None) or {}).get("von_mises_stress") or []
    for index, rho_i in enumerate(history.physical_densities, start=1):
        cells = {"density": np.asarray(rho_i).reshape(-1)}
        if len(stress_frames) >= index:
            cells["von_mises_normalized"] = per_cell_max(stress_frames[index - 1])
        frames.write(index, cell_data={name: value for name, value in cells.items() if name in VTU_FIELDS})
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {method} k={order}: {summary['optimization_iterations']} 步, "
          f"volfrac={summary['volume_fraction']:.6f}, g_max(实体)={g_solid}, "
          f"converged={summary['converged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
