"""投稿论文全部算例的组装层: 共享原语 + 各物理问题的构造与校验.

一个文件承载三件事, 与 ``experiments/topopt_simp_ea/pipeline.py`` 同构:

1. **共享组装原语** —— 网格、材料、材料插值、分析器、过滤器的构造, 保证受控比较
   协议 (LFEM ``p=k`` 对 Hu--Zhang 应力阶 ``k``, 统一积分阶 ``q=2k+2``) 在所有
   算例中完全一致;
2. **三族算例的装配器** —— 两端固支梁 (柔顺度)、轴承装置 (近不可压)、悬臂梁
   (局部应力约束). ``cases.toml`` 只存数值, 这里存「数值 -> soptx 对象」的构造:
   各物理类的构造签名不同, 跨字段校验 TOML 表达不了, 且局部牵引的 P1 迹投影与
   AL-MMA 选项本就不是参数;
3. **装配器注册表** —— ``ASSEMBLERS`` 按 ``cases.toml`` 的模型名分派, 供
   ``driver.py`` 与 ``metrics.py`` 取用, 调用方不必知道模块内部布局.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    LagrangeFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
    create_huzhang_symmetric_single_diagonal_mesh,
    project_patch_traction_to_p1_trace,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    BearingDevice2d,
    CantileverMiddle2d,
    FixedFixedBeamCenterLoad2d,
    FixedFixedBeamHalfDomain2d,
)
from soptx.topology.constraints import (
    HuZhangStressConstraint,
    LagrangeStressConstraint,
    build_exemption_mask,
    EpsilonRelaxedStressFormulation,
    PolynomialVanishingStressFormulation,
    VolumeConstraint,
)
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import (
    AugmentedLagrangianObjective,
    ComplianceObjective,
    VolumeObjective,
)
from soptx.topology.optimizers import (
    ALMMMAOptimizer,
    ALMMMAOptions,
    MMAOptimizer,
    OCOptimizer,
)


# ============================================================ 零、共享组装原语
MethodName = Literal["lfem", "huzhang"]
OptimizerName = Literal["oc", "mma"]
InterpolationMethod = Literal["simp", "msimp", "ramp"]
FilterType = Literal["density", "sensitivity", "projection"]
SolveMethod = Literal["mumps", "scipy"]
InterpolationVariables = Literal["auto", "E", "E+nu"]
MeshType = Literal[
    "triangle-checkerboard",
    "triangle-single-diagonal-symmetric",
]

INTERPOLATION_METHODS = ("simp", "msimp", "ramp")
FILTER_TYPES = ("density", "sensitivity", "projection")
SOLVE_METHODS = ("mumps", "scipy")
COMPLIANCE_OPTIMIZERS = ("oc", "mma")
# 材料插值对象: auto = 近不可压缩材料 (nu >= 0.49) 取 E+nu, 否则取 E; 显式给出时按给定值
INTERPOLATION_VARIABLES = ("auto", "E", "E+nu")
# 网格剖分: checkerboard = 棋盘格交替对角 (nx, ny 须为偶数, 内部结点 4/8 三角形交替);
# single-diagonal-symmetric = 左半 "/" 右半 "\" + 两个顶角落翻转 (nx 须为偶数; 内部结点
# 一律 6 三角形, 低阶位移元体积闭锁的经典构型, 且与左右对称问题同对称性).
# 两者四个几何角点都满足 Hu--Zhang 角点松弛的拓扑要求.
MESH_TYPES = (
    "triangle-checkerboard",
    "triangle-single-diagonal-symmetric",
)
_MESH_BUILDERS = {
    "triangle-checkerboard": create_huzhang_checkerboard_mesh,
    "triangle-single-diagonal-symmetric": create_huzhang_symmetric_single_diagonal_mesh,
}
# 各剖分对 nx / ny 奇偶性的要求 (角点落在 2 单元角上 / 镜像中缝落在网格线上)
MESH_EVEN_AXES = {
    "triangle-checkerboard": ("nx", "ny"),
    "triangle-single-diagonal-symmetric": ("nx",),
}

# 目前唯一支持的共同载荷离散方式, 见各算例模块的 build_problem
P1_TRACE_L2_PROJECTION = "p1_trace_l2_projection"


def validate_choice(value: str, allowed: tuple[str, ...], message: str) -> str:
    """校验枚举型参数取值."""
    if value not in allowed:
        raise ValueError(f"{message}: {value} 不在 {allowed} 中.")
    return value


def validate_mesh_size(nx: int, ny: int, mesh_type: str = "triangle-checkerboard") -> None:
    """校验角点松弛网格对剖分数的要求 (正数; 奇偶性按剖分方式, 见 MESH_EVEN_AXES)."""
    if nx <= 0 or ny <= 0:
        raise ValueError("网格剖分数必须为正数.")
    sizes = {"nx": nx, "ny": ny}
    odd = [axis for axis in MESH_EVEN_AXES.get(mesh_type, ("nx", "ny")) if sizes[axis] % 2]
    if odd:
        raise ValueError(f"{mesh_type} 剖分要求 {' 和 '.join(odd)} 为偶数.")


def validate_orders(
    comparison_orders: tuple[int, ...],
    allowed: tuple[int, ...] = (1, 2, 3, 4),
) -> None:
    """校验受控比较阶次集合."""
    if not comparison_orders:
        raise ValueError("比较阶次不能为空.")
    if any(order not in allowed for order in comparison_orders):
        allowed_text = ", ".join(str(k) for k in allowed)
        raise ValueError(f"Hu--Zhang 投稿算例仅允许 k={allowed_text}.")


def create_mesh(
    problem: Any,
    nx: int,
    ny: int,
    mesh_type: str = "triangle-checkerboard",
) -> Any:
    """构造两种离散共享的 Hu--Zhang 兼容三角形网格 (剖分方式见 MESH_TYPES).

    ``meshdata`` 是 soptx 附加在 fealpy 网格对象上的元数据字典 (过滤器矩阵等
    依赖它), fealpy 的网格类并未声明该属性, 因此这里显式放宽为 ``Any``.
    """
    mesh_type = validate_choice(mesh_type, MESH_TYPES, "不支持的网格类型")
    mesh: Any = _MESH_BUILDERS[mesh_type](box=problem.domain, nx=nx, ny=ny)
    xmin, xmax, ymin, ymax = problem.domain
    mesh.meshdata = {
        "domain": list(problem.domain),
        "mesh_type": mesh_type,
        "nx": nx,
        "ny": ny,
        "hx": (xmax - xmin) / nx,
        "hy": (ymax - ymin) / ny,
    }
    return mesh


def build_material(problem: Any) -> IsotropicLinearElasticMaterial:
    """按物理问题的材料参数构造各向同性线弹性材料."""
    return IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )


def resolve_interpolation_variables(
    material: IsotropicLinearElasticMaterial, interpolation_variables: str
) -> str:
    """把配置里的插值对象落成实际生效值 ("E" 或 "E+nu").

    ``auto`` 按材料是否近不可压缩 (``nu >= 0.49``) 决定; 显式 ``E+nu`` 用在可压缩
    材料上直接拒绝, 因为 ``MaterialInterpolationScheme`` 只在近不可压缩材料上插值
    Poisson 比, 静默忽略会让目录名与实际计算不一致.
    """
    if interpolation_variables == "auto":
        return "E+nu" if material.is_incompressible else "E"
    if interpolation_variables == "E+nu" and not material.is_incompressible:
        raise ValueError(
            f"interpolation_variables=E+nu 要求近不可压缩材料 (nu >= 0.49), "
            f"当前 nu={material.poisson_ratio:g}; 可压缩材料请用 E 或 auto."
        )
    return interpolation_variables


def build_interpolation(
    material: IsotropicLinearElasticMaterial,
    *,
    interpolation_method: str,
    penalty_factor: float,
    void_youngs_modulus: float,
    interpolation_variables: str = "auto",
    nu_penalty_factor: float = 1.0,
    void_poisson_ratio: float = 0.3,
) -> MaterialInterpolationScheme:
    """构造材料插值格式.

    插值对象由 ``interpolation_variables`` 决定 (见 resolve_interpolation_variables):
    ``E+nu`` 额外对 Poisson 比插值, 使空区域退化为可压缩弱材料, 避免空单元的体积
    锁定污染实体区域的应力; ``E`` 只插值 Young 模量, Poisson 比固定为实体值.
    """
    variables = resolve_interpolation_variables(material, interpolation_variables)
    options: dict[str, Any] = {
        "penalty_factor": penalty_factor,
        "void_youngs_modulus": void_youngs_modulus,
        "target_variables": ["E"],
    }
    if variables == "E+nu":
        options["target_variables"] = ["E", "nu"]
        options["nu_penalty_factor"] = nu_penalty_factor
        options["void_poisson_ratio"] = void_poisson_ratio
    return MaterialInterpolationScheme(
        density_location="element",
        interpolation_method=interpolation_method,
        options=options,
        enable_logging=False,
    )


def build_analyzer(
    *,
    method: MethodName,
    mesh: Any,
    problem: Any,
    material: IsotropicLinearElasticMaterial,
    order: int,
    solve_method: str,
    use_relaxation: bool,
    interpolation: MaterialInterpolationScheme,
) -> tuple[Any, str]:
    """按受控比较协议组装 LFEM 或 Hu--Zhang 分析器, 并返回其状态变量名.

    对同一 ``order=k``, LFEM 使用位移阶 ``p=k``, Hu--Zhang 使用应力阶 ``k``
    (对应位移阶 ``k-1``), 二者统一使用积分阶 ``q=2k+2``.
    """
    integration_order = 2 * order + 2
    if method == "lfem":
        lfem_analyzer = LagrangeFEMAnalyzer(
            disp_mesh=mesh,
            pde=problem,
            material=material,
            space_degree=order,
            integration_order=integration_order,
            assembly_method="standard",
            solve_method=solve_method,
            topopt_algorithm="density_based",
            interpolation_scheme=interpolation,
        )
        return lfem_analyzer, "u"

    huzhang_analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=order,
        integration_order=integration_order,
        use_relaxation=use_relaxation,
        solve_method=solve_method,
        topopt_algorithm="density_based",
        interpolation_scheme=interpolation,
    )
    return huzhang_analyzer, "sigma"


def build_density_filter(
    mesh: Any,
    *,
    filter_type: str,
    filter_radius: float,
    projection_params: Optional[dict[str, Any]] = None,
    passive_mask: Optional[Any] = None,
) -> Filter:
    """构造单元密度过滤器."""
    return Filter(
        design_mesh=mesh,
        filter_type=filter_type,
        rmin=filter_radius,
        density_location="element",
        # 非结构网格走 KD-tree 通用路径, 权重为 (1 - d/rmin)^q。q 曾在
        # FilterMatrixBuilder 内部写死为 3, 现已参数化 (默认 1 = 线性锥形);
        # 这里显式钉住 3, 保持既有结果不变。
        filter_q=3,
        projection_params=projection_params,
        passive_mask=passive_mask,
        enable_logging=False,
    )


@dataclass
class OptimizationPipeline:
    """保存一次柔顺度优化所需的已组装对象."""

    method: MethodName
    order: int
    problem: Any
    mesh: Any
    analyzer: Any
    design_variable: Any
    density_distribution: Any
    objective: ComplianceObjective
    constraint: VolumeConstraint
    optimizer: OCOptimizer | MMAOptimizer | None


def build_compliance_pipeline(
    problem: Any,
    config: Any,
    method: MethodName,
    order: int,
) -> OptimizationPipeline:
    """组装柔顺度目标 + 体积约束的分析链 (不含优化器)."""
    if order not in (*config.comparison_orders, *config.supplementary_orders):
        raise ValueError(f"比较阶次 {order} 不在投稿配置中.")

    mesh = create_mesh(problem, config.nx, config.ny, config.mesh_type)
    material = build_material(problem)
    interpolation = build_interpolation(
        material,
        interpolation_method=config.interpolation_method,
        penalty_factor=config.penalty_factor,
        void_youngs_modulus=config.void_youngs_modulus,
        interpolation_variables=config.interpolation_variables,
        nu_penalty_factor=config.nu_penalty_factor,
        void_poisson_ratio=config.void_poisson_ratio,
    )
    analyzer, state_variable = build_analyzer(
        method=method,
        mesh=mesh,
        problem=problem,
        material=material,
        order=order,
        solve_method=config.solve_method,
        use_relaxation=config.use_relaxation,
        interpolation=interpolation,
    )
    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh,
        displacement_mesh=mesh,
        relative_density=config.volume_fraction,
    )
    return OptimizationPipeline(
        method=method,
        order=order,
        problem=problem,
        mesh=mesh,
        analyzer=analyzer,
        design_variable=design_variable,
        density_distribution=density,
        objective=ComplianceObjective(
            analyzer=analyzer,
            state_variable=state_variable,
            diff_mode="manual",
            enable_logging=False,
        ),
        constraint=VolumeConstraint(
            analyzer=analyzer,
            volume_fraction=config.volume_fraction,
            diff_mode="manual",
            enable_logging=False,
        ),
        optimizer=None,
    )


def attach_compliance_optimizer(
    pipeline: OptimizationPipeline,
    config: Any,
) -> OptimizationPipeline:
    """在分析链上挂载 OC 或 MMA 优化器与密度过滤器."""
    density_filter = build_density_filter(
        pipeline.mesh,
        filter_type=config.filter_type,
        filter_radius=config.filter_radius,
    )
    options = {
        "max_iterations": config.max_iterations,
        "change_tolerance": config.change_tolerance,
    }
    if config.optimizer == "oc":
        optimizer: OCOptimizer | MMAOptimizer = OCOptimizer(
            objective=pipeline.objective,
            constraint=pipeline.constraint,
            filter=density_filter,
            options=options,
            enable_logging=True,
        )
        optimizer.options.set_advanced_options(
            move_limit=config.move_limit,
            damping_coef=0.5,
            initial_lambda=1.0e9,
            bisection_tol=1.0e-3,
        )
    else:
        optimizer = MMAOptimizer(
            objective=pipeline.objective,
            constraint=pipeline.constraint,
            filter=density_filter,
            options=options,
            enable_logging=True,
        )
        # 步长参数由 config 决定: move_limit 限制单步位移, asymp_init 定初始渐近线
        # 距离. 两者共同决定前几步的激进程度, 是 MMA 与 OC 路径分叉的主因.
        optimizer.options.move_limit = config.move_limit
        optimizer.options.asymp_init = config.asymp_init
    pipeline.optimizer = optimizer
    return pipeline


def compliance_config_fields(parameters: dict[str, Any]) -> dict[str, Any]:
    """解析两族柔顺度算例共有的 23 个配置字段 (17 个必填 + 2 个步长可选项 + 3 个插值对象可选项 + 网格类型).

    ``fixed_fixed_beam`` 与 ``bearing_device`` 的实验配置只在载荷描述上分叉
    (前者 ``load``/``load_width``/``load_discretization``, 后者 ``traction``),
    其余字段的解析与合法值校验逐字相同, 故统一在此处理.
    """
    return {
        "nx": int(parameters["nx"]),
        "ny": int(parameters["ny"]),
        "volume_fraction": float(parameters["volume_fraction"]),
        "filter_radius": float(parameters["filter_radius"]),
        "plane_type": str(parameters["plane_type"]),
        "youngs_modulus": float(parameters["youngs_modulus"]),
        "poisson_ratio": float(parameters["poisson_ratio"]),
        "comparison_orders": tuple(int(value) for value in parameters["comparison_orders"]),
        "supplementary_orders": tuple(int(value) for value in parameters.get("supplementary_orders", ())),
        "interpolation_method": validate_choice(
            str(parameters["interpolation_method"]), INTERPOLATION_METHODS, "不支持的材料插值方法"
        ),
        "penalty_factor": float(parameters["penalty_factor"]),
        "void_youngs_modulus": float(parameters["void_youngs_modulus"]),
        "filter_type": validate_choice(
            str(parameters["filter_type"]), FILTER_TYPES, "不支持的过滤器类型"
        ),
        "max_iterations": int(parameters["max_iterations"]),
        "change_tolerance": float(parameters["change_tolerance"]),
        "use_relaxation": bool(parameters["use_relaxation"]),
        "solve_method": validate_choice(
            str(parameters["solve_method"]), SOLVE_METHODS, "不支持的直接求解器"
        ),
        "optimizer": validate_choice(
            str(parameters["optimizer"]), COMPLIANCE_OPTIMIZERS, "不支持的优化器"
        ),
        # D 算法: 优化器步长. 不写时取 OC / MMAOptions 的现行默认, 已有运行语义不变;
        # 进 config 是为了能用 --override 扫描并让取值写进目录名与 summary.json.
        "move_limit": float(parameters.get("move_limit", 0.2)),
        "asymp_init": float(parameters.get("asymp_init", 0.5)),
        # C 拓扑建模: 材料插值对象. 不写时 auto = 按材料近不可压缩与否自动决定,
        # 与旧运行语义一致; 显式写 E / E+nu 可在同一材料上对照 Poisson 比插不插值.
        # 后两项只在 E+nu 生效.
        "interpolation_variables": validate_choice(
            str(parameters.get("interpolation_variables", "auto")),
            INTERPOLATION_VARIABLES, "不支持的材料插值对象"
        ),
        "nu_penalty_factor": float(parameters.get("nu_penalty_factor", 1.0)),
        "void_poisson_ratio": float(parameters.get("void_poisson_ratio", 0.3)),
        # B 离散: 三角剖分方式 (见 MESH_TYPES). 不写时取棋盘格 (已有运行语义不变);
        # 单向对角两种供低阶位移元体积闭锁对照, 命令行用 --mesh-type 切换.
        "mesh_type": validate_choice(
            str(parameters.get("mesh_type", "triangle-checkerboard")),
            MESH_TYPES, "不支持的网格类型"
        ),
    }


def validate_compliance_config(config: Any, allowed_orders: tuple[int, ...]) -> None:
    """两族柔顺度算例共有的配置校验; 允许阶次集合按算例给定."""
    validate_mesh_size(config.nx, config.ny, getattr(config, "mesh_type", "triangle-checkerboard"))
    validate_orders(config.comparison_orders, allowed_orders)
    if config.supplementary_orders:
        validate_orders(config.supplementary_orders, allowed_orders)
    if not 0.0 < config.volume_fraction <= 1.0:
        raise ValueError("体积分数必须位于 (0, 1] 区间.")
    if config.filter_radius <= 0.0:
        raise ValueError("过滤半径必须为正数.")
    if config.youngs_modulus <= 0.0 or config.void_youngs_modulus <= 0.0:
        raise ValueError("实体和空材料 Young 模量必须为正数.")
    if config.max_iterations <= 0 or config.change_tolerance <= 0.0:
        raise ValueError("迭代次数和变化容限必须为正数.")
    if not 0.0 < config.move_limit <= 1.0:
        raise ValueError("move_limit 必须位于 (0, 1] 区间.")
    if config.asymp_init <= 0.0:
        raise ValueError("asymp_init 必须为正数.")
    if config.nu_penalty_factor <= 0.0:
        raise ValueError("nu_penalty_factor 必须为正数.")
    if not 0.0 <= config.void_poisson_ratio < 0.5:
        raise ValueError("void_poisson_ratio 必须位于 [0, 0.5) 区间.")


# ============================================================ 一、两端固支梁 (柔顺度)


@dataclass(frozen=True)
class FixedFixedBeamExperimentConfig:
    """投稿论文两端固支梁 (完整域或左半域对称降维) 的实验参数."""

    # 前 17 个字段由 compliance_config_fields 统一解析, 顺序与之对齐
    nx: int
    ny: int
    volume_fraction: float
    filter_radius: float
    plane_type: str
    youngs_modulus: float
    poisson_ratio: float
    comparison_orders: tuple[int, ...]
    interpolation_method: InterpolationMethod
    penalty_factor: float
    void_youngs_modulus: float
    filter_type: FilterType
    max_iterations: int
    change_tolerance: float
    use_relaxation: bool
    solve_method: SolveMethod
    optimizer: OptimizerName
    # 本族特有: 集中载荷的大小、载荷区宽度与离散方式
    load: float
    load_width: float
    load_discretization: str
    # 优化器步长 (OC 与 MMA 共用 move_limit, asymp_init 仅 MMA 读取); 带默认值故置尾
    move_limit: float = 0.2
    asymp_init: float = 0.5
    # 材料插值对象 (auto / E / E+nu) 与 Poisson 比插值参数 (仅 E+nu 生效)
    interpolation_variables: InterpolationVariables = "auto"
    nu_penalty_factor: float = 1.0
    void_poisson_ratio: float = 0.3
    # 三角剖分方式, 取值见 MESH_TYPES
    mesh_type: MeshType = "triangle-checkerboard"
    # 补充专题阶次: 只放宽 --order 白名单, 不进缺省也不进 --full (与 config.resolve_runs 同口径)
    supplementary_orders: tuple[int, ...] = ()


# 模型名到物理问题类的映射; 未注册模型在 assembler_for 层被拒绝
_FIXED_FIXED_CLASSES: dict[str, type] = {
    "FixedFixedBeamCenterLoad2d": FixedFixedBeamCenterLoad2d,
    "FixedFixedBeamHalfDomain2d": FixedFixedBeamHalfDomain2d,
}

# 两个模型接口兼容但没有共同基类, 用联合类型标注共享的物理问题对象
FixedFixedBeamProblem = FixedFixedBeamCenterLoad2d | FixedFixedBeamHalfDomain2d


def _instantiate_fixed_fixed(
    parameters: dict[str, Any],
    model_name: str = "FixedFixedBeamCenterLoad2d",
    traction: Optional[Callable[[TensorLike], TensorLike]] = None,
) -> FixedFixedBeamProblem:
    """按 case 参数实例化物理问题, 可选地替换牵引函数."""
    model_class = _FIXED_FIXED_CLASSES.get(model_name)
    if model_class is None:
        raise ValueError(f"未注册的固定梁物理模型: {model_name}.")
    return model_class(
        P=float(parameters["load"]),
        E=float(parameters["youngs_modulus"]),
        nu=float(parameters["poisson_ratio"]),
        load_width=float(parameters["load_width"]),
        plane_type=str(parameters["plane_type"]),
        traction=traction,
        point_force=parameters.get("load_discretization") == "point_force",
    )


def build_fixed_fixed_problem(
    parameters: dict[str, Any],
    model_name: str = "FixedFixedBeamCenterLoad2d",
    *,
    n_cells: Optional[int] = None,
) -> FixedFixedBeamProblem:
    """从 case 参数构造指定的两端固支梁物理问题.

    ``n_cells`` 缺省时返回原始物理问题, 其局部牵引在载荷区边缘不连续.
    给定 ``n_cells`` (底边单元数) 时, 把该牵引替换为它在底边连续 P1 迹
    空间上的 L2 投影: 合力被精确保持, 而 LFEM 的边界数值积分与 Hu--Zhang
    的迹插值都能精确重现它, 于是两条离散路径看到同一个载荷泛函. 受控比较
    必须走这条分支.

    投影只依赖计算域、底边剖分数和载荷区几何, 不需要网格对象; 载荷区几何一律
    从物理问题的 ``traction_patch``/``traction_level``/``traction_intensity``
    读取, 不在此处复述. 左半域模型 ``FixedFixedBeamHalfDomain2d`` 的载荷区
    位于对称面底端, 投影合力自动为完整域的一半 ``P/2``.
    """
    problem = _instantiate_fixed_fixed(parameters, model_name)
    if n_cells is None or parameters.get("load_discretization") == "point_force":
        return problem

    discretization = str(parameters["load_discretization"])
    if discretization != P1_TRACE_L2_PROJECTION:
        raise ValueError(f"不支持的共同载荷离散方式: {discretization}.")

    xmin, xmax = problem.domain[0], problem.domain[1]
    common_load = project_patch_traction_to_p1_trace(
        line=(xmin, xmax),
        n_cells=n_cells,
        level=problem.traction_level,
        patch=problem.traction_patch,
        intensity=problem.traction_intensity,
    )
    return _instantiate_fixed_fixed(parameters, model_name, traction=common_load)


def build_fixed_fixed_config(parameters: dict[str, Any]) -> FixedFixedBeamExperimentConfig:
    """校验并返回实验配置."""
    config = FixedFixedBeamExperimentConfig(
        **compliance_config_fields(parameters),
        load=float(parameters["load"]),
        load_width=float(parameters["load_width"]),
        load_discretization=str(parameters["load_discretization"]),
    )
    if config.load_discretization not in (P1_TRACE_L2_PROJECTION, "point_force"):
        raise ValueError(f"不支持的载荷离散方式: {config.load_discretization}.")
    validate_compliance_config(config, (1, 2, 3, 4))
    if config.load_width <= 0.0:
        raise ValueError("载荷宽度必须为正数.")
    return config


def build_fixed_fixed_analysis_pipeline(
    config: FixedFixedBeamExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "FixedFixedBeamCenterLoad2d",
) -> OptimizationPipeline:
    """按受控比较协议组装一条 LFEM 或 Hu--Zhang 分析链."""
    if config.load_discretization == "point_force" and method != "lfem":
        raise ValueError("point_force 仅支持 LFEM, 请指定 --analyzer lfem.")
    # 覆盖后的物理参数必须传入问题工厂, 不能继续使用注册表原值.
    parameters = {**parameters, **vars(config)}
    # 底边单元数即 nx, 交叉网格的底边界正好有 nx 条边
    problem = build_fixed_fixed_problem(parameters, model_name, n_cells=config.nx)
    return build_compliance_pipeline(problem, config, method, order)


def build_fixed_fixed_pipeline(
    config: FixedFixedBeamExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "FixedFixedBeamCenterLoad2d",
) -> OptimizationPipeline:
    """在单次状态分析链上增加优化器 (OC 或 MMA) 与过滤器."""
    pipeline = build_fixed_fixed_analysis_pipeline(config, parameters, method, order, model_name)
    return attach_compliance_optimizer(pipeline, config)


# ============================================================ 二、轴承装置 (近不可压)


@dataclass(frozen=True)
class BearingDeviceExperimentConfig:
    """投稿论文二维轴承装置的实验参数."""

    # 前 17 个字段由 compliance_config_fields 统一解析, 顺序与之对齐
    nx: int
    ny: int
    volume_fraction: float
    filter_radius: float
    plane_type: str
    youngs_modulus: float
    poisson_ratio: float
    comparison_orders: tuple[int, ...]
    interpolation_method: InterpolationMethod
    penalty_factor: float
    void_youngs_modulus: float
    filter_type: FilterType
    max_iterations: int
    change_tolerance: float
    use_relaxation: bool
    solve_method: SolveMethod
    optimizer: OptimizerName
    # 本族特有: 边界牵引强度
    traction: float
    # 优化器步长 (OC 与 MMA 共用 move_limit, asymp_init 仅 MMA 读取); 带默认值故置尾
    move_limit: float = 0.2
    asymp_init: float = 0.5
    # 材料插值对象 (auto / E / E+nu) 与 Poisson 比插值参数 (仅 E+nu 生效)
    interpolation_variables: InterpolationVariables = "auto"
    nu_penalty_factor: float = 1.0
    void_poisson_ratio: float = 0.3
    # 三角剖分方式, 取值见 MESH_TYPES
    mesh_type: MeshType = "triangle-checkerboard"
    # 补充专题阶次: 只放宽 --order 白名单, 不进缺省也不进 --full (与 config.resolve_runs 同口径)
    supplementary_orders: tuple[int, ...] = ()


def _traction_value(parameters: dict[str, Any]) -> float:
    """读取牵引强度, 兼容 traction 与 load 两种键名."""
    return float(parameters.get("traction", parameters.get("load", -8.0e-2)))


def build_bearing_problem(
    parameters: dict[str, Any],
    model_name: str = "BearingDevice2d",
) -> BearingDevice2d:
    """从 case 参数构造指定的轴承装置物理问题."""
    if model_name != "BearingDevice2d":
        raise ValueError(f"未注册的轴承装置物理模型: {model_name}.")
    return BearingDevice2d(
        t=_traction_value(parameters),
        E=float(parameters["youngs_modulus"]),
        nu=float(parameters["poisson_ratio"]),
        plane_type=str(parameters.get("plane_type", "plane_stress")),
    )


def build_bearing_config(parameters: dict[str, Any]) -> BearingDeviceExperimentConfig:
    """校验并返回实验配置."""
    config = BearingDeviceExperimentConfig(
        **compliance_config_fields(parameters),
        traction=_traction_value(parameters),
    )
    validate_compliance_config(config, (1, 2, 3, 4))
    return config


def build_bearing_analysis_pipeline(
    config: BearingDeviceExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "BearingDevice2d",
) -> OptimizationPipeline:
    """按受控比较协议组装一条 LFEM 或 Hu--Zhang 分析链.

    材料插值对象由 ``config.interpolation_variables`` 决定 (见 build_interpolation);
    近不可压缩算例 (``nu = 0.4999``) 在注册表里显式登记为 E+nu 双参数插值.
    """
    problem = build_bearing_problem(parameters, model_name)
    return build_compliance_pipeline(problem, config, method, order)


def build_bearing_pipeline(
    config: BearingDeviceExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "BearingDevice2d",
) -> OptimizationPipeline:
    """在单次状态分析链上增加优化器与过滤器."""
    pipeline = build_bearing_analysis_pipeline(config, parameters, method, order, model_name)
    return attach_compliance_optimizer(pipeline, config)


# ============================================================ 三、悬臂梁 (局部应力约束)


StressOptimizerName = Literal["al_mma"]
StressConstraintFormulation = Literal["apparent", "vanishing"]
STRESS_CONSTRAINT_FORMULATIONS = ("apparent", "vanishing")


@dataclass(frozen=True)
class CantileverStressExperimentConfig:
    """投稿论文二维悬臂梁局部应力约束优化的实验参数."""

    nx: int
    ny: int
    # 三角剖分方式 (见 MESH_TYPES); 此前该字段被静默丢弃, create_mesh 的默认值
    # 恰为棋盘格, 补齐后行为不变, 但 [mesh] 头部与 summary 能如实报告。
    mesh_type: MeshType
    filter_radius: float
    load_width: float
    load_discretization: str
    plane_type: str
    load: float
    youngs_modulus: float
    poisson_ratio: float
    stress_limit: float
    epsilon: float
    # 载荷引入垫片半径 (mm), 0 表示无垫片 (历史行为). 该邻域同时做两件事:
    # 物理密度钉为 1 (实体保留) 且移出应力约束集合 (豁免). 两者必须成对施加,
    # 只豁免会被优化器用来减料换体积; 按固定物理尺寸定义, 不随网格加密缩小.
    # 掩码构造见 soptx.topology.constraints.exemption.
    load_pad_radius: float
    # 固支角点垫片半径 (mm), 0 表示不处置 (2026-09-16 之前的历史行为). 左端固支边
    # 的两个端点是 Dirichlet--Neumann 混合边界角点, 应力按 r^(lambda-1) 奇异
    # (直角楔, 平面应力 nu=0.25 时 lambda=0.78107, 即 sigma ~ r^(-0.2189)).
    # 与载荷侧的区别: 该奇异性由边界条件类型改变产生, 不能由载荷分布化削弱, 故
    # 豁免 + 实体保留是仅有的两步处置. 半径必须单独标定, 不能沿用 load_pad_radius
    # —— 幂律奇点的污染区比载荷侧的对数型宽, 半径取小了只会把热点搬到掩码边界.
    support_pad_radius: float
    # 历史字段: 指定整组对照中的 LFEM 模型, Hu--Zhang 当前固定采用 apparent.
    stress_constraint_formulation: StressConstraintFormulation
    comparison_orders: tuple[int, ...]
    interpolation_method: InterpolationMethod
    penalty_factor: float
    void_youngs_modulus: float
    filter_type: FilterType
    max_al_iterations: int
    mma_iters_per_al: int
    change_tolerance: float
    stress_tolerance: float
    hold_steps: int
    inner_stop_rule: str
    inner_relative_tolerance: float
    inner_absolute_tolerance: float
    use_relaxation: bool
    solve_method: SolveMethod
    optimizer: StressOptimizerName
    initial_density: float
    mu_0: float
    mu_max: float
    alpha: float
    lambda_0_init_val: float
    move_limit: float
    asymptote_min_distance: float
    move_limit_decay: float
    move_limit_min: float
    move_limit_progress_window: int
    move_limit_progress_ratio: float
    move_limit_progress_cell: float
    change_measure: str
    mu_update_rule: str
    mu_violation_ratio: float
    # 2026-09-18: 乘子安全阈与 C2 实体验收子集, None 均复现旧行为 (无阈 / 全域).
    lambda_max: Optional[float]
    acceptance_solid_threshold: Optional[float]
    kkt_diagnostics_enabled: bool
    kkt_acceptance_enabled: bool
    kkt_stationarity_tolerance: float
    kkt_complementarity_tolerance: float
    kkt_dual_tolerance: float

    @property
    def max_iterations(self) -> int:
        return self.max_al_iterations


@dataclass
class StressOptimizationPipeline:
    """保存一次应力约束优化所需的已组装对象."""

    method: MethodName
    order: int
    problem: CantileverMiddle2d
    mesh: Any
    analyzer: Any
    design_variable: Any
    density_distribution: Any
    volume_objective: VolumeObjective
    stress_constraint: Any
    al_objective: AugmentedLagrangianObjective
    optimizer: ALMMMAOptimizer | None
    # 载荷侧垫片单元掩码 (贴片端点邻域), 无垫片时全 False. 仅供诊断分列.
    load_pad_mask: Any = None
    # 支撑侧垫片单元掩码 (固支角点邻域), 无垫片时全 False. 仅供诊断分列.
    support_pad_mask: Any = None
    # 两侧并集: 实际施加于应力豁免与实体保留的掩码, 是"哪些单元不受考核也不可
    # 设计"的唯一口径, 优化器与过滤器都读它.
    pad_mask: Any = None


def _instantiate_cantilever(
    parameters: dict[str, Any],
    traction: Optional[Callable[[TensorLike], TensorLike]] = None,
) -> CantileverMiddle2d:
    """按 case 参数实例化物理问题, 可选地替换牵引函数."""
    return CantileverMiddle2d(
        P=float(parameters.get("load", -100.0)),
        load_width=float(parameters.get("load_width", 4.0)),
        E=float(parameters.get("youngs_modulus", 70000.0)),
        nu=float(parameters.get("poisson_ratio", 0.25)),
        plane_type=str(parameters.get("plane_type", "plane_stress")),
        traction=traction,
    )


def build_stress_problem(
    parameters: dict[str, Any],
    model_name: str = "CantileverMiddle2d",
    *,
    n_cells: int | None = None,
) -> CantileverMiddle2d:
    """从 case 参数构造指定的悬臂梁物理问题.

    ``load_discretization`` 取 ``p1_trace_l2_projection`` 且给定 ``n_cells``
    (右端面单元数) 时, 把局部牵引替换为它在该边连续 P1 迹空间上的 L2 投影,
    使 LFEM 与 Hu--Zhang 看到同一个载荷泛函; 载荷区几何一律从物理问题的
    ``traction_patch``/``traction_level``/``traction_intensity`` 读取.
    """
    if model_name != "CantileverMiddle2d":
        raise ValueError(f"未注册的悬臂梁物理模型: {model_name}.")

    problem = _instantiate_cantilever(parameters)
    discretization = str(parameters.get("load_discretization", "patch"))
    if discretization != P1_TRACE_L2_PROJECTION or n_cells is None:
        return problem

    ymin, ymax = problem.domain[2], problem.domain[3]
    common_load = project_patch_traction_to_p1_trace(
        line=(ymin, ymax),
        n_cells=n_cells,
        level=problem.traction_level,
        patch=problem.traction_patch,
        intensity=problem.traction_intensity,
    )
    return _instantiate_cantilever(parameters, common_load)


def build_stress_config(parameters: dict[str, Any]) -> CantileverStressExperimentConfig:
    """校验并返回实验配置."""
    stress_constraint_formulation = validate_choice(
        str(parameters.get("stress_constraint_formulation", "apparent")),
        STRESS_CONSTRAINT_FORMULATIONS,
        "未知的应力约束形式",
    )
    config = CantileverStressExperimentConfig(
        nx=int(parameters["nx"]),
        ny=int(parameters["ny"]),
        mesh_type=validate_choice(
            str(parameters.get("mesh_type", "triangle-checkerboard")),
            MESH_TYPES, "不支持的网格类型"
        ),
        filter_radius=float(parameters.get("filter_radius", 2.0)),
        load_width=float(parameters.get("load_width", 4.0)),
        load_discretization=str(parameters.get("load_discretization", "patch")),
        plane_type=str(parameters.get("plane_type", "plane_stress")),
        load=float(parameters.get("load", -100.0)),
        youngs_modulus=float(parameters.get("youngs_modulus", 70000.0)),
        poisson_ratio=float(parameters.get("poisson_ratio", 0.25)),
        stress_limit=float(parameters.get("stress_limit", 180.0)),
        epsilon=float(parameters.get("epsilon", 1.0e-4)),
        load_pad_radius=float(parameters.get("load_pad_radius", 0.0)),
        support_pad_radius=float(parameters.get("support_pad_radius", 0.0)),
        stress_constraint_formulation=stress_constraint_formulation,
        comparison_orders=tuple(int(v) for v in parameters.get("comparison_orders", [2])),
        interpolation_method=str(parameters.get("interpolation_method", "simp")),
        penalty_factor=float(parameters.get("penalty_factor", 3.0)),
        void_youngs_modulus=float(parameters.get("void_youngs_modulus", 1.0e-9)),
        filter_type=str(parameters.get("filter_type", "density")),
        max_al_iterations=int(parameters.get("max_al_iterations", 100)),
        mma_iters_per_al=int(parameters.get("mma_iters_per_al", 5)),
        change_tolerance=float(parameters.get("change_tolerance", 2.0e-3)),
        stress_tolerance=float(parameters.get("stress_tolerance", 3.0e-3)),
        hold_steps=int(parameters.get("hold_steps", 3)),
        inner_stop_rule=str(parameters.get("inner_stop_rule", "legacy")),
        inner_relative_tolerance=float(parameters.get("inner_relative_tolerance", 0.1)),
        inner_absolute_tolerance=float(parameters.get("inner_absolute_tolerance", 1.0e-6)),
        use_relaxation=bool(parameters.get("use_relaxation", True)),
        solve_method=str(parameters.get("solve_method", "mumps")),
        optimizer=str(parameters.get("optimizer", "al_mma")),
        initial_density=float(parameters.get("initial_density", 0.5)),
        mu_0=float(parameters.get("mu_0", 50.0)),
        mu_max=float(parameters.get("mu_max", 10000.0)),
        alpha=float(parameters.get("alpha", 1.1)),
        lambda_0_init_val=float(parameters.get("lambda_0_init_val", 0.0)),
        move_limit=float(parameters.get("move_limit", 0.15)),
        asymptote_min_distance=float(parameters.get("asymptote_min_distance", 1.0e-4)),
        move_limit_decay=float(parameters.get("move_limit_decay", 1.0)),
        move_limit_min=float(parameters.get("move_limit_min", 5.0e-3)),
        move_limit_progress_window=int(parameters.get("move_limit_progress_window", 10)),
        move_limit_progress_ratio=float(parameters.get("move_limit_progress_ratio", 0.3)),
        move_limit_progress_cell=float(parameters.get("move_limit_progress_cell", 0.7)),
        change_measure=str(parameters.get("change_measure", "design")),
        mu_update_rule=str(parameters.get("mu_update_rule", "unconditional")),
        mu_violation_ratio=float(parameters.get("mu_violation_ratio", 0.5)),
        lambda_max=_optional_float(parameters.get("lambda_max", None)),
        acceptance_solid_threshold=_optional_float(
            parameters.get("acceptance_solid_threshold", None)
        ),
        kkt_diagnostics_enabled=bool(parameters.get("kkt_diagnostics_enabled", False)),
        kkt_acceptance_enabled=bool(parameters.get("kkt_acceptance_enabled", False)),
        kkt_stationarity_tolerance=float(parameters.get("kkt_stationarity_tolerance", 0.0)),
        kkt_complementarity_tolerance=float(
            parameters.get("kkt_complementarity_tolerance", 0.0)
        ),
        kkt_dual_tolerance=float(parameters.get("kkt_dual_tolerance", 0.0)),
    )
    validate_mesh_size(config.nx, config.ny, config.mesh_type)
    if not math.isfinite(config.load_pad_radius) or config.load_pad_radius < 0.0:
        raise ValueError("load_pad_radius 必须为有限非负数")
    if not math.isfinite(config.support_pad_radius) or config.support_pad_radius < 0.0:
        raise ValueError("support_pad_radius 必须为有限非负数")
    # check-only 只构造分析管线，也必须验证 KKT 验收开关及容差；复用优化器
    # 选项的唯一校验实现，避免执行模式与配置检查模式给出不同结论.
    _build_al_options(config)
    return config


def _optional_float(value: object) -> Optional[float]:
    """把配置值转成 Optional[float]: None 与文本 none/null 视为 None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return float(value)


def _build_al_options(config: CantileverStressExperimentConfig) -> ALMMMAOptions:
    """按博士论文第五章配方构造 ALM-MMA 选项."""
    return ALMMMAOptions(
        change_tolerance=config.change_tolerance,
        stress_tolerance=config.stress_tolerance,
        hold_steps=config.hold_steps,
        inner_stop_rule=config.inner_stop_rule,
        inner_relative_tolerance=config.inner_relative_tolerance,
        inner_absolute_tolerance=config.inner_absolute_tolerance,
        max_al_iterations=config.max_al_iterations,
        mma_iters_per_al=config.mma_iters_per_al,
        mu_0=config.mu_0,
        mu_max=config.mu_max,
        alpha=config.alpha,
        lambda_0_init_val=config.lambda_0_init_val,
        move_limit=config.move_limit,
        asymptote_min_distance=config.asymptote_min_distance,
        move_limit_decay=config.move_limit_decay,
        move_limit_min=config.move_limit_min,
        move_limit_progress_window=config.move_limit_progress_window,
        move_limit_progress_ratio=config.move_limit_progress_ratio,
        move_limit_progress_cell=config.move_limit_progress_cell,
        change_measure=config.change_measure,
        mu_update_rule=config.mu_update_rule,
        mu_violation_ratio=config.mu_violation_ratio,
        lambda_max=config.lambda_max,
        acceptance_solid_threshold=config.acceptance_solid_threshold,
        kkt_diagnostics_enabled=config.kkt_diagnostics_enabled,
        kkt_acceptance_enabled=config.kkt_acceptance_enabled,
        kkt_stationarity_tolerance=config.kkt_stationarity_tolerance,
        kkt_complementarity_tolerance=config.kkt_complementarity_tolerance,
        kkt_dual_tolerance=config.kkt_dual_tolerance,
    )


def resolve_stress_constraint_formulation(
    config: CantileverStressExperimentConfig,
    method: MethodName,
) -> StressConstraintFormulation:
    """解析当前分析链实际采用的应力约束模型.

    Parameters
    ----------
    config : CantileverStressExperimentConfig
        实验配置. ``stress_constraint_formulation`` 保留为 LFEM 对照协议.
    method : MethodName
        有限元分析方法.

    Returns
    -------
    StressConstraintFormulation
        当前计算链的模型名, 与目录中的 LFEM 协议标签分别记录.

    Notes
    -----
    Hu--Zhang 当前只登记 apparent 模型. 新增模型时须显式扩展本解析器,
    不根据非 LFEM 分支隐式回退.
    """
    formulation = validate_choice(
        config.stress_constraint_formulation,
        STRESS_CONSTRAINT_FORMULATIONS,
        "未知的应力约束形式",
    )
    if method == "lfem":
        return formulation
    if method == "huzhang":
        return "apparent"
    raise ValueError(f"应力约束不支持分析方法 {method!r}.")


def build_stress_analysis_pipeline(
    config: CantileverStressExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "CantileverMiddle2d",
) -> StressOptimizationPipeline:
    """组装悬臂梁应力约束分析求解链.

    默认让两条路径使用同一表观应力松弛约束. Hu--Zhang 直接评价独立应力;
    LFEM 从位移场恢复实体应力后乘相对刚度. 原多项式消失约束由
    ``stress_constraint_formulation=vanishing`` 保留为 LFEM 对照选项.
    """
    formulation = resolve_stress_constraint_formulation(config, method)
    problem = build_stress_problem(parameters, model_name, n_cells=config.ny)
    mesh = create_mesh(problem, config.nx, config.ny, config.mesh_type)
    material = build_material(problem)
    interpolation = build_interpolation(
        material,
        interpolation_method=config.interpolation_method,
        penalty_factor=config.penalty_factor,
        void_youngs_modulus=config.void_youngs_modulus,
    )
    analyzer, _ = build_analyzer(
        method=method,
        mesh=mesh,
        problem=problem,
        material=material,
        order=order,
        solve_method=config.solve_method,
        use_relaxation=config.use_relaxation,
        interpolation=interpolation,
    )
    relaxation = (
        EpsilonRelaxedStressFormulation(epsilon=config.epsilon)
        if formulation == "apparent"
        else PolynomialVanishingStressFormulation()
    )
    constraint_type = (
        LagrangeStressConstraint if method == "lfem" else HuZhangStressConstraint
    )
    # 几何应力奇点有两处, 都不随设计消失, 按各自的固定物理半径处置, 两条路径
    # 用同一组掩码, 保证对照在同一验收区域上进行: 既剔除其应力评价点, 又把它钉
    # 成实体 (下面注入 problem, 由优化器与过滤器共同施加). 只做前者时优化器会把
    # 该处减料换体积, 制造出不受约束的过应力 (2026-09-16 对照).
    #   载荷侧: 贴片端点的牵引间断. 已先经载荷分布化削弱为对数型.
    #   支撑侧: 固支边两端的 Dirichlet--Neumann 角点, 幂律型 r^(-0.2189), 没有
    #           可做的分布化, 故残留奇异性比载荷侧强, 半径自然也不该相同.
    load_pad_mask = build_exemption_mask(
        mesh=mesh,
        centers=problem.traction_patch_endpoints,
        radius=config.load_pad_radius,
    )
    support_pad_mask = build_exemption_mask(
        mesh=mesh,
        centers=problem.clamped_corner_points,
        radius=config.support_pad_radius,
    )
    pad_mask = bm.logical_or(load_pad_mask, support_pad_mask)
    problem.set_passive_element_mask(pad_mask)
    stress_constraint: Any = constraint_type(
        analyzer=analyzer,
        stress_limit=config.stress_limit,
        formulation=relaxation,
        exemption_mask=pad_mask,
        enable_logging=False,
    )

    # 初始设计变量取 0.5 (博士论文配方; 满密度 1.0 启动会导致 ALM 发散)
    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh,
        displacement_mesh=mesh,
        relative_density=config.initial_density,
    )
    volume_objective = VolumeObjective(analyzer=analyzer, enable_logging=False)
    al_objective = AugmentedLagrangianObjective(
        volume_objective=volume_objective,
        stress_constraint=stress_constraint,
        options=_build_al_options(config),
        enable_logging=False,
    )
    return StressOptimizationPipeline(
        method=method,
        order=order,
        problem=problem,
        mesh=mesh,
        analyzer=analyzer,
        design_variable=design_variable,
        density_distribution=density,
        volume_objective=volume_objective,
        stress_constraint=stress_constraint,
        al_objective=al_objective,
        optimizer=None,
        load_pad_mask=load_pad_mask,
        support_pad_mask=support_pad_mask,
        pad_mask=pad_mask,
    )


def build_stress_pipeline(
    config: CantileverStressExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "CantileverMiddle2d",
) -> StressOptimizationPipeline:
    """构建带 ALM-MMA 优化器的完整执行流水线."""
    pipeline = build_stress_analysis_pipeline(config, parameters, method, order, model_name)
    projection_params = None
    if config.filter_type == "projection":
        # 博士论文配方: tanh 投影 + beta 加法连续化 (1.0 -> 10.0, 每 5 步 +1)
        projection_params = {
            "continuation_strategy": "additive",
            "projection_type": "tanh",
            "beta": 1.0,
            "beta_max": 10.0,
            "continuation_iter": 5,
            "beta_increment": 1.0,
        }
    pipeline.optimizer = ALMMMAOptimizer(
        al_objective=pipeline.al_objective,
        filter=build_density_filter(
            pipeline.mesh,
            filter_type=config.filter_type,
            filter_radius=config.filter_radius,
            projection_params=projection_params,
            # 实体保留必须施加在过滤/投影之后: 只固定设计变量时, rmin=6 的
            # 宽过滤下垫片单元的物理密度仍由邻域决定, 达不到实体.
            passive_mask=pipeline.pad_mask,
        ),
        options=_build_al_options(config),
        enable_logging=True,
    )
    return pipeline


# ============================================================ 四、装配器注册表


@dataclass(frozen=True)
class CaseAssembler:
    """一族物理算例的三个装配入口.

    调用方 (``driver.py`` / ``metrics.py``) 只按模型名取装配器, 不关心本模块
    内部把哪几个函数归成一族.
    """

    build_config: Callable[[dict[str, Any]], Any]
    build_analysis_pipeline: Callable[..., Any]
    build_pipeline: Callable[..., Any]


_FIXED_FIXED = CaseAssembler(
    build_config=build_fixed_fixed_config,
    build_analysis_pipeline=build_fixed_fixed_analysis_pipeline,
    build_pipeline=build_fixed_fixed_pipeline,
)
_BEARING = CaseAssembler(
    build_config=build_bearing_config,
    build_analysis_pipeline=build_bearing_analysis_pipeline,
    build_pipeline=build_bearing_pipeline,
)
_STRESS = CaseAssembler(
    build_config=build_stress_config,
    build_analysis_pipeline=build_stress_analysis_pipeline,
    build_pipeline=build_stress_pipeline,
)

# cases.toml 的 [cases.model] name -> 装配器; 未登记的模型没有 Runner
ASSEMBLERS: dict[str, CaseAssembler] = {
    "FixedFixedBeamCenterLoad2d": _FIXED_FIXED,
    "FixedFixedBeamHalfDomain2d": _FIXED_FIXED,
    "BearingDevice2d": _BEARING,
    "CantileverMiddle2d": _STRESS,
}


def assembler_for(model_name: str) -> CaseAssembler:
    """按模型名取装配器; 未注册时抛 KeyError, 由调用方翻译成自己的异常类型."""
    return ASSEMBLERS[model_name]
