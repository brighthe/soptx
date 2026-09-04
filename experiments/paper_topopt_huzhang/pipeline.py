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

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from fealpy.typing import TensorLike

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    LagrangeFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
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
    ApparentStressConstraint,
    VanishingStressConstraint,
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

INTERPOLATION_METHODS = ("simp", "msimp", "ramp")
FILTER_TYPES = ("density", "sensitivity", "projection")
SOLVE_METHODS = ("mumps", "scipy")
COMPLIANCE_OPTIMIZERS = ("oc", "mma")

# 目前唯一支持的共同载荷离散方式, 见各算例模块的 build_problem
P1_TRACE_L2_PROJECTION = "p1_trace_l2_projection"


def validate_choice(value: str, allowed: tuple[str, ...], message: str) -> str:
    """校验枚举型参数取值."""
    if value not in allowed:
        raise ValueError(f"{message}: {value} 不在 {allowed} 中.")
    return value


def validate_mesh_size(nx: int, ny: int) -> None:
    """校验棋盘格角点松弛网格对剖分数的要求."""
    if nx <= 0 or ny <= 0:
        raise ValueError("网格剖分数必须为正数.")
    if nx % 2 or ny % 2:
        raise ValueError("棋盘格角点松弛网格要求 nx 和 ny 均为偶数.")


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


def create_mesh(problem: Any, nx: int, ny: int) -> Any:
    """构造两种离散共享的 Hu--Zhang 兼容棋盘格交叉三角形网格.

    ``meshdata`` 是 soptx 附加在 fealpy 网格对象上的元数据字典 (过滤器矩阵等
    依赖它), fealpy 的网格类并未声明该属性, 因此这里显式放宽为 ``Any``.
    """
    mesh: Any = create_huzhang_checkerboard_mesh(box=problem.domain, nx=nx, ny=ny)
    xmin, xmax, ymin, ymax = problem.domain
    mesh.meshdata = {
        "domain": list(problem.domain),
        "mesh_type": "uniform_crisscross_tri",
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


def build_interpolation(
    material: IsotropicLinearElasticMaterial,
    *,
    interpolation_method: str,
    penalty_factor: float,
    void_youngs_modulus: float,
) -> MaterialInterpolationScheme:
    """构造材料插值格式.

    近不可压缩材料 (``nu >= 0.49``) 额外对 Poisson 比插值, 使空区域退化为可
    压缩弱材料, 避免空单元的体积锁定污染实体区域的应力.
    """
    options: dict[str, Any] = {
        "penalty_factor": penalty_factor,
        "void_youngs_modulus": void_youngs_modulus,
        "target_variables": ["E"],
    }
    if material.is_incompressible:
        options["target_variables"] = ["E", "nu"]
        options["nu_penalty_factor"] = 1.0
        options["void_poisson_ratio"] = 0.3
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
    if order not in config.comparison_orders:
        raise ValueError(f"比较阶次 {order} 不在投稿配置中.")

    mesh = create_mesh(problem, config.nx, config.ny)
    material = build_material(problem)
    interpolation = build_interpolation(
        material,
        interpolation_method=config.interpolation_method,
        penalty_factor=config.penalty_factor,
        void_youngs_modulus=config.void_youngs_modulus,
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
            move_limit=0.2,
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
    pipeline.optimizer = optimizer
    return pipeline


def compliance_config_fields(parameters: dict[str, Any]) -> dict[str, Any]:
    """解析两族柔顺度算例共有的 17 个配置字段.

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
    }


def validate_compliance_config(config: Any, allowed_orders: tuple[int, ...]) -> None:
    """两族柔顺度算例共有的配置校验; 允许阶次集合按算例给定."""
    validate_mesh_size(config.nx, config.ny)
    validate_orders(config.comparison_orders, allowed_orders)
    if not 0.0 < config.volume_fraction <= 1.0:
        raise ValueError("体积分数必须位于 (0, 1] 区间.")
    if config.filter_radius <= 0.0:
        raise ValueError("过滤半径必须为正数.")
    if config.youngs_modulus <= 0.0 or config.void_youngs_modulus <= 0.0:
        raise ValueError("实体和空材料 Young 模量必须为正数.")
    if config.max_iterations <= 0 or config.change_tolerance <= 0.0:
        raise ValueError("迭代次数和变化容限必须为正数.")


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
    if n_cells is None:
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
    if config.load_discretization != P1_TRACE_L2_PROJECTION:
        raise ValueError(f"不支持的共同载荷离散方式: {config.load_discretization}.")
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
    validate_compliance_config(config, (2, 3, 4))
    return config


def build_bearing_analysis_pipeline(
    config: BearingDeviceExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "BearingDevice2d",
) -> OptimizationPipeline:
    """按受控比较协议组装一条 LFEM 或 Hu--Zhang 分析链.

    近不可压缩算例 (``nu = 0.4999``) 由 ``common.build_interpolation`` 自动
    启用 E 与 nu 的双参数插值.
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


@dataclass(frozen=True)
class CantileverStressExperimentConfig:
    """投稿论文二维悬臂梁局部应力约束优化的实验参数."""

    nx: int
    ny: int
    filter_radius: float
    load_width: float
    load_discretization: str
    plane_type: str
    load: float
    youngs_modulus: float
    poisson_ratio: float
    stress_limit: float
    epsilon: float
    comparison_orders: tuple[int, ...]
    interpolation_method: InterpolationMethod
    penalty_factor: float
    void_youngs_modulus: float
    filter_type: FilterType
    max_al_iterations: int
    mma_iters_per_al: int
    change_tolerance: float
    stress_tolerance: float
    use_relaxation: bool
    solve_method: SolveMethod
    optimizer: StressOptimizerName
    initial_density: float
    mu_0: float
    mu_max: float
    alpha: float
    lambda_0_init_val: float
    move_limit: float

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
    return CantileverStressExperimentConfig(
        nx=int(parameters["nx"]),
        ny=int(parameters["ny"]),
        filter_radius=float(parameters.get("filter_radius", 2.0)),
        load_width=float(parameters.get("load_width", 4.0)),
        load_discretization=str(parameters.get("load_discretization", "patch")),
        plane_type=str(parameters.get("plane_type", "plane_stress")),
        load=float(parameters.get("load", -100.0)),
        youngs_modulus=float(parameters.get("youngs_modulus", 70000.0)),
        poisson_ratio=float(parameters.get("poisson_ratio", 0.25)),
        stress_limit=float(parameters.get("stress_limit", 180.0)),
        epsilon=float(parameters.get("epsilon", 1.0e-4)),
        comparison_orders=tuple(int(v) for v in parameters.get("comparison_orders", [2])),
        interpolation_method=str(parameters.get("interpolation_method", "simp")),
        penalty_factor=float(parameters.get("penalty_factor", 3.0)),
        void_youngs_modulus=float(parameters.get("void_youngs_modulus", 1.0e-9)),
        filter_type=str(parameters.get("filter_type", "density")),
        max_al_iterations=int(parameters.get("max_al_iterations", 100)),
        mma_iters_per_al=int(parameters.get("mma_iters_per_al", 5)),
        change_tolerance=float(parameters.get("change_tolerance", 2.0e-3)),
        stress_tolerance=float(parameters.get("stress_tolerance", 3.0e-3)),
        use_relaxation=bool(parameters.get("use_relaxation", True)),
        solve_method=str(parameters.get("solve_method", "mumps")),
        optimizer=str(parameters.get("optimizer", "al_mma")),
        initial_density=float(parameters.get("initial_density", 0.5)),
        mu_0=float(parameters.get("mu_0", 50.0)),
        mu_max=float(parameters.get("mu_max", 10000.0)),
        alpha=float(parameters.get("alpha", 1.1)),
        lambda_0_init_val=float(parameters.get("lambda_0_init_val", 0.0)),
        move_limit=float(parameters.get("move_limit", 0.15)),
    )


def _build_al_options(config: CantileverStressExperimentConfig) -> ALMMMAOptions:
    """按博士论文第五章配方构造 ALM-MMA 选项."""
    return ALMMMAOptions(
        change_tolerance=config.change_tolerance,
        stress_tolerance=config.stress_tolerance,
        max_al_iterations=config.max_al_iterations,
        mma_iters_per_al=config.mma_iters_per_al,
        mu_0=config.mu_0,
        mu_max=config.mu_max,
        alpha=config.alpha,
        lambda_0_init_val=config.lambda_0_init_val,
        move_limit=config.move_limit,
    )


def build_stress_analysis_pipeline(
    config: CantileverStressExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
    model_name: str = "CantileverMiddle2d",
) -> StressOptimizationPipeline:
    """组装悬臂梁应力约束分析求解链.

    Hu--Zhang 路径直接用求解得到的应力自由度构造表观应力约束; LFEM 路径
    则由位移场后处理出应力, 走消失约束 (vanishing constraint) 形式.
    """
    problem = build_stress_problem(parameters, model_name, n_cells=config.ny)
    mesh = create_mesh(problem, config.nx, config.ny)
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
    if method == "lfem":
        stress_constraint: Any = VanishingStressConstraint(
            analyzer=analyzer,
            stress_limit=config.stress_limit,
            enable_logging=False,
        )
    else:
        stress_constraint = ApparentStressConstraint(
            analyzer=analyzer,
            stress_limit=config.stress_limit,
            epsilon=config.epsilon,
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
