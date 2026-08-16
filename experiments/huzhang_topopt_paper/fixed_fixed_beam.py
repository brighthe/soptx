"""两端固支梁柔顺度优化的组装入口."""

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
from soptx.problems import FixedFixedBeamCenterLoad2d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import OCOptimizer


MethodName = Literal["lfem", "huzhang"]
InterpolationMethod = Literal["simp", "msimp", "ramp"]
FilterType = Literal["density", "sensitivity"]
SolveMethod = Literal["mumps", "scipy"]

# 目前唯一支持的共同载荷离散方式, 见 build_problem
P1_TRACE_L2_PROJECTION = "p1_trace_l2_projection"


@dataclass(frozen=True)
class FixedFixedBeamExperimentConfig:
    """投稿论文全域两端固支梁的实验参数."""

    nx: int
    ny: int
    volume_fraction: float
    filter_radius: float
    load_width: float
    load_discretization: str
    plane_type: str
    load: float
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


@dataclass
class OptimizationPipeline:
    """保存一次优化所需的已组装对象."""

    method: MethodName
    order: int
    problem: FixedFixedBeamCenterLoad2d
    mesh: Any
    analyzer: Any
    design_variable: Any
    density_distribution: Any
    objective: ComplianceObjective
    constraint: VolumeConstraint
    optimizer: OCOptimizer | None


def _instantiate(
    parameters: dict[str, Any],
    traction: Optional[Callable[[TensorLike], TensorLike]] = None,
) -> FixedFixedBeamCenterLoad2d:
    """按 case 参数实例化物理问题, 可选地替换牵引函数."""
    return FixedFixedBeamCenterLoad2d(
        P=float(parameters["load"]),
        E=float(parameters["youngs_modulus"]),
        nu=float(parameters["poisson_ratio"]),
        load_width=float(parameters["load_width"]),
        plane_type=str(parameters["plane_type"]),
        traction=traction,
    )


def build_problem(
    parameters: dict[str, Any],
    *,
    n_cells: Optional[int] = None,
) -> FixedFixedBeamCenterLoad2d:
    """从 case 参数构造唯一的两端固支梁物理问题.

    ``n_cells`` 缺省时返回原始物理问题, 其贴片牵引在贴片边缘不连续.
    给定 ``n_cells`` (底边单元数) 时, 把该牵引替换为它在底边连续 P1 迹
    空间上的 L2 投影: 合力被精确保持, 而 LFEM 的边界数值积分与 Hu--Zhang
    的迹插值都能精确重现它, 于是两条离散路径看到同一个载荷泛函. 受控比较
    必须走这条分支.

    投影只依赖计算域、底边剖分数和贴片几何, 不需要网格对象; 贴片几何一律
    从物理问题的 ``traction_patch``/``traction_level``/``traction_intensity``
    读取, 不在此处复述.
    """
    problem = _instantiate(parameters)
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
    return _instantiate(parameters, traction=common_load)


def build_config(parameters: dict[str, Any]) -> FixedFixedBeamExperimentConfig:
    """校验并返回实验配置."""
    interpolation_method = str(parameters["interpolation_method"])
    if interpolation_method not in ("simp", "msimp", "ramp"):
        raise ValueError("不支持的材料插值方法.")
    filter_type = str(parameters["filter_type"])
    if filter_type not in ("density", "sensitivity"):
        raise ValueError("投稿算例仅允许 density 或 sensitivity 过滤.")
    solve_method = str(parameters["solve_method"])
    if solve_method not in ("mumps", "scipy"):
        raise ValueError("直接求解器必须为 scipy 或 mumps.")

    config = FixedFixedBeamExperimentConfig(
        nx=int(parameters["nx"]),
        ny=int(parameters["ny"]),
        volume_fraction=float(parameters["volume_fraction"]),
        filter_radius=float(parameters["filter_radius"]),
        load_width=float(parameters["load_width"]),
        load_discretization=str(parameters["load_discretization"]),
        plane_type=str(parameters["plane_type"]),
        load=float(parameters["load"]),
        youngs_modulus=float(parameters["youngs_modulus"]),
        poisson_ratio=float(parameters["poisson_ratio"]),
        comparison_orders=tuple(int(value) for value in parameters["comparison_orders"]),
        interpolation_method=interpolation_method,
        penalty_factor=float(parameters["penalty_factor"]),
        void_youngs_modulus=float(parameters["void_youngs_modulus"]),
        filter_type=filter_type,
        max_iterations=int(parameters["max_iterations"]),
        change_tolerance=float(parameters["change_tolerance"]),
        use_relaxation=bool(parameters["use_relaxation"]),
        solve_method=solve_method,
    )
    if config.load_discretization != P1_TRACE_L2_PROJECTION:
        raise ValueError(f"不支持的共同载荷离散方式: {config.load_discretization}.")
    if config.nx <= 0 or config.ny <= 0:
        raise ValueError("网格剖分数必须为正数.")
    if config.nx % 2 or config.ny % 2:
        raise ValueError("棋盘格角点松弛网格要求 nx 和 ny 均为偶数.")
    if not 0.0 < config.volume_fraction <= 1.0:
        raise ValueError("体积分数必须位于 (0, 1] 区间.")
    if config.filter_radius <= 0.0 or config.load_width <= 0.0:
        raise ValueError("过滤半径和载荷宽度必须为正数.")
    if config.youngs_modulus <= 0.0 or config.void_youngs_modulus <= 0.0:
        raise ValueError("实体和空材料 Young 模量必须为正数.")
    if not config.comparison_orders:
        raise ValueError("比较阶次不能为空.")
    if any(order not in (2, 3, 4) for order in config.comparison_orders):
        raise ValueError("Hu--Zhang 投稿算例仅允许 k=2,3,4.")
    if config.max_iterations <= 0 or config.change_tolerance <= 0.0:
        raise ValueError("迭代次数和变化容限必须为正数.")
    return config


def create_mesh(problem: FixedFixedBeamCenterLoad2d, nx: int, ny: int) -> Any:
    """构造两种离散共享的 Hu--Zhang 兼容棋盘格网格.

    ``meshdata`` 是 soptx 附加在 fealpy 网格对象上的元数据字典 (过滤器矩阵等
    依赖它), fealpy 的网格类并未声明该属性, 因此这里显式放宽为 ``Any``.
    """
    mesh: Any = create_huzhang_checkerboard_mesh(
        box=problem.domain,
        nx=nx,
        ny=ny,
    )
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


def build_analysis_pipeline(
    config: FixedFixedBeamExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
) -> OptimizationPipeline:
    """按受控比较协议组装一条 LFEM 或 Hu--Zhang 分析链.

    对同一 ``order=k``，LFEM 使用 ``p=k``，Hu--Zhang 使用应力阶
    ``k``，二者统一使用积分阶 ``q=2k+2``。
    """
    if order not in config.comparison_orders:
        raise ValueError(f"比较阶次 {order} 不在投稿配置中.")

    # 底边单元数即 nx, 交叉网格的底边界正好有 nx 条边
    problem = build_problem(parameters, n_cells=config.nx)
    mesh = create_mesh(problem, config.nx, config.ny)
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    interpolation = MaterialInterpolationScheme(
        density_location="element",
        interpolation_method=config.interpolation_method,
        options={
            "penalty_factor": config.penalty_factor,
            "void_youngs_modulus": config.void_youngs_modulus,
            "target_variables": ["E"],
        },
        enable_logging=False,
    )

    integration_order = 2 * order + 2
    if method == "lfem":
        analyzer = LagrangeFEMAnalyzer(
            disp_mesh=mesh,
            pde=problem,
            material=material,
            space_degree=order,
            integration_order=integration_order,
            assembly_method="standard",
            solve_method=config.solve_method,
            topopt_algorithm="density_based",
            interpolation_scheme=interpolation,
        )
        state_variable = "u"
    else:
        analyzer = HuZhangMFEMAnalyzer(
            disp_mesh=mesh,
            pde=problem,
            material=material,
            space_degree=order,
            integration_order=integration_order,
            use_relaxation=config.use_relaxation,
            solve_method=config.solve_method,
            topopt_algorithm="density_based",
            interpolation_scheme=interpolation,
        )
        state_variable = "sigma"

    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh,
        displacement_mesh=mesh,
        relative_density=config.volume_fraction,
    )
    objective = ComplianceObjective(
        analyzer=analyzer,
        state_variable=state_variable,
        diff_mode="manual",
        enable_logging=False,
    )
    constraint = VolumeConstraint(
        analyzer=analyzer,
        volume_fraction=config.volume_fraction,
        diff_mode="manual",
        enable_logging=False,
    )
    return OptimizationPipeline(
        method=method,
        order=order,
        problem=problem,
        mesh=mesh,
        analyzer=analyzer,
        design_variable=design_variable,
        density_distribution=density,
        objective=objective,
        constraint=constraint,
        optimizer=None,
    )


def build_pipeline(
    config: FixedFixedBeamExperimentConfig,
    parameters: dict[str, Any],
    method: MethodName,
    order: int,
) -> OptimizationPipeline:
    """在单次状态分析链上增加 OC 优化器."""
    pipeline = build_analysis_pipeline(config, parameters, method, order)
    density_filter = Filter(
        design_mesh=pipeline.mesh,
        filter_type=config.filter_type,
        rmin=config.filter_radius,
        density_location="element",
        enable_logging=False,
    )
    optimizer = OCOptimizer(
        objective=pipeline.objective,
        constraint=pipeline.constraint,
        filter=density_filter,
        options={
            "max_iterations": config.max_iterations,
            "change_tolerance": config.change_tolerance,
        },
        enable_logging=True,
    )
    optimizer.options.set_advanced_options(
        move_limit=0.2,
        damping_coef=0.5,
        initial_lambda=1.0e9,
        bisection_tol=1.0e-3,
    )
    pipeline.optimizer = optimizer
    return pipeline

