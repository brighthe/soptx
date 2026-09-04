# -*- coding: utf-8 -*-
"""使用 ``soptx`` 公共接口组装 EA 变密度优化链.

``operator_level`` 参数化: 默认组装 EA (matrix-free) 链;
``compare.py`` 用同一函数以 ``operator_level="fa"`` 在进程内构建
FA-cg 参照侧, 保证锁步对照两侧只差算子层级。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    BearingDevice2d,
    CantileverCorner2d,
    CantileverMiddle2d,
    CantileverRightBottomEdge3d,
    HalfMBBBeamRight2d,
    SimplySupportedBridge2d,
)
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import MMAOptimizer, OCOptimizer

from config import MESH_LAYOUT, TopOptCase


# 本模块默认走 EA (matrix-free) 算子层级; compare.py 会显式传 "fa" 在
# 进程内构建 FA 参照。算子层级由目录名声明, 不进 --list;
# ANALYZER_KIND 只是 analyzer 列的离散方法前缀, 后半段取逐工况 solve_method。
OPERATOR_LEVEL = "ea"
ANALYZER_KIND = "lfem"

# 本模块所有工况都是同一个提法: 体积分数约束下的柔顺度最小化
# (build_pipeline 里 ComplianceObjective + VolumeConstraint 写死, 无分支);
# 逐工况只变约束上限 volfrac, 故提法进代码不进 cases.toml 的 summary。
FORMULATION = "体积分数约束下的柔顺度最小化"


MESH_FACTORIES = {
    (2, "tri"): TriangleMesh,
    (2, "quad"): QuadrangleMesh,
    (3, "tet"): TetrahedronMesh,
    (3, "hex"): HexahedronMesh,
}
MESH_DATA_TYPES = {
    "tri": "structured_tri",
    "quad": "uniform_quad",
    "tet": "structured_tet",
    "hex": "uniform_hex",
}


@dataclass
class PipelineComponents:
    """一个工况组装出的全部公共组件."""

    problem: Any
    mesh: Any
    analyzer: LagrangeFEMAnalyzer
    design_variable: Any
    density: Any
    objective: ComplianceObjective
    constraint: VolumeConstraint
    density_filter: Filter
    optimizer: OCOptimizer | MMAOptimizer


def _create_problem(case: TopOptCase) -> Any:
    if case.problem == "cantilever_corner":
        lx, ly = case.domain
        return CantileverCorner2d(
            domain=(0.0, lx, 0.0, ly),
            P=case.load,
            E=case.emax,
            nu=case.nu,
        )
    if case.problem == "cantilever_middle":
        lx, ly = case.domain
        return CantileverMiddle2d(
            domain=(0.0, lx, 0.0, ly),
            P=case.load,
            E=case.emax,
            nu=case.nu,
        )
    if case.problem == "bearing_device":
        lx, ly = case.domain
        # load 字段映射为顶边均布牵引强度 t (N/mm), 而非合力。
        return BearingDevice2d(
            domain=(0.0, lx, 0.0, ly),
            t=case.load,
            E=case.emax,
            nu=case.nu,
        )
    if case.problem == "half_mbb_beam_right":
        lx, ly = case.domain
        # domain 是对称右半域 (整梁跨度为 2 * lx); 左边界为对称面。
        return HalfMBBBeamRight2d(
            domain=(0.0, lx, 0.0, ly),
            P=case.load,
            E=case.emax,
            nu=case.nu,
        )
    if case.problem == "simply_supported_bridge":
        lx, ly = case.domain
        # load 字段映射为顶边均布牵引强度 t (N/mm), 而非合力。桥面实体非设计域
        # 厚度取 H/10 (博士论文算例 3.2: L/30, H = L/3), 随 domain 走, 不进注册表。
        return SimplySupportedBridge2d(
            domain=(0.0, lx, 0.0, ly),
            t=case.load,
            E=case.emax,
            nu=case.nu,
            deck_height=ly / 10.0,
        )
    lx, ly, lz = case.domain
    return CantileverRightBottomEdge3d(
        domain=(0.0, lx, 0.0, ly, 0.0, lz),
        P=case.load,
        E=case.emax,
        nu=case.nu,
    )


def _create_mesh(case: TopOptCase, problem: Any) -> Any:
    subdivisions = {"nx": case.grid[0], "ny": case.grid[1]}
    if case.dimension == 3:
        subdivisions["nz"] = case.grid[2]
    mesh = MESH_FACTORIES[(case.dimension, case.cell_type)].from_box(
        box=list(problem.domain),
        **subdivisions,
    )
    sizes = {
        axis: case.domain[index] / case.grid[index]
        for index, axis in enumerate(("hx", "hy", "hz")[: case.dimension])
    }
    mesh.meshdata = {
        **dict(getattr(mesh, "meshdata", {}) or {}),
        "domain": list(problem.domain),
        "mesh_layout": MESH_LAYOUT,
        "mesh_type": MESH_DATA_TYPES[case.cell_type],
        "cell_type": case.cell_type,
        "nx": case.grid[0],
        "ny": case.grid[1],
        **({"nz": case.grid[2]} if case.dimension == 3 else {}),
        **sizes,
    }
    return mesh


def build_components(
    case: TopOptCase,
    operator_level: Literal["fa", "ea"] = OPERATOR_LEVEL,
) -> PipelineComponents:
    """组装完整组件链, ``compare.py`` 依赖 analyzer 与 objective."""
    problem = _create_problem(case)
    mesh = _create_mesh(case, problem)
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    interpolation = MaterialInterpolationScheme(
        density_location="element",
        interpolation_method=case.interpolation_method,
        options={
            "penalty_factor": case.simp_penalty,
            "void_youngs_modulus": case.void_youngs_modulus,
            "target_variables": ["E"],
        },
        enable_logging=False,
    )
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=case.space_degree,
        integration_order=case.integration_order,
        assembly_method=case.assembly_method,
        operator_level=operator_level,
        solve_method=case.solve_method,
        solver_options={
            "maxiter": case.cg_maxiter,
            "atol": case.cg_atol,
            "rtol": case.cg_rtol,
            # None 表示不加预条件; 取 "jacobi" 时分析器会建 DiagonalPreconditioner,
            # 并把真残差刷新间隔一并绑上 (Jacobi 的 M-范数递推残差会失真)。
            "precond": None if case.cg_precond == "none" else case.cg_precond,
        },
        topopt_algorithm="density_based",
        interpolation_scheme=interpolation,
        enable_logging=False,
    )
    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh,
        displacement_mesh=mesh,
        relative_density=case.volfrac,
    )
    objective = ComplianceObjective(
        analyzer=analyzer,
        state_variable="u",
        diff_mode="manual",
        enable_logging=False,
    )
    constraint = VolumeConstraint(
        analyzer=analyzer,
        volume_fraction=case.volfrac,
        diff_mode="manual",
        enable_logging=False,
    )
    projection_params = None
    if case.filter_type == "projection":
        projection_params = {
            # 与 topopt_simp_fa 一致: 显式钉住投影类型。旧版 Filter 的门面默认
            # 值 (已删除) 会把未指定的 projection_type 静默设成 exponential,
            # 不写死会落到签名默认的 tanh, 改变本实验既有的数值行为。
            "projection_type": "exponential",
            "beta": case.projection_beta,
            "eta": case.projection_eta,
            "beta_max": case.projection_beta_max,
            "continuation_iter": case.projection_continuation_iter,
        }
    density_filter = Filter(
        design_mesh=mesh,
        filter_type=case.filter_type,
        rmin=case.filter_radius,
        density_location="element",
        # 非结构网格走 KD-tree 通用路径, 权重为 (1 - d/rmin)^q。q 曾在
        # FilterMatrixBuilder 内部写死为 3, 现已参数化 (默认 1 = 线性锥形);
        # 这里显式钉住 3, 保持既有结果不变。
        filter_q=3,
        projection_params=projection_params,
        enable_logging=False,
    )
    common = {
        "objective": objective,
        "constraint": constraint,
        "filter": density_filter,
        "enable_logging": False,
    }
    options = {
        "max_iterations": case.max_iter,
        "change_tolerance": case.tol_change,
    }
    if case.optimizer == "oc":
        optimizer = OCOptimizer(options=options, **common)
        optimizer.options.set_advanced_options(
            move_limit=case.move,
            design_variable_min=case.density_min,
        )
    else:
        optimizer = MMAOptimizer(
            options={**options, "move_limit": case.move},
            **common,
        )
    return PipelineComponents(
        problem=problem,
        mesh=mesh,
        analyzer=analyzer,
        design_variable=design_variable,
        density=density,
        objective=objective,
        constraint=constraint,
        density_filter=density_filter,
        optimizer=optimizer,
    )


def build_pipeline(
    case: TopOptCase,
    operator_level: Literal["fa", "ea"] = OPERATOR_LEVEL,
) -> tuple[Any, Any, Any, Any, Any]:
    """返回 ``optimizer, design_variable, density_distribution, mesh, analyzer``."""
    components = build_components(case, operator_level)
    return (
        components.optimizer,
        components.design_variable,
        components.density,
        components.mesh,
        components.analyzer,
    )
