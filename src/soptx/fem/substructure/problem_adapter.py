"""将 elasticity Problem 的约束与物理载荷投影到子结构求解系统."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.fem import LinearForm

from soptx.fem.integrators import (
    LagrangeBoundarySourceIntegrator,
    SourceIntegrator,
)
from soptx.fem.load_projection import project_line_traction, project_point_force
from soptx.protocols import (
    BodyForce,
    BoundaryTraction,
    LineTraction,
    Load,
    PointForce,
)


@dataclass(frozen=True)
class InterfaceConditions:
    """完整接口系统及对应全尺度系统上的载荷与齐次约束."""

    interface_force: Any
    interface_fixed_dofs: Any
    full_force: Any
    full_fixed_dofs: Any


def project_problem_conditions_to_nodes(
    problem: Any,
    node_coordinates: Any,
    *,
    tensor_space: Optional[Any] = None,
    coordinate_offset: Optional[Any] = None,
    integration_order: int = 4,
    degree: int = 1,
) -> Tuple[Any, Any]:
    """把 Problem 的 Dirichlet 约束与 ``Load`` 对象投影到节点自由度.

    参数
    ----
    problem : Any
        提供 ``dimension``, ``domain``, ``is_dirichlet_boundary()`` 与
        ``loads()`` 的 elasticity Problem.
    node_coordinates : TensorLike
        物理坐标, 形状为 ``(N, dimension)``. NumPy 数组与当前 bm 后端数组
        均可使用.
    tensor_space : TensorFunctionSpace, optional
        与 ``node_coordinates`` 对应的全尺度 Lagrange 位移空间. 提供后可通过
        数值积分装配 ``BodyForce`` 与 ``BoundaryTraction``; 省略时只支持
        ``PointForce`` 与 ``LineTraction``.
    coordinate_offset : TensorLike, optional
        ``tensor_space`` 局部坐标到 Problem 物理坐标的平移量.
    degree : int, optional
        提供 ``node_coordinates`` 的位移空间阶次, 决定线载荷按几阶 Lagrange
        线单元形成一致节点力.

    返回
    ----
    force : TensorLike
        节点优先排列的全局载荷向量.
    fixed_dofs : TensorLike
        升序、去重后的 Dirichlet 自由度编号.

    异常
    ----
    ValueError
        坐标、Problem 或载荷契约不一致时抛出.
    TypeError
        载荷对象不满足公共契约, 或缺少其离散所需的有限元空间时抛出.
    """
    dim = int(problem.dimension)
    coordinates = bm.asarray(node_coordinates, dtype=bm.float64)
    if len(coordinates.shape) != 2 or int(coordinates.shape[1]) != dim:
        raise ValueError(
            "node_coordinates 应为形状 (N, problem.dimension) 的二维数组; "
            f"实际为 {tuple(coordinates.shape)}."
        )

    n_nodes = int(coordinates.shape[0])
    device = bm.get_device(coordinates)
    force = bm.zeros(
        (dim * n_nodes,),
        dtype=bm.float64,
        device=device,
    )
    fixed_parts = []
    dirichlet_predicates = tuple(problem.is_dirichlet_boundary())
    if len(dirichlet_predicates) != dim:
        raise ValueError(
            "is_dirichlet_boundary() 必须按位移分量返回 predicate; "
            f"dimension={dim}, predicate 数量={len(dirichlet_predicates)}."
        )

    for component, predicate in enumerate(dirichlet_predicates):
        if predicate is None:
            continue
        mask = bm.asarray(predicate(coordinates), dtype=bm.bool)
        if tuple(mask.shape) != (n_nodes,):
            raise ValueError(
                "Dirichlet predicate 必须返回形状 (N,) 的布尔掩码; "
                f"第 {component} 分量返回 {tuple(mask.shape)}."
            )
        nodes = bm.nonzero(mask)[0]
        fixed_parts.append(dim * nodes + component)

    if fixed_parts:
        fixed_dofs = bm.unique(bm.concat(fixed_parts, axis=0))
    else:
        fixed_dofs = bm.zeros((0,), dtype=bm.int64, device=device)

    loads = tuple(problem.loads())
    for load in loads:
        if not isinstance(load, Load):
            raise TypeError(f"{type(load).__name__} 不满足公共 Load 契约.")
        if int(load.dimension) != dim:
            raise ValueError(
                f"{type(load).__name__}.dimension={load.dimension} 与 "
                f"problem.dimension={dim} 不一致."
            )

        if isinstance(load, PointForce):
            force = force + project_point_force(
                load,
                coordinates,
                dim,
                mode="nearest_boundary",
                domain=tuple(problem.domain),
            )
            continue

        if isinstance(load, LineTraction):
            force = force + project_line_traction(
                load,
                coordinates,
                dim,
                degree=degree,
            )
            continue

        if isinstance(load, (BodyForce, BoundaryTraction)):
            if tensor_space is None:
                raise TypeError(
                    f"{type(load).__name__} 需要全尺度有限元空间执行数值积分; "
                    "仅有节点坐标时不能精确装配."
                )
            force = force + _assemble_integrated_load(
                load,
                tensor_space,
                dim,
                n_nodes,
                coordinate_offset,
                integration_order,
            )
            continue

        raise TypeError(
            f"子结构条件投影不支持载荷对象 {type(load).__name__}."
        )

    return force, fixed_dofs


def _assemble_integrated_load(
    load: Load,
    tensor_space: Any,
    dim: int,
    n_nodes: int,
    coordinate_offset: Optional[Any],
    integration_order: int,
) -> Any:
    """在全尺度 Lagrange 空间中装配体力或边界牵引."""
    if int(tensor_space.number_of_global_dofs()) != dim * n_nodes:
        raise ValueError(
            "tensor_space 自由度数与 node_coordinates 不一致."
        )
    if int(getattr(tensor_space.scalar_space, "p", 1)) != 1:
        raise NotImplementedError(
            "子结构分布载荷装配当前只验证了 p=1 全尺度 Lagrange 空间."
        )
    offset = (
        bm.zeros((dim,), dtype=bm.float64, device=tensor_space.device)
        if coordinate_offset is None
        else bm.asarray(coordinate_offset, dtype=bm.float64)
    )

    if isinstance(load, BodyForce):
        @cartesian
        def source(points: Any) -> Any:
            return load.body_force(points + offset)

        integrator = SourceIntegrator(source=source, q=integration_order)
    elif isinstance(load, BoundaryTraction):
        @cartesian
        def source(points: Any) -> Any:
            return load.traction(points + offset)

        @cartesian
        def threshold(points: Any) -> Any:
            return load.is_load_boundary(points + offset)

        integrator = LagrangeBoundarySourceIntegrator(
            source=source,
            q=integration_order,
            threshold=threshold,
        )
    else:
        raise TypeError("数值积分入口只接受 BodyForce 或 BoundaryTraction.")

    lform = LinearForm(tensor_space)
    lform.add_integrator(integrator)
    vector = lform.assembly(format="dense")
    if tensor_space.dof_priority:
        component_values = bm.reshape(vector, (dim, n_nodes))
        vector = bm.reshape(
            bm.transpose(component_values, (1, 0)),
            (-1,),
        )
    return vector


def project_problem_conditions_to_macro_system(
    problem: Any,
    assembler: Any,
) -> Tuple[Any, Any]:
    """把 problem 公共契约投影到 ``GlobalAssembler`` 的宏观自由度.

    ``GlobalAssembler.macro_node_coordinates()`` 使用从零开始的局部坐标;
    本函数根据公开的 ``problem.domain`` 补上物理域原点偏移.

    角点宏观系统当前直接支持 ``PointForce`` 与 ``LineTraction``. 体力和边界
    牵引还需要与局部缩聚一致的载荷迹降阶, 不能只凭宏观节点坐标装配.
    """
    dim = int(problem.dimension)
    if int(assembler.dim) != dim:
        raise ValueError(
            f"problem.dimension={dim} 与 assembler.dim={assembler.dim} 不一致."
        )
    domain = tuple(problem.domain)
    if len(domain) != 2 * dim:
        raise ValueError(
            f"problem.domain 长度应为 {2 * dim}, 实际为 {len(domain)}."
        )
    origin = bm.asarray(domain[0::2], dtype=bm.float64)
    physical_coordinates = assembler.macro_node_coordinates() + origin
    return project_problem_conditions_to_nodes(problem, physical_coordinates)


def project_problem_conditions_to_full_system(
    problem: Any,
    assembler: Any,
) -> Tuple[Any, Any]:
    """把 problem 公共契约投影到全尺度结构化网格自由度.

    ``GlobalAssembler.node_coordinates()`` 使用从零开始的局部坐标; 本函数
    根据 ``problem.domain`` 加上物理域原点偏移. 返回向量采用节点优先自由度
    排列, 与 ``GlobalAssembler`` 的全局自由度编号一致.
    """
    dim = int(problem.dimension)
    if int(assembler.dim) != dim:
        raise ValueError(
            f"problem.dimension={dim} 与 assembler.dim={assembler.dim} 不一致."
        )
    domain = tuple(problem.domain)
    if len(domain) != 2 * dim:
        raise ValueError(
            f"problem.domain 长度应为 {2 * dim}, 实际为 {len(domain)}."
        )
    origin = bm.asarray(domain[0::2], dtype=bm.float64)
    physical_coordinates = assembler.node_coordinates() + origin
    needs_integration = any(
        isinstance(load, (BodyForce, BoundaryTraction))
        for load in problem.loads()
    )
    return project_problem_conditions_to_nodes(
        problem,
        physical_coordinates,
        tensor_space=assembler.space_full if needs_integration else None,
        coordinate_offset=origin,
        integration_order=max(4, int(assembler.degree) + 2),
        degree=int(assembler.degree),
    )


def project_problem_conditions_to_interface_system(
    problem: Any,
    assembler: Any,
    system: Any,
) -> InterfaceConditions:
    """把完整网格载荷与约束无损投影到完整接口系统.

    当前静力缩聚结果只包含齐次内部恢复 ``u_i = N u_b``, 尚不包含内部载荷
    引起的 particular solution. 因此本函数显式拒绝作用在子结构内部自由度上的
    等效节点载荷或 Dirichlet 约束, 避免静默丢失右端项或改变缩聚问题. MBB 等载荷
    与支承位于子结构外边界的基准满足该契约.
    """
    full_force, full_fixed_dofs = project_problem_conditions_to_full_system(
        problem,
        assembler,
    )
    interface_global_dofs = bm.asarray(system.global_dofs, dtype=bm.int64)

    nonzero_load_dofs = bm.nonzero(bm.abs(full_force) > 0.0)[0]
    load_on_interface = bm.isin(nonzero_load_dofs, interface_global_dofs)
    if len(nonzero_load_dofs) > 0 and not bool(bm.all(load_on_interface)):
        missing = nonzero_load_dofs[~load_on_interface]
        raise ValueError(
            "完整接口静力缩聚暂不支持子结构内部载荷; "
            f"检测到 {len(missing)} 个内部载荷自由度."
        )

    fixed_on_interface = bm.isin(full_fixed_dofs, interface_global_dofs)
    if len(full_fixed_dofs) > 0 and not bool(bm.all(fixed_on_interface)):
        missing = full_fixed_dofs[~fixed_on_interface]
        raise ValueError(
            "完整接口静力缩聚暂不支持子结构内部 Dirichlet 约束; "
            f"检测到 {len(missing)} 个内部约束自由度."
        )

    return InterfaceConditions(
        interface_force=assembler.project_global_vector(system, full_force),
        interface_fixed_dofs=assembler.project_global_dofs(
            system,
            full_fixed_dofs,
        ),
        full_force=full_force,
        full_fixed_dofs=full_fixed_dofs,
    )


__all__ = [
    "InterfaceConditions",
    "project_problem_conditions_to_full_system",
    "project_problem_conditions_to_interface_system",
    "project_problem_conditions_to_macro_system",
    "project_problem_conditions_to_nodes",
]
