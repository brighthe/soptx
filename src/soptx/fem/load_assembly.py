"""Lagrange 张量空间上的物理载荷装配.

把 ``Problem.loads()`` 给出的载荷对象装配成按空间自由度排序的全局载荷向量:

- ``BodyForce``: 体积分 (``SourceIntegrator``);
- ``BoundaryTraction``: 边界积分 (``LagrangeBoundarySourceIntegrator``);
- ``PointForce`` 与 ``LineTraction``: 经 ``soptx.fem.load_projection.project_nodal_loads``
  投影到节点, 再换成空间的自由度排序.

体力与非体力分开装配, 因为分析器按边界类型决定非体力载荷是否进入右端项. 边界类型
分派、载荷缓存与伴随载荷属于分析器, 不在本模块.
"""

from typing import Iterable

from soptx.backend import backend_manager as bm
from soptx.fem.integrators import LagrangeBoundarySourceIntegrator, SourceIntegrator
from soptx.fem.linear_form import LinearForm
from soptx.fem.load_projection import project_nodal_loads
from soptx.functionspace import TensorFunctionSpace
from soptx.protocols import BodyForce, BoundaryTraction, LineTraction, Load, PointForce
from soptx.typing import TensorLike


def assemble_body_forces(space: TensorFunctionSpace, loads: Iterable[Load], *, q: int) -> TensorLike:
    """装配 ``loads`` 中全部体力的体积分, 其余载荷跳过.

    Parameters
    ----------
    space : 位移张量函数空间.
    loads : 载荷对象序列, 通常为 ``Problem.loads()``.
    q : 体积分的积分阶次.

    Returns
    -------
    F : (gdof, ) 的载荷向量, 按 ``space`` 的自由度排序; 没有体力时为零向量.
    """
    F = space.function()
    for load in loads:
        if not isinstance(load, BodyForce):
            continue
        lform = LinearForm(space)
        lform.add_integrator(SourceIntegrator(source=load.body_force, q=q))
        F = F + lform.assembly(format='dense')

    return F


def assemble_non_body_loads(space: TensorFunctionSpace, loads: Iterable[Load], *, q: int) -> TensorLike:
    """装配 ``loads`` 中的边界牵引、集中力与线载荷, 体力跳过.

    Parameters
    ----------
    space : 位移张量函数空间; 线载荷按其标量空间的阶次形成一致节点力.
    loads : 载荷对象序列, 通常为 ``Problem.loads()``.
    q : 边界积分的积分阶次.

    Returns
    -------
    F : (gdof, ) 的载荷向量, 按 ``space`` 的自由度排序.

    Raises
    ------
    TypeError
        出现尚未定义装配语义的载荷类型.
    """
    GD = space.mesh.geo_dimension()
    F = space.function()
    nodal_loads = []

    for load in loads:
        if isinstance(load, BodyForce):
            continue
        if isinstance(load, (PointForce, LineTraction)):
            nodal_loads.append(load)
            continue
        if isinstance(load, BoundaryTraction):
            integrator = LagrangeBoundarySourceIntegrator(
                source=load.traction,
                q=q,
                threshold=load.is_load_boundary,
            )
            lform = LinearForm(space)
            lform.add_integrator(integrator)
            F = F + lform.assembly(format='dense')
            continue
        raise TypeError(
            "LFEM 不支持载荷对象 "
            f"{type(load).__name__}; 请提供已定义装配语义的 Load."
        )

    if nodal_loads:
        # project_nodal_loads 给出节点优先排序; 分量优先的空间要转置一次
        node_major = project_nodal_loads(
            nodal_loads,
            space.interpolation_points(),
            GD,
            degree=space.scalar_space.p,
        )
        if space.dof_priority:
            node_values = bm.reshape(node_major, (-1, GD))
            nodal_vector = bm.reshape(bm.transpose(node_values, (1, 0)), (-1, ))
        else:
            nodal_vector = node_major
        F = F + nodal_vector

    return F
