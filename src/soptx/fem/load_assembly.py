"""Lagrange 张量空间上的物理载荷装配.

把 ``Problem.loads()`` 给出的载荷对象装配成按空间自由度排序的全局载荷向量:

- ``BodyForce``: 体积分 (``SourceIntegrator``);
- ``BoundaryTraction``: 边界积分 (``LagrangeBoundarySourceIntegrator``);
- ``PointForce``: 按实际位移空间的形函数形成一致节点力;
- ``LineTraction``: 经 ``project_nodal_loads`` 积分, 再换成空间的自由度排序.

体力与非体力分开装配, 因为分析器按边界类型决定非体力载荷是否进入右端项. 边界类型
分派、载荷缓存与伴随载荷属于分析器, 不在本模块.
"""

from typing import Iterable
import math

from soptx.backend import backend_manager as bm
from soptx.fem.integrators import LagrangeBoundarySourceIntegrator, SourceIntegrator
from soptx.fem.linear_form import LinearForm
from soptx.fem.load_projection import project_nodal_loads
from soptx.functionspace import TensorFunctionSpace
from soptx.protocols import BodyForce, BoundaryTraction, LineTraction, Load, PointForce
from soptx.typing import TensorLike


def assemble_point_force(space: TensorFunctionSpace, load: PointForce, *,
                         coordinate_offset=None, tolerance: float = 1.0e-10) -> TensorLike:
    """按形函数装配物理集中力, 返回空间自由度排序的载荷向量.

    Parameters
    ----------
    space : TensorFunctionSpace
        连续位移空间. 非节点点力首轮支持轴对齐 Q1 六面体.
    load : PointForce
        物理作用点与合力向量.
    coordinate_offset : TensorLike, optional
        网格局部坐标到物理坐标的平移量.
    tolerance : float, optional
        节点匹配和单元边界定位的绝对容差.

    Returns
    -------
    TensorLike
        满足离散虚功关系的全局载荷向量.

    Raises
    ------
    ValueError
        点力无效, 定位失败或作用点在网格外.
    NotImplementedError
        非节点点力所需的网格或空间类型尚不支持.
    """
    from soptx.mesh import HexahedronMesh

    scalar = space.scalar_space
    mesh = space.mesh
    dim = mesh.geo_dimension()
    if scalar.ctype != 'C' or space.dof_shape != (dim,):
        raise NotImplementedError('集中力装配需要连续 Lagrange 位移向量空间.')
    point_values = tuple(float(x) for x in load.point)
    force_values = tuple(float(x) for x in load.force())
    if (load.dimension != dim or len(point_values) != dim or len(force_values) != dim
            or not all(math.isfinite(x) for x in (*point_values, *force_values))):
        raise ValueError('集中力坐标和合力必须是与空间维数一致的有限向量.')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('集中力定位容差必须为有限正数.')
    coordinates = scalar.interpolation_points()
    context = bm.context(coordinates)
    point = bm.asarray(point_values, **context)
    if coordinate_offset is not None:
        offset = bm.asarray(coordinate_offset, **context)
        if tuple(offset.shape) != (dim,) or not bool(bm.all(bm.isfinite(offset))):
            raise ValueError('coordinate_offset 必须是与空间维数一致的有限向量.')
        point = point - offset
    value = bm.asarray(force_values, **context)
    nodes = bm.nonzero(bm.max(bm.abs(coordinates - point), axis=1) <= tolerance)[0]
    if len(nodes) > 1:
        raise ValueError('集中力命中多个插值节点, 请检查网格或定位容差.')
    if len(nodes) == 1:
        weights = bm.ones((1,), **context)
    else:
        if not isinstance(mesh, HexahedronMesh) or scalar.p != 1:
            raise NotImplementedError('非节点集中力目前仅支持轴对齐 Q1 六面体网格.')
        cell_id, bcs = _locate_box_point(mesh, point, tolerance)
        selection = slice(cell_id, cell_id + 1)
        nodes = scalar.cell_to_dof(index=selection).reshape(-1)
        weights = scalar.basis(bcs, index=selection).reshape(-1)
    components = bm.arange(dim, dtype=bm.int64, device=bm.get_device(coordinates))
    if space.dof_priority:
        dofs = nodes[:, None] + scalar.number_of_global_dofs() * components[None, :]
    else:
        dofs = dim * nodes[:, None] + components[None, :]
    vector = bm.zeros((space.number_of_global_dofs(),), **context)
    return bm.add_at(vector, dofs.reshape(-1), (weights[:, None] * value).reshape(-1))


def _locate_box_point(mesh, point, tolerance):
    """分块定位轴对齐六面体内的点, 共享面和边只选编号最小的单元.

    Returns
    -------
    cell_id : int
        包含作用点的唯一选定单元编号.
    bcs : tuple
        三个参考方向的区间重心坐标.
    """
    coordinates = mesh.entity('node')
    cells = mesh.entity('cell')
    context = bm.context(coordinates)
    # 分块避免大网格定位时分配全部单元的顶点坐标副本.
    for start in range(0, len(cells), 65536):
        vertices = coordinates[cells[start:start + 65536]]
        lower, upper = bm.min(vertices, axis=1), bm.max(vertices, axis=1)
        candidates = bm.nonzero(bm.all((point >= lower - tolerance)
                                      & (point <= upper + tolerance), axis=1))[0]
        if len(candidates) == 0:
            continue
        local = int(candidates[0])
        vertex = vertices[local]
        extent = upper[local] - lower[local]
        if bool(bm.any(extent <= 2 * tolerance)):
            raise ValueError('单元尺寸不足以在指定容差下定位集中力.')
        # 使用六面体公开的顶点拓扑核验盒形几何, 不接受畸变单元的包围盒近似.
        corners = bm.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), **context)
        if not bool(bm.all(bm.abs(vertex - (lower[local] + corners * extent)) <= tolerance)):
            raise NotImplementedError('非节点集中力目前仅支持标准顶点顺序的轴对齐六面体.')
        xi = (point - lower[local]) / extent
        # 仅吸收边界容差内的舍入误差, 防止产生伪内部载荷.
        xi = bm.where(bm.abs(point - lower[local]) <= tolerance, 0.0, xi)
        xi = bm.where(bm.abs(point - upper[local]) <= tolerance, 1.0, xi)
        bcs = tuple(bm.stack((1.0 - xi[d], xi[d])).reshape(1, 2) for d in range(3))
        return start + local, bcs
    raise ValueError('集中力作用点位于网格外或无法定位到单元.')


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
        if isinstance(load, PointForce):
            F = F + assemble_point_force(space, load)
            continue
        if isinstance(load, LineTraction):
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
