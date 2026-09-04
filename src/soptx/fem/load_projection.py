"""点力与线载荷到节点自由度向量的公共投影.

本模块是 ``PointForce`` 与 ``LineTraction`` 的唯一离散入口: LFEM 分析器与
子结构条件投影都经由此处, 以保证同一个物理载荷对象在不同求解路径上得到
同一个离散载荷泛函. 两条路径在点力上的差别由 ``mode`` 显式声明, 不再各自
实现一套吸附规则.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.protocols import LineTraction, Load, PointForce

PointForceMode = Literal["exact", "nearest_boundary"]

# 相邻插值节点等距性与共线性的相对判据, 以线段长度为基准
_GEOMETRY_RELATIVE_TOLERANCE = 1.0e-9


def _scatter_node_values(
    vector: Any,
    nodes: Any,
    values: Any,
    dimension: int,
) -> Any:
    """把节点向量值累加到节点优先自由度向量."""
    offsets = bm.arange(
        dimension,
        dtype=bm.int64,
        device=bm.get_device(vector),
    )
    dofs = dimension * nodes[:, None] + offsets[None, :]
    return bm.add_at(
        vector,
        bm.reshape(dofs, (-1,)),
        bm.reshape(values, (-1,)),
    )


def _empty_vector(coordinates: Any, dimension: int) -> Any:
    """按节点坐标构造零初值的节点优先载荷向量."""
    return bm.zeros(
        (dimension * int(coordinates.shape[0]),),
        dtype=bm.float64,
        device=bm.get_device(coordinates),
    )


def _validated_coordinates(node_coordinates: Any, dimension: int) -> Any:
    """校验并返回形状为 ``(N, dimension)`` 的节点坐标."""
    coordinates = bm.asarray(node_coordinates, dtype=bm.float64)
    if len(coordinates.shape) != 2 or int(coordinates.shape[1]) != dimension:
        raise ValueError(
            "node_coordinates 应为形状 (N, dimension) 的二维数组; "
            f"实际为 {tuple(coordinates.shape)}."
        )
    return coordinates


def _validated_point_force(load: PointForce, dimension: int) -> tuple[Any, Any]:
    """校验点力的维数与返回形状, 返回作用点和合力的纯 Python 元组."""
    if int(load.dimension) != dimension:
        raise ValueError("PointForce.dimension 与节点坐标维数不一致.")
    point = tuple(float(value) for value in load.point)
    value = tuple(float(value) for value in load.force())
    if len(point) != dimension or len(value) != dimension:
        raise ValueError(
            "PointForce 的 point 与 force() 必须返回长度 dimension 的向量; "
            f"实际为 {len(point)} 与 {len(value)}."
        )
    return point, value


def _lagrange_basis_at(degree: int, xi: float) -> tuple[float, ...]:
    """返回参考区间 ``[-1, 1]`` 上等距节点的 Lagrange 基函数值.

    节点取 ``xi_i = -1 + 2 i / degree``, 与 fealpy Lagrange 空间在单元边上的
    插值点排布一致. 基函数在纯 Python 浮点上求值: 阶次低、与后端无关, 且避免
    在积分循环里反复构造后端张量.
    """
    nodes = tuple(-1.0 + 2.0 * index / degree for index in range(degree + 1))
    basis = []
    for index, node in enumerate(nodes):
        value = 1.0
        for other_index, other_node in enumerate(nodes):
            if other_index == index:
                continue
            value *= (xi - other_node) / (node - other_node)
        basis.append(value)
    return tuple(basis)


def _gauss_legendre(count: int) -> tuple[tuple[float, float], ...]:
    """返回参考区间 ``[-1, 1]`` 上 ``count`` 点 Gauss--Legendre 求积对."""
    positions, weights = np.polynomial.legendre.leggauss(count)
    return tuple(
        (float(position), float(weight))
        for position, weight in zip(positions, weights)
    )


def project_point_force(
    load: PointForce,
    node_coordinates: Any,
    dimension: int,
    *,
    mode: PointForceMode = "exact",
    domain: Sequence[float] | None = None,
    tolerance: float = 1.0e-10,
) -> Any:
    """把集中力投影为节点优先载荷向量.

    Parameters
    ----------
    mode : {"exact", "nearest_boundary"}
        ``"exact"`` 要求作用点精确落在唯一插值节点上, 否则报错: 位移元的
        点力离散只有这一种无歧义写法. ``"nearest_boundary"`` 允许作用点不在
        节点上, 把合力均分到同一物理边界上距离最近的全部节点; 子结构宏观/
        全尺度投影使用该模式, 必须同时给出 ``domain``. 作用点恰好落在节点上
        时两种模式给出同一个结果.
    domain : Sequence[float], optional
        轴对齐盒形区域边界 ``(x0, x1, y0, y1, ...)``, 仅 ``"nearest_boundary"``
        需要.
    """
    coordinates = _validated_coordinates(node_coordinates, dimension)
    point_values, force_values = _validated_point_force(load, dimension)
    device = bm.get_device(coordinates)
    point = bm.tensor(point_values, dtype=bm.float64, device=device)
    value = bm.tensor(force_values, dtype=bm.float64, device=device)
    vector = _empty_vector(coordinates, dimension)

    if mode == "exact":
        distance = bm.max(bm.abs(coordinates - point[None, :]), axis=1)
        nodes = bm.nonzero(distance <= tolerance)[0]
        if int(nodes.shape[0]) != 1:
            raise ValueError(
                "mode='exact' 要求 PointForce 精确落在唯一插值节点上; "
                f"命中节点数为 {int(nodes.shape[0])}."
            )
        return _scatter_node_values(
            vector,
            nodes,
            bm.reshape(value, (1, dimension)),
            dimension,
        )

    if mode != "nearest_boundary":
        raise ValueError(
            f"不支持的点力投影模式 {mode!r}; 可选 'exact' 或 'nearest_boundary'."
        )

    if domain is None:
        raise ValueError("mode='nearest_boundary' 需要给出 problem.domain.")
    bounds = tuple(float(bound) for bound in domain)
    if len(bounds) != 2 * dimension:
        raise ValueError(
            f"domain 长度应为 {2 * dimension}, 实际为 {len(bounds)}."
        )

    # 作用点落在某个坐标面上时, 候选节点必须也落在同一个坐标面上: 避免把边界
    # 载荷吸附到内部节点上
    scale = max(1.0, *(abs(bound) for bound in bounds))
    face_tolerance = tolerance * scale
    n_nodes = int(coordinates.shape[0])
    candidate = bm.ones((n_nodes,), dtype=bm.bool, device=device)
    for axis in range(dimension):
        lower, upper = bounds[2 * axis], bounds[2 * axis + 1]
        coordinate = point_values[axis]
        if coordinate < lower - face_tolerance or coordinate > upper + face_tolerance:
            raise ValueError("PointForce 作用点位于 problem.domain 之外.")
        on_face = (
            abs(coordinate - lower) <= face_tolerance
            or abs(coordinate - upper) <= face_tolerance
        )
        if on_face:
            candidate = bm.logical_and(
                candidate,
                bm.abs(coordinates[:, axis] - point[axis]) <= face_tolerance,
            )

    distance_squared = bm.sum((coordinates - point[None, :]) ** 2, axis=1)
    infinity = bm.full(
        distance_squared.shape,
        float("inf"),
        **bm.context(distance_squared),
    )
    nearest = bm.min(bm.where(candidate, distance_squared, infinity))
    tie_tolerance = max(
        face_tolerance * face_tolerance,
        1.0e-14 * max(1.0, abs(float(nearest))),
    )
    nodes = bm.nonzero(
        bm.logical_and(
            candidate,
            bm.abs(distance_squared - nearest) <= tie_tolerance,
        )
    )[0]
    count = int(nodes.shape[0])
    if count == 0:
        raise ValueError("PointForce 没有找到可用于投影的边界节点.")
    values = bm.broadcast_to(value[None, :] / count, (count, dimension))
    return _scatter_node_values(vector, nodes, values, dimension)


def project_line_traction(
    load: LineTraction,
    node_coordinates: Any,
    dimension: int,
    *,
    degree: int = 1,
) -> Any:
    """把线载荷投影为节点优先载荷向量.

    命中节点按线方向排序后每 ``degree`` 个区间归为一个 ``degree`` 阶 Lagrange
    线单元, 单元内用 ``degree + 2`` 点 Gauss--Legendre 积分形成一致节点力.
    ``degree`` 必须与提供 ``node_coordinates`` 的位移空间阶次一致: 把高阶空间
    的插值点当作 P1 节点串处理会保持合力、却给出错误的节点力分布.

    命中节点必须构成一条直线, 且每个单元内的插值点等距 —— 这正是 fealpy
    Lagrange 空间在直边上的插值点排布. 不满足时显式报错, 不做几何近似.
    """
    if int(load.dimension) != dimension:
        raise ValueError("LineTraction.dimension 与节点坐标维数不一致.")
    if degree < 1:
        raise ValueError(f"degree 必须为正整数, 实际为 {degree}.")

    coordinates = _validated_coordinates(node_coordinates, dimension)
    device = bm.get_device(coordinates)
    vector = _empty_vector(coordinates, dimension)

    mask = bm.asarray(load.is_load_line(coordinates), dtype=bm.bool)
    if tuple(mask.shape) != (int(coordinates.shape[0]),):
        raise ValueError("LineTraction 区域标记必须返回形状 (N,) 的布尔数组.")
    nodes = bm.nonzero(mask)[0]
    n_points = int(nodes.shape[0])
    if n_points < degree + 1:
        raise ValueError(
            f"degree={degree} 的线载荷至少需要命中 {degree + 1} 个节点; "
            f"实际命中 {n_points} 个."
        )
    if (n_points - 1) % degree != 0:
        raise ValueError(
            f"命中节点数 {n_points} 与 degree={degree} 不相容: "
            "线上的插值点数应为 degree * n_cells + 1."
        )

    line_points = coordinates[nodes]
    spread = bm.max(line_points, axis=0) - bm.min(line_points, axis=0)
    axis = int(bm.argmax(spread))
    if float(spread[axis]) <= 0.0:
        raise ValueError("LineTraction 命中节点不能确定有效线方向.")
    order = bm.argsort(line_points[:, axis])
    nodes = nodes[order]
    line_points = line_points[order]

    quadrature = _gauss_legendre(degree + 2)
    n_cells = (n_points - 1) // degree

    for cell in range(n_cells):
        first = cell * degree
        last = first + degree
        cell_nodes = nodes[first : last + 1]
        cell_points = line_points[first : last + 1]

        start_point = cell_points[0]
        end_point = cell_points[degree]
        delta = end_point - start_point
        length = float(bm.sqrt(bm.sum(delta * delta)))
        if length <= 0.0:
            raise ValueError("LineTraction 中存在长度为零的线单元.")
        tangent = delta / length

        # 单元内插值点必须共线且等距, 否则等参映射的 Jacobian 不是常数,
        # 下面按 length / 2 缩放的积分不成立
        for index in range(1, degree):
            expected = start_point + (index / degree) * delta
            deviation = float(
                bm.max(bm.abs(cell_points[index] - expected))
            )
            if deviation > _GEOMETRY_RELATIVE_TOLERANCE * length:
                raise ValueError(
                    "LineTraction 命中节点在单元内不共线或不等距; "
                    f"第 {cell} 个单元的第 {index} 个插值点偏差 {deviation:.3e}."
                )

        cell_values = bm.zeros(
            (degree + 1, dimension),
            dtype=bm.float64,
            device=device,
        )
        for xi, weight in quadrature:
            basis = _lagrange_basis_at(degree, xi)
            shape = bm.tensor(basis, dtype=bm.float64, device=device)
            quadrature_point = bm.sum(
                shape[:, None] * cell_points,
                axis=0,
            )[None, :]
            traction = bm.asarray(
                load.traction(
                    quadrature_point,
                    tangents=tangent[None, :],
                ),
                dtype=bm.float64,
            )
            if tuple(traction.shape) != (1, dimension):
                raise ValueError(
                    "LineTraction.traction() 必须返回与求值点相同的形状; "
                    f"实际为 {tuple(traction.shape)}."
                )
            cell_values = cell_values + (
                length / 2.0 * weight * shape[:, None] * traction[0][None, :]
            )

        vector = _scatter_node_values(
            vector,
            cell_nodes,
            cell_values,
            dimension,
        )
    return vector


def project_nodal_loads(
    loads: Sequence[Load],
    node_coordinates: Any,
    dimension: int,
    *,
    degree: int = 1,
    point_tolerance: float = 1.0e-10,
    point_mode: PointForceMode = "exact",
    domain: Sequence[float] | None = None,
) -> Any:
    """把 ``PointForce`` 与 ``LineTraction`` 投影到节点优先载荷向量.

    ``degree`` 是提供 ``node_coordinates`` 的位移空间阶次, 决定线载荷按几阶
    Lagrange 线单元形成一致节点力; 调用方必须传入实际阶次而不是沿用默认值.

    ``BodyForce`` 与 ``BoundaryTraction`` 应分别由体积分和边界积分装配, 传入
    本函数会显式报错而不是被静默忽略.
    """
    coordinates = _validated_coordinates(node_coordinates, dimension)
    vector = _empty_vector(coordinates, dimension)

    for load in loads:
        if isinstance(load, PointForce):
            vector = vector + project_point_force(
                load,
                coordinates,
                dimension,
                mode=point_mode,
                domain=domain,
                tolerance=point_tolerance,
            )
        elif isinstance(load, LineTraction):
            vector = vector + project_line_traction(
                load,
                coordinates,
                dimension,
                degree=degree,
            )
        else:
            raise TypeError(
                "project_nodal_loads 只接受 PointForce 或 LineTraction; "
                f"收到 {type(load).__name__}."
            )
    return vector


__all__ = [
    "PointForceMode",
    "project_line_traction",
    "project_nodal_loads",
    "project_point_force",
]
