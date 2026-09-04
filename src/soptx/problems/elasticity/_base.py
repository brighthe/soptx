"""弹性问题数据共享的校验函数与边界默认实现."""

from __future__ import annotations

from math import isfinite
from typing import Callable, Protocol, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.problems.loads import BodyForceLoad


def validated_domain(
    domain: Sequence[float],
    dimension: int,
) -> tuple[float, ...]:
    """校验并返回轴对齐盒形区域的边界值."""

    values = tuple(float(value) for value in domain)
    if len(values) != 2 * dimension:
        raise ValueError(
            f"{dimension}D problems require {2 * dimension} domain bounds, "
            f"received {len(values)}"
        )
    for axis in range(dimension):
        lower, upper = values[2 * axis : 2 * axis + 2]
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError(
                f"domain axis {axis} must contain finite lower < upper "
                f"bounds, received ({lower}, {upper})"
            )
    return values


def axis_aligned_box_corners(
    node: TensorLike,
    domain: Sequence[float],
    dimension: int,
    eps: float,
) -> TensorLike:
    """返回轴对齐盒形区域的角点坐标."""
    on_every_axis = bm.ones(node.shape[:-1], dtype=bm.bool)
    for axis in range(dimension):
        coordinate = node[:, axis]
        lower, upper = domain[2 * axis], domain[2 * axis + 1]
        lower_distance = bm.abs(
            bm.subtract(coordinate, bm.full_like(coordinate, lower))
        )
        upper_distance = bm.abs(
            bm.subtract(coordinate, bm.full_like(coordinate, upper))
        )
        tolerance = bm.full_like(coordinate, eps)
        on_lower = bm.less(lower_distance, tolerance)
        on_upper = bm.less(upper_distance, tolerance)
        on_axis = bm.logical_or(on_lower, on_upper)
        on_every_axis = bm.logical_and(on_every_axis, on_axis)
    return node[on_every_axis]

class _AllDisplacementBoundaryHost(Protocol):
    """全位移边界 Mixin 所要求的宿主问题协议."""

    @property
    def domain(self) -> tuple[float, ...]:
        """返回轴对齐盒形区域的边界值."""
        ...

    dimension: int
    _eps: float

    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回位移本质边界数据."""
        ...

    def _body_force(self, points: TensorLike) -> TensorLike:
        """返回连续体力场."""
        ...


class AllDisplacementBoundaryMixin:
    """全 Dirichlet 盒形问题在混合形式下的边界默认实现.

    ``HuZhangMFEMAnalyzer`` 把边界划分为弱施加的位移部分和强施加的
    traction 部分. 对整个边界都给定位移的问题, 这个划分是平凡的; 这里把它
    显式写出来, 使这类问题无需每个调用方各写一层适配就满足
    ``MixedBoundaryElasticityProblem`` 协议.

    宿主类必须满足 ``_AllDisplacementBoundaryHost`` 协议.
    """

    _eps = 1.0e-12

    def loads(
        self: _AllDisplacementBoundaryHost,
    ) -> tuple[BodyForceLoad, ...]:
        """返回全 Dirichlet 制造解的体力载荷."""
        return (
            BodyForceLoad(
                dimension=self.dimension,
                value=self._body_force,
            ),
        )

    def mark_corners(
        self: _AllDisplacementBoundaryHost,
        node: TensorLike,
    ) -> TensorLike:
        """返回轴对齐盒形区域的角点坐标."""
        return axis_aligned_box_corners(
            node,
            self.domain,
            self.dimension,
            self._eps,
        )

    def is_displacement_boundary(
        self: _AllDisplacementBoundaryHost,
        points: TensorLike,
    ) -> TensorLike:
        """整个边界都给定位移."""
        return bm.ones(points.shape[:-1], dtype=bm.bool)

    def is_traction_boundary(
        self: _AllDisplacementBoundaryHost,
        points: TensorLike,
    ) -> TensorLike:
        """边界上没有任何部分给定 traction."""
        return bm.zeros(points.shape[:-1], dtype=bm.bool)

    def displacement_bc(
        self: _AllDisplacementBoundaryHost,
        points: TensorLike,
    ) -> TensorLike:
        """返回混合形式下弱施加的位移数据.

        ``HuZhangMFEMAnalyzer`` 用带默认值的 ``getattr`` 查找本成员, 缺失时
        回退到齐次数据. 对精确位移在边界上不为零的问题, 这种回退会静默出错.
        复用 ``dirichlet_bc`` 使弱数据保持精确, 并与主形式的强数据完全一致.
        """
        return self.dirichlet_bc(points)
