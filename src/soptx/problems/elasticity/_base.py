"""弹性问题数据共享的校验函数与边界默认实现."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


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


class AllDisplacementBoundaryMixin:
    """全 Dirichlet 盒形问题在混合形式下的边界默认实现.

    ``HuZhangMFEMAnalyzer`` 把边界划分为弱施加的位移部分和强施加的
    traction 部分。对整个边界都给定位移的问题, 这个划分是平凡的; 这里把它
    显式写出来, 使这类问题无需每个调用方各写一层适配就满足
    ``MixedBoundaryElasticityProblem`` 协议。

    要求宿主类提供 ``domain``、``dimension`` 和 ``dirichlet_bc``。
    """

    _eps = 1.0e-12

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """返回轴对齐盒形区域的角点坐标.

        一个节点在 *每个* 坐标轴上都落在边界值上时即为角点; 这个判据把二维
        正方形的情形推广到任意维数。
        """
        domain = self.domain
        on_every_axis = None
        for axis in range(self.dimension):
            coordinate = node[:, axis]
            lower, upper = domain[2 * axis], domain[2 * axis + 1]
            on_axis = (
                bm.abs(coordinate - lower) < self._eps
            ) | (bm.abs(coordinate - upper) < self._eps)
            on_every_axis = (
                on_axis if on_every_axis is None else on_every_axis & on_axis
            )
        return node[on_every_axis]

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """整个边界都给定位移."""
        return bm.ones(points.shape[:-1], dtype=bm.bool)

    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """边界上没有任何部分给定 traction."""
        return bm.zeros(points.shape[:-1], dtype=bm.bool)

    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """混合形式下弱施加的位移数据.

        ``HuZhangMFEMAnalyzer`` 用带默认值的 ``getattr`` 查找本成员, 缺失时
        回退到齐次数据——对精确位移在边界上不为零的问题, 这种回退是静默出错。
        复用 ``dirichlet_bc`` 使弱数据保持精确, 并与主形式的强数据完全一致。
        """
        return self.dirichlet_bc(points)

    def traction_bc(self, points: TensorLike) -> TensorLike:
        """拒绝 traction 查询, 而不是编造零 traction 数据.

        ``is_traction_boundary`` 为空, 所以任何分析器在正确的路径上都到不了
        这里。抛异常既让协议要求的成员存在, 又拒绝回答这个问题本身无法回答
        的询问。
        """
        raise NotImplementedError(
            f"{type(self).__name__} prescribes displacement on the whole "
            "boundary and carries no traction data"
        )
