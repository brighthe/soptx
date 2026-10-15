"""全 Dirichlet 制造解共用的混合形式边界默认实现与二维基类."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike
from soptx.core.numerics import GEOMETRIC_TOL
from soptx.problems.loads import BodyForceLoad


class _AllDisplacementBoundaryHost(Protocol):
    """全位移边界 Mixin 所要求的宿主问题协议."""

    @property
    def domain(self) -> tuple[float, ...]:
        """返回轴对齐盒形区域的边界值."""
        ...

    dimension: int

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


class _AllDirichletElasticity2D(AllDisplacementBoundaryMixin):
    dimension = 2
    plane_type = "plane_strain"
    boundary_type = "dirichlet"

    def __init__(self, domain: Sequence[float]) -> None:
        self._domain = tuple(float(value) for value in domain)
        self._eps = GEOMETRIC_TOL

    @property
    def domain(self) -> tuple[float, ...]:
        return self._domain

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return self.disp_solution(points)

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        domain = self.domain
        return (
            (bm.abs(x - domain[0]) < self._eps)
            | (bm.abs(x - domain[1]) < self._eps)
            | (bm.abs(y - domain[2]) < self._eps)
            | (bm.abs(y - domain[3]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.is_dirichlet_boundary_dof_x(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )
