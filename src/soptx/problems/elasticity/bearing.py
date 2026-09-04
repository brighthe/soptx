"""二维轴承装置线弹性问题模型."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad

from ._base import axis_aligned_box_corners, validated_domain


class BearingDevice2d:
    """二维轴承装置线弹性与拓扑优化测试模型.

    几何域:
        - 盒形区域: [0, 120] x [0, 40] mm (长宽比 3:1)

    边界条件:
        - 底部全固支 (y = ymin): u_x = u_y = 0
        - 左右侧边 (x = xmin, x = xmax): 自由边界, traction 为 0
        - 顶部边界 (y = ymax): 施加竖直向下均布牵引载荷 t = (0, t_y) = (0, -8e-2) N/mm
    """

    dimension = 2
    boundary_type = "mixed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 40.0),
        *,
        t: float = -8.0e-2,
        E: float = 1.0,
        nu: float = 0.5,
        plane_type: str = "plane_stress",
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._t = float(t)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type

    @property
    def domain(self) -> tuple[float, ...]:
        """计算域边界 ``(xmin, xmax, ymin, ymax)``."""
        return self._domain

    @property
    def t(self) -> float:
        """顶边竖向均布牵引面力强度 (N/mm)."""
        return self._t

    @property
    def E(self) -> float:
        """实体材料 Young 模量."""
        return self._E

    @property
    def nu(self) -> float:
        """实体材料 Poisson 比."""
        return self._nu

    @property
    def plane_type(self) -> str:
        """平面假设类型 ('plane_stress' 或 'plane_strain')."""
        return self._plane_type

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回底部固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 混合形式中弱施加的位移数据 (底部固支齐次位移)."""
        return self.dirichlet_bc(points)

    @cartesian
    def _on_bottom_boundary(self, points: TensorLike) -> TensorLike:
        """标记底边固支边界 ``y=ymin``."""
        y = points[..., 1]
        tolerance = bm.full_like(y, self._eps)
        return bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.domain[2]))),
            tolerance,
        )

    @cartesian
    def _on_top_boundary(self, points: TensorLike) -> TensorLike:
        """标记顶边载荷边界 ``y=ymax``."""
        y = points[..., 1]
        tolerance = bm.full_like(y, self._eps)
        return bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.domain[3]))),
            tolerance,
        )

    @cartesian
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左侧自由边界 ``x=xmin``."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        return bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[0]))),
            tolerance,
        )

    @cartesian
    def _on_right_boundary(self, points: TensorLike) -> TensorLike:
        """标记右侧自由边界 ``x=xmax``."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        return bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[1]))),
            tolerance,
        )

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 LFEM 水平位移的 Dirichlet 边界 (底边固支 ``u_x=0``)."""
        return self._on_bottom_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 LFEM 竖向位移的 Dirichlet 边界 (底边固支 ``u_y=0``)."""
        return self._on_bottom_boundary(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], Callable[[TensorLike], TensorLike]]:
        """返回 LFEM 位移分量的边界标记函数对."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    @cartesian
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记 Hu--Zhang 混合形式中弱施加的位移边界 (底边固支)."""
        return self._on_bottom_boundary(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记 Hu--Zhang 混合形式中的牵引边界 (顶边、左侧边、右侧边)."""
        return bm.logical_or(
            self._on_top_boundary(points),
            bm.logical_or(self._on_left_boundary(points), self._on_right_boundary(points)),
        )

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 角点松弛所需的几何角点坐标."""
        return axis_aligned_box_corners(
            node,
            self.domain,
            self.dimension,
            self._eps,
        )

    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回牵引边界数据. 顶边施加竖向面力 ``t``, 侧边为 0."""
        on_top = self._on_top_boundary(points)
        val = bm.zeros(points.shape, **bm.context(points))
        val = bm.set_at(val, (on_top, 1), self._t)
        return val

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回顶边均布牵引载荷."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )
