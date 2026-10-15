"""二维轴承装置工程基准问题."""

from __future__ import annotations

from typing import Callable, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad

from ._base import ElasticityBenchmark


class BearingTraction2d(ElasticityBenchmark):
    """顶边整边受均布压载的二维轴承装置问题.

    底边 ``y = y0`` 完全固支 ``u_x = u_y = 0``; 左右侧边自由, 牵引为零; 顶边 ``y = y1``
    施加竖直均布牵引, 合力 ``P``, 强度 ``P / (x1 - x0)``.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 120, 0, 40)``.
    P : float, optional
        顶边牵引的竖向合力, 负值向下, 默认 ``-9.6`` (强度 ``-8e-2``).
    E : float, optional
        实体材料的 Young 模量, 默认 ``1``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.5``, 面向 Hu--Zhang 混合元的不可压测试;
        位移元应传入可压缩的取值.
    plane_type : str, optional
        平面假设, ``'plane_stress'`` (默认) 或 ``'plane_strain'``.
    """

    dimension = 2

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 40.0),
        *,
        P: float = -9.6,
        E: float = 1.0,
        nu: float = 0.5,
        plane_type: str = "plane_stress",
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=plane_type)

    # ------------------------------------------------------------------ 载荷
    @property
    def traction_intensity(self) -> float:
        """顶边均布牵引的竖向强度 ``P / (x1 - x0)``."""
        return self._P / (self._domain[1] - self._domain[0])

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回底边固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 底边."""
        return self._on_bottom_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 底边."""
        return self._on_bottom_boundary(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], ...]:
        """返回 Lagrange 位移分量的边界标记函数."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    # ------------------------------------------------------------ Hu--Zhang 层
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """返回混合形式中弱施加的位移数据."""
        return self.dirichlet_bc(points)

    @cartesian
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中弱施加位移的边界: 底边."""
        return self._on_bottom_boundary(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中强施加牵引的边界: 顶边与左右侧边."""
        return (
            self._on_top_boundary(points)
            | self._on_x(points, self.domain[0])
            | self._on_x(points, self.domain[1])
        )

    # ------------------------------------------------------------ 边界标记
    def _on_bottom_boundary(self, points: TensorLike) -> TensorLike:
        """标记底边 ``y = y0``."""
        return self._on_y(points, self.domain[2])

    def _on_top_boundary(self, points: TensorLike) -> TensorLike:
        """标记顶边 ``y = y1``."""
        return self._on_y(points, self.domain[3])

    # ---------------------------------------------------------------- 牵引
    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回牵引边界数据: 顶边为常值竖向牵引, 侧边为零."""
        on_top = self._on_top_boundary(points)
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (on_top, 1), self.traction_intensity)

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回顶边均布牵引载荷."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )
