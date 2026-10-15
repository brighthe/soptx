"""二维简支桥梁工程基准问题."""

from __future__ import annotations

from typing import Callable, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad

from ._base import ElasticityBenchmark


class BridgeTraction2d(ElasticityBenchmark):
    """顶边整边受均布压载的二维简支桥梁问题.

    左下角 ``(x0, y0)`` 固定铰支 ``u_x = u_y = 0``, 右下角 ``(x1, y0)`` 滑动铰支
    ``u_y = 0``; 顶边 ``y = y1`` 施加竖直均布牵引, 合力 ``P``, 强度 ``P / (x1 - x0)``;
    其余边界自由. 顶部厚 ``deck_height`` 的条带是桥面实体非设计域, 由
    :meth:`get_passive_element_mask` 按单元重心给出, 只支持单元密度表征.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 120, 0, 40)``.
    P : float, optional
        顶边牵引的竖向合力, 负值向下, 默认 ``-120`` (强度 ``-1``).
    deck_height : float, optional
        桥面非设计域的厚度, 默认 ``4``; ``0`` 表示没有被动区.
    E : float, optional
        实体材料的 Young 模量, 默认 ``350``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.3``.
    plane_type : str, optional
        平面假设, ``'plane_stress'`` (默认) 或 ``'plane_strain'``.

    Raises
    ------
    ValueError
        ``deck_height`` 不在 ``[0, y1 - y0]`` 内.
    """

    dimension = 2

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 40.0),
        *,
        P: float = -120.0,
        deck_height: float = 4.0,
        E: float = 350.0,
        nu: float = 0.3,
        plane_type: str = "plane_stress",
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=plane_type)
        self._deck_height = float(deck_height)
        height = self._domain[3] - self._domain[2]
        if not 0.0 <= self._deck_height <= height:
            raise ValueError(
                f"deck_height 必须落在 [0, {height}] 内, 实际为 {self._deck_height}."
            )

    # ------------------------------------------------------------------ 几何
    @property
    def deck_height(self) -> float:
        """桥面实体非设计域的厚度; ``0`` 表示没有被动区."""
        return self._deck_height

    # ------------------------------------------------------------------ 载荷
    @property
    def traction_intensity(self) -> float:
        """顶边均布牵引的竖向强度 ``P / (x1 - x0)``."""
        return self._P / (self._domain[1] - self._domain[0])

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回支座处的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左下角."""
        return self._on_bottom_corner(points, self.domain[0])

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 左下角与右下角."""
        return (
            self._on_bottom_corner(points, self.domain[0])
            | self._on_bottom_corner(points, self.domain[1])
        )

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], ...]:
        """返回 Lagrange 位移分量的边界标记函数."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    # ------------------------------------------------------------ 边界标记
    def _on_bottom_corner(self, points: TensorLike, x_value: float) -> TensorLike:
        """标记底边 ``y = y0`` 上 ``x = x_value`` 的角点."""
        return self._on_x(points, x_value) & self._on_y(points, self.domain[2])

    def _on_top_boundary(self, points: TensorLike) -> TensorLike:
        """标记顶边 ``y = y1``."""
        return self._on_y(points, self.domain[3])

    # ---------------------------------------------------------------- 牵引
    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回顶边的常值竖向牵引."""
        on_top = self._on_top_boundary(points)
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (on_top, 1), self.traction_intensity)

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回顶边均布牵引载荷."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self._on_top_boundary,
                value=self._boundary_traction,
            ),
        )

    # ------------------------------------------------------------ 非设计区
    def get_passive_element_mask(self, mesh) -> TensorLike:
        """返回桥面被动单元掩码, 形状 ``(NC,)`` 的布尔张量.

        单元重心落在 ``y > y1 - deck_height`` 的条带内即视为桥面单元; 按几何判定,
        不依赖单元编号顺序, 结构化 tri/quad 网格通用.
        """
        barycenter = mesh.entity_barycenter("cell")
        y = barycenter[..., 1]
        threshold = self.domain[3] - self._deck_height
        return bm.greater(y, bm.full_like(y, threshold))
