"""二维简支桥梁工程基准问题."""

from __future__ import annotations

from typing import Callable, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad

from ._base import validated_domain


class SimplySupportedBridge2d:
    """二维简支桥梁线弹性与拓扑优化测试模型 (博士论文算例 3.2).

    几何域:
        - 盒形区域: [0, 120] x [0, 40] mm (长宽比 3:1)
        - 顶部厚度 ``deck_height`` 的条带 ``ymax - deck_height <= y <= ymax``
          是桥面实体非设计域, 拓扑优化中密度锁定为 1

    边界条件:
        - 左下角节点: 固定铰支 ``u_x = u_y = 0``
        - 右下角节点: 滑动铰支 ``u_y = 0``
        - 其余边界: 自由边界, traction 为 0
        - 顶部边界 (y = ymax): 竖直均布牵引载荷 ``t = (0, t_y)``

    问题定义只包含区域、材料参数、载荷与边界条件; 网格由调用方显式创建。
    被动区通过 :meth:`get_passive_element_mask` 按单元重心判定, 只支持
    单元密度表征。
    """

    dimension = 2
    boundary_type = "mixed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 40.0),
        *,
        t: float = -1.0,
        E: float = 350.0,
        nu: float = 0.3,
        plane_type: str = "plane_stress",
        deck_height: float = 4.0,
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._t = float(t)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type
        height = self._domain[3] - self._domain[2]
        deck_height = float(deck_height)
        if not 0.0 <= deck_height <= height:
            raise ValueError(
                f"deck_height must lie in [0, {height}], received {deck_height}"
            )
        self._deck_height = deck_height

    @property
    def domain(self) -> tuple[float, ...]:
        """计算域边界 ``(xmin, xmax, ymin, ymax)``."""
        return self._domain

    @property
    def t(self) -> float:
        """顶边竖向均布牵引面力强度 (N/mm, 带符号的 y 分量)."""
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

    @property
    def deck_height(self) -> float:
        """桥面实体非设计域的厚度; 0 表示没有被动区."""
        return self._deck_height

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回支座处的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def _on_left_bottom_corner(self, points: TensorLike) -> TensorLike:
        """标记左下角固定铰支节点 ``(xmin, ymin)``."""
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[0]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    @cartesian
    def _on_right_bottom_corner(self, points: TensorLike) -> TensorLike:
        """标记右下角滑动铰支节点 ``(xmax, ymin)``."""
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[1]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
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
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记水平位移的 Dirichlet 自由度 (仅左下角固定铰支 ``u_x=0``)."""
        return self._on_left_bottom_corner(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记竖向位移的 Dirichlet 自由度 (左右两个底角 ``u_y=0``)."""
        return bm.logical_or(
            self._on_left_bottom_corner(points),
            self._on_right_bottom_corner(points),
        )

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], Callable[[TensorLike], TensorLike]]:
        """返回 LFEM 位移分量的边界标记函数对."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回牵引边界数据: 顶边竖向面力 ``t``."""
        on_top = self._on_top_boundary(points)
        val = bm.zeros(points.shape, **bm.context(points))
        val = bm.set_at(val, (on_top, 1), self._t)
        return val

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回顶边均布牵引载荷."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self._on_top_boundary,
                value=self._boundary_traction,
            ),
        )

    def get_passive_element_mask(self, mesh) -> TensorLike:
        """返回桥面被动单元掩码 (形状 ``(NC,)`` 的布尔张量).

        单元重心落在 ``y > ymax - deck_height`` 的条带内即视为桥面单元;
        按几何判定, 不依赖单元编号顺序, 结构化 tri/quad 网格通用。
        """
        barycenter = mesh.entity_barycenter("cell")
        y = barycenter[..., 1]
        threshold = self.domain[3] - self._deck_height
        return bm.greater(y, bm.full_like(y, threshold))
