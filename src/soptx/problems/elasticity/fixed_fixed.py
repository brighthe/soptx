"""两端固支梁全域工程基准问题."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from ._base import axis_aligned_box_corners, validated_domain


class FixedFixedBeamCenterLoad2d:
    """底部中点局部牵引受载的两端固支梁全域问题.

    参数与博士论文第五章的历史 Hu--Zhang 算例一致: 计算域为
    ``160 mm x 20 mm``, 左右端完全固支, 底部中点局部牵引的合力为
    ``P=-3 N``. ``load_width`` 是局部牵引区间长度, 默认值为 ``1 mm``.

    该类只描述唯一的物理问题. Lagrange 位移元通过 Neumann 边界弱施加
    牵引, Hu--Zhang 混合元通过应力自由度强施加同一牵引. 两条离散路径
    共享完全相同的区域、材料参数、位移边界和牵引函数.

    默认牵引在贴片边缘处不连续, 弱施加与强施加各自的离散误差不同, 两条
    路径看到的载荷合力也就不同. 需要做受控比较时, 用 ``traction`` 注入
    一个等价的连续牵引 (例如 ``soptx.fem.project_patch_traction_to_p1_trace``
    投影出来的 P1 迹载荷): 几何、材料与边界标记全部不变, 只替换牵引函数.
    """

    dimension = 2
    boundary_type = "mixed"
    load_type = "distributed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 160.0, 0.0, 20.0),
        *,
        P: float = -3.0,
        load_width: float = 1.0,
        E: float = 30.0,
        nu: float = 0.4,
        plane_type: str = "plane_stress",
        traction: Optional[Callable[[TensorLike], TensorLike]] = None,
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._P = float(P)
        self._load_width = float(load_width)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type
        self._traction = traction
        if self._load_width <= 0.0:
            raise ValueError("load_width 必须为正数.")

    @property
    def domain(self) -> tuple[float, ...]:
        """完整计算域 ``(xmin, xmax, ymin, ymax)``."""
        return self._domain

    @property
    def P(self) -> float:
        """局部牵引的竖向合力."""
        return self._P

    @property
    def load_width(self) -> float:
        """底部中点局部牵引区间长度."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """底边上受载区间的起止横坐标."""
        x_mid = (self._domain[0] + self._domain[1]) / 2.0
        half_width = self._load_width / 2.0
        return (x_mid - half_width, x_mid + half_width)

    @property
    def traction_level(self) -> float:
        """受载边所在的纵坐标, 即底边的 ``y``."""
        return self._domain[2]

    @property
    def traction_intensity(self) -> float:
        """受载区间内的常值竖向牵引强度."""
        return self._P / self._load_width

    @property
    def E(self) -> float:
        """实体材料的 Young 模量."""
        return self._E

    @property
    def nu(self) -> float:
        """实体材料的 Poisson 比."""
        return self._nu

    @property
    def plane_type(self) -> str:
        """二维本构假设类型."""
        return self._plane_type

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """返回零体力."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左右固支端的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 混合形式中弱施加的位移数据."""
        return self.dirichlet_bc(points)

    @cartesian
    def _is_left_or_right_boundary(self, points: TensorLike) -> TensorLike:
        """标记左右两条垂直边界."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        on_left = bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[0]))),
            tolerance,
        )
        on_right = bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[1]))),
            tolerance,
        )
        return bm.logical_or(on_left, on_right)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记左右固支端的水平位移自由度."""
        return self._is_left_or_right_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记左右固支端的竖向位移自由度."""
        return self._is_left_or_right_boundary(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], Callable[[TensorLike], TensorLike]]:
        """返回 Lagrange 位移分量的边界标记函数."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    @cartesian
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记 Hu--Zhang 混合形式中的位移边界."""
        return self._is_left_or_right_boundary(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记 Hu--Zhang 混合形式中的上、下牵引边界."""
        x, y = points[..., 0], points[..., 1]
        tolerance = bm.full_like(y, self._eps)
        on_bottom = bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.domain[2]))),
            tolerance,
        )
        on_top = bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.domain[3]))),
            tolerance,
        )
        on_horizontal = bm.logical_or(on_bottom, on_top)
        return bm.logical_and(
            on_horizontal,
            bm.logical_not(self._is_left_or_right_boundary(points)),
        )

    def is_neumann_boundary(self) -> Callable[[TensorLike], TensorLike]:
        """返回 Lagrange 形式的 Neumann 边界标记函数."""
        return self.is_traction_boundary

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 角点松弛所需的几何角点."""
        return axis_aligned_box_corners(
            node,
            self.domain,
            self.dimension,
            self._eps,
        )

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """返回上、下边界的牵引数据.

        顶边始终为零牵引. 底边中点长度 ``load_width`` 的区间内施加常值
        ``P / load_width`` 的竖向牵引, 其余位置为零. 因此该局部牵引的
        连续合力严格等于 ``P``.

        构造时若注入了 ``traction``, 则直接转发给它.
        """
        if self._traction is not None:
            return self._traction(points)

        x, y = points[..., 0], points[..., 1]
        left, right = self.traction_patch
        x_mid = (left + right) / 2.0
        half_width = self._load_width / 2.0
        tolerance = bm.full_like(x, self._eps)
        on_bottom = bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.traction_level))),
            bm.full_like(y, self._eps),
        )
        within_patch = bm.less_equal(
            bm.abs(bm.subtract(x, bm.full_like(x, x_mid))),
            bm.add(bm.full_like(x, half_width), tolerance),
        )
        in_patch = bm.logical_and(on_bottom, within_patch)
        traction = bm.zeros(points.shape, **bm.context(points))
        traction = bm.set_at(traction, (..., 1), self.traction_intensity)
        return bm.where(bm.expand_dims(in_patch, axis=-1), traction, bm.zeros_like(traction))

    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """返回 Lagrange 形式中弱施加的同一局部牵引."""
        return self.traction_bc(points)