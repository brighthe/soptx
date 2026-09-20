"""两端固支梁全域工程基准问题."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad, PointForceLoad

from ._base import axis_aligned_box_corners, validated_domain


class FixedFixedBeamCenterLoad2d:
    """底部中点局部牵引受载的两端固支梁全域问题.

    参数与博士论文第五章的历史 Hu--Zhang 算例一致: 计算域为
    ``160 mm x 20 mm``, 左右端完全固支, 底部中点局部牵引的合力为
    ``P=-3 N``. ``load_width`` 是局部牵引区间长度, 默认值为 ``1 mm``.

    该类只描述唯一的物理问题. Lagrange 位移元通过 Neumann 边界弱施加
    牵引, Hu--Zhang 混合元通过应力自由度强施加同一牵引. 两条离散路径
    共享完全相同的区域、材料参数、位移边界和牵引函数.

    默认牵引在载荷区边缘处不连续, 弱施加与强施加各自的离散误差不同, 两条
    路径看到的载荷合力也就不同. 需要做受控比较时, 用 ``traction`` 注入
    一个等价的连续牵引 (例如 ``soptx.fem.project_patch_traction_to_p1_trace``
    投影出来的 P1 迹载荷): 几何、材料与边界标记全部不变, 只替换牵引函数.
    """

    # point_force=True 时通过 loads 返回点力, load_width 不参与载荷计算.
    dimension = 2
    boundary_type = "mixed"
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
        point_force: bool = False,
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._P = float(P)
        self._load_width = float(load_width)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type
        self._traction = traction
        self._point_force = bool(point_force)
        if self._point_force and traction is not None:
            raise ValueError("点力与非零牵引输入不可同时指定.")
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

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 角点松弛所需的几何角点."""
        return axis_aligned_box_corners(
            node,
            self.domain,
            self.dimension,
            self._eps,
        )

    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
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

    def loads(self) -> tuple[BoundaryTractionLoad | PointForceLoad, ...]:
        """返回底边局部牵引, 或中点合力为 P 的节点集中力."""
        if self._point_force:
            return (PointForceLoad(
                point=((self.domain[0] + self.domain[1]) / 2, self.domain[2]),
                vector=(0.0, self.P),
            ),)
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )


class FixedFixedBeamHalfDomain2d:
    """两端固支梁中点受载问题的左半域对称降维问题.

    完整域 ``160 mm x 20 mm`` 关于竖直中线 ``x=80 mm`` 对称. 本类只取左半域
    ``[0, 80] x [0, 20]`` 离散求解: 左端 ``x=0`` 完全固支, 对称面 ``x=80``
    施加对称约束 (法向位移 ``u_x=0`` 与切向牵引 ``sigma_xy=0``), 底部对称面
    底端施加局部牵引. 完整域合力 ``P`` 由左右两半对称分担, 故左半域合力为
    ``P/2``, 左半域柔顺度为完整域的一半, 报告完整结构柔顺度时应乘以 2.

    参数与 ``FixedFixedBeamCenterLoad2d`` 一致, 但 ``domain`` 默认取左半域.
    与完整域模型一样, 受控比较时用 ``traction`` 注入 ``project_patch_traction_to_p1_trace``
    投影出的 P1 迹载荷, 消除强施加与弱积分的几何不对齐误差.
    """

    # point_force=True 时通过 loads 返回点力, load_width 不参与载荷计算.
    dimension = 2
    boundary_type = "mixed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 80.0, 0.0, 20.0),
        *,
        P: float = -3.0,
        load_width: float = 1.0,
        E: float = 30.0,
        nu: float = 0.4,
        plane_type: str = "plane_stress",
        traction: Optional[Callable[[TensorLike], TensorLike]] = None,
        point_force: bool = False,
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._P = float(P)
        self._load_width = float(load_width)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type
        self._traction = traction
        self._point_force = bool(point_force)
        if self._point_force and traction is not None:
            raise ValueError("点力与非零牵引输入不可同时指定.")
        if self._load_width <= 0.0:
            raise ValueError("load_width 必须为正数.")
        half_width = self._load_width / 2.0
        if self._domain[1] - half_width < self._domain[0]:
            raise ValueError("载荷区左端超出左半域, 请检查 load_width 与 domain.")

    @property
    def domain(self) -> tuple[float, ...]:
        """左半计算域 ``(xmin, xmax, ymin, ymax)``, 对称面位于 ``x=xmax``."""
        return self._domain

    @property
    def P(self) -> float:
        """完整域局部牵引的竖向合力."""
        return self._P

    @property
    def load_width(self) -> float:
        """完整域底部中点局部牵引的名义区间长度."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """左半域内受载区间的起止横坐标.

        载荷名义区间 ``[x_load - load_width/2, x_load + load_width/2]`` 关于
        对称面 ``x_load = domain[1]`` 对称, 左半域只保留其左半边, 故区间宽
        ``load_width/2``, 合力自动为完整域的一半 ``P/2``.
        """
        x_load = self._domain[1]
        half_width = self._load_width / 2.0
        return (x_load - half_width, x_load)

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
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回固支端与对称面的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 混合形式中弱施加的位移数据."""
        return self.dirichlet_bc(points)

    @cartesian
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端垂直固支边界 ``x=xmin``."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        return bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[0]))),
            tolerance,
        )

    @cartesian
    def _on_symmetry_plane(self, points: TensorLike) -> TensorLike:
        """标记对称面 ``x=xmax``."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        return bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[1]))),
            tolerance,
        )

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记水平位移自由度的 Dirichlet 边界: 左端固支与对称面 ``u_x=0``."""
        return bm.logical_or(
            self._on_left_boundary(points),
            self._on_symmetry_plane(points),
        )

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记竖向位移自由度的 Dirichlet 边界: 仅左端固支 ``u_y=0``."""
        return self._on_left_boundary(points)

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
        """标记 Hu--Zhang 混合形式中的位移边界: 仅左端固支."""
        return self._on_left_boundary(points)

    @cartesian
    def is_symmetry_boundary(self, points: TensorLike) -> TensorLike:
        """标记 Hu--Zhang 混合形式中的对称面 ``x=xmax``.

        对称面上切向牵引 ``sigma_nt=0`` 需强施加, 法向牵引 ``sigma_nn`` 自由;
        法向位移 ``u_n=0`` 在切向牵引固定后自然满足, 位移边界项贡献为零.
        """
        return self._on_symmetry_plane(points)

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
        not_left = bm.logical_not(self._on_left_boundary(points))
        not_symmetry = bm.logical_not(self._on_symmetry_plane(points))
        return bm.logical_and(on_horizontal, bm.logical_and(not_left, not_symmetry))

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 角点松弛所需的几何角点."""
        return axis_aligned_box_corners(
            node,
            self.domain,
            self.dimension,
            self._eps,
        )

    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回上、下边界的牵引数据.

        底边对称面底端 ``x=xmax`` 附近施加常值竖向牵引 ``P / load_width``,
        其余位置为零; 顶边始终为零牵引. 载荷名义区间关于对称面对称, 左半域
        只保留其左半边, 合力为完整域的一半 ``P/2``. 构造时若注入了
        ``traction``, 则直接转发给它.
        """
        if self._traction is not None:
            return self._traction(points)

        x, y = points[..., 0], points[..., 1]
        x_load = self._domain[1]
        half_width = self._load_width / 2.0
        tolerance = bm.full_like(x, self._eps)
        on_bottom = bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.traction_level))),
            bm.full_like(y, self._eps),
        )
        within_patch = bm.less_equal(
            bm.abs(bm.subtract(x, bm.full_like(x, x_load))),
            bm.add(bm.full_like(x, half_width), tolerance),
        )
        in_patch = bm.logical_and(on_bottom, within_patch)
        traction = bm.zeros(points.shape, **bm.context(points))
        traction = bm.set_at(traction, (..., 1), self.traction_intensity)
        return bm.where(bm.expand_dims(in_patch, axis=-1), traction, bm.zeros_like(traction))

    def loads(self) -> tuple[BoundaryTractionLoad | PointForceLoad, ...]:
        """返回半域局部牵引, 或对称面底端合力为 P/2 的节点集中力."""
        if self._point_force:
            return (PointForceLoad(
                point=(self.domain[1], self.domain[2]),
                vector=(0.0, self.P / 2),
            ),)
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )
