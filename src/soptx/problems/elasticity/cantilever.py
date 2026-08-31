"""二维悬臂梁线弹性与局部应力约束拓扑优化测试模型."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from ._base import axis_aligned_box_corners, validated_domain


class CantileverCorner2d:
    """右下角受集中载荷的二维悬臂梁设计域模型.

    默认参数对应博士论文算例 6.1: ``160 mm x 100 mm`` 矩形域,
    左端全固支, 右下角施加竖直向下的单位集中载荷.
    """

    dimension = 2
    boundary_type = "mixed"
    load_type = "concentrated"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 160.0, 0.0, 100.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
        plane_type: str = "plane_stress",
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._P = float(P)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type

    @property
    def domain(self) -> tuple[float, ...]:
        """完整计算域 ``(xmin, xmax, ymin, ymax)``."""
        return self._domain

    @property
    def P(self) -> float:
        """右下角承受的竖向集中力."""
        return self._P

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
        """平面假设类型."""
        return self._plane_type

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """返回零体力."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左端固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支边界 ``x=xmin``."""
        return bm.abs(points[..., 0] - self.domain[0]) < self._eps

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable]:
        """返回两个位移分量共享的左端边界标记."""
        return (self._on_left_boundary, self._on_left_boundary)

    @cartesian
    def _concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """返回右下角沿竖直方向施加的集中力值."""
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P)

    def concentrate_load_bc(self) -> list[Callable]:
        """返回集中力值函数列表."""
        return [self._concentrate_load_bc]

    @cartesian
    def is_concentrate_load_boundary_dof(
        self,
        points: TensorLike,
    ) -> TensorLike:
        """标记右下角节点 ``(xmax, ymin)``."""
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[1]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    def is_concentrate_load_boundary(self) -> list[Callable]:
        """返回与集中力值函数一一对应的节点标记."""
        return [self.is_concentrate_load_boundary_dof]


class CantileverMiddle2d:
    """右端中点局部受载的二维悬臂梁设计域模型.

    几何域:
        - 盒形区域: [0, 80] x [0, 40] mm (长宽比 2:1)

    边界条件:
        - 左侧边界全固支 (x = xmin): u_x = u_y = 0
        - 顶底边界 (y = ymin, y = ymax): 自由边界, traction 为 0
        - 右侧中点局部区间 (x = xmax, y in [18, 22] mm):
          施加竖直向下均布外力 P = -100 N (或指定载荷强度)

    支持向 ``traction`` 参数注入连续载荷函数 (例如 P1TraceLoad) 实现位移法与混合法的受控对齐.
    """

    dimension = 2
    boundary_type = "mixed"
    load_type = "distributed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 80.0, 0.0, 40.0),
        *,
        P: float = -100.0,
        load_width: float = 4.0,
        E: float = 70000.0,
        nu: float = 0.25,
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
        """局部受载的竖向合力 (N)."""
        return self._P

    @property
    def load_width(self) -> float:
        """右侧中点局部受载区间长度 (mm)."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """右端面上受载区间的起止纵坐标 ``(ymin_patch, ymax_patch)``."""
        y_mid = (self._domain[2] + self._domain[3]) / 2.0
        half_width = self._load_width / 2.0
        return (y_mid - half_width, y_mid + half_width)

    @property
    def traction_level(self) -> float:
        """受载边所在的横坐标, 即右边界的 ``x``."""
        return self._domain[1]

    @property
    def traction_intensity(self) -> float:
        """受载区间内的常值竖向牵引强度 (N/mm)."""
        return self._P / self._load_width

    @property
    def E(self) -> float:
        """实体材料的 Young 模量 (MPa)."""
        return self._E

    @property
    def nu(self) -> float:
        """实体材料的 Poisson 比."""
        return self._nu

    @property
    def plane_type(self) -> str:
        """平面假设类型 ('plane_stress' 或 'plane_strain')."""
        return self._plane_type

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """返回零体力."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左侧固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 混合形式中弱施加的位移数据 (左端全固支齐次位移)."""
        return self.dirichlet_bc(points)

    @cartesian
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支边界 ``x=xmin``."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        return bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[0]))),
            tolerance,
        )

    @cartesian
    def _on_right_boundary(self, points: TensorLike) -> TensorLike:
        """标记右侧边界 ``x=xmax``."""
        x = points[..., 0]
        tolerance = bm.full_like(x, self._eps)
        return bm.less(
            bm.abs(bm.subtract(x, bm.full_like(x, self.domain[1]))),
            tolerance,
        )

    @cartesian
    def _on_top_boundary(self, points: TensorLike) -> TensorLike:
        """标记顶边自由边界 ``y=ymax``."""
        y = points[..., 1]
        tolerance = bm.full_like(y, self._eps)
        return bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.domain[3]))),
            tolerance,
        )

    @cartesian
    def _on_bottom_boundary(self, points: TensorLike) -> TensorLike:
        """标记底边自由边界 ``y=ymin``."""
        y = points[..., 1]
        tolerance = bm.full_like(y, self._eps)
        return bm.less(
            bm.abs(bm.subtract(y, bm.full_like(y, self.domain[2]))),
            tolerance,
        )

    @cartesian
    def _in_traction_patch(self, points: TensorLike) -> TensorLike:
        """标记右端中点加载贴片区域 ``y in [y_mid - l/2, y_mid + l/2]``."""
        on_right = self._on_right_boundary(points)
        y = points[..., 1]
        y_min_patch, y_max_patch = self.traction_patch
        tol = self._eps
        in_y = bm.logical_and(
            bm.greater_equal(y, bm.full_like(y, y_min_patch - tol)),
            bm.less_equal(y, bm.full_like(y, y_max_patch + tol)),
        )
        return bm.logical_and(on_right, in_y)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记左端固支的横向位移自由度."""
        return self._on_left_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记左端固支的竖向位移自由度."""
        return self._on_left_boundary(points)

    def is_dirichlet_boundary(
        self, points: Optional[TensorLike] = None
    ) -> Union[
        tuple[TensorLike, TensorLike],
        tuple[Callable[[TensorLike], TensorLike], Callable[[TensorLike], TensorLike]],
    ]:
        """返回 Lagrange 位移分量的边界标记函数或标记张量."""
        if points is None:
            return (
                self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y,
            )
        return (
            self.is_dirichlet_boundary_dof_x(points),
            self.is_dirichlet_boundary_dof_y(points),
        )

    @cartesian
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """Hu--Zhang 混合有限元的弱位移边界条件标记 (左边界)."""
        return self._on_left_boundary(points)

    def is_neumann_boundary(
        self, points: Optional[TensorLike] = None
    ) -> Union[TensorLike, Callable[[TensorLike], TensorLike]]:
        """返回或计算 Lagrange 形式的 Neumann 边界标记 (右侧边界)."""
        if points is None:
            return self.is_traction_boundary
        return self.is_traction_boundary(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """Hu--Zhang 混合有限元的本质牵引边界条件标记 (右、顶、底边界)."""
        on_right = self._on_right_boundary(points)
        on_top = self._on_top_boundary(points)
        on_bottom = self._on_bottom_boundary(points)
        return bm.logical_or(on_right, bm.logical_or(on_top, on_bottom))

    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """Lagrange 位移元的外力面力数据."""
        if self._traction is not None:
            return self._traction(points)
        return self._step_traction(points)

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """Hu--Zhang 混合有限元的本质牵引面力数据."""
        if self._traction is not None:
            return self._traction(points)
        return self._step_traction(points)

    @cartesian
    def _step_traction(self, points: TensorLike) -> TensorLike:
        """右端局部加载贴片内的常值均布牵引力 (向下为负).

        点集按边 (面) 成批传入 ``(..., NP, GD)`` 时做整边选取: 以边中心
        是否落在贴片内决定整条边取常值强度 ``P/load_width`` 还是全零.
        贴片端点与网格节点对齐时 (本仓库全部算例如此), 插值/积分后的
        合力严格等于 ``P``; 逐点阶跃语义则会在贴片端点顶点处向相邻边
        泄漏二次插值尾巴, 使 Hu--Zhang 本质边界的有效合力偏大
        ``2|t|h/6`` (80x40 基准算例约 +5.6%). 纯点集 ``(NP, GD)`` 输入
        退化为逐点判断.
        """
        result = bm.zeros(points.shape, **bm.context(points))
        if points.ndim >= 3:
            centers = bm.mean(points, axis=-2)
            edge_mask = self._in_traction_patch(centers)
            mask = bm.broadcast_to(edge_mask[..., None], points.shape[:-1])
        else:
            mask = self._in_traction_patch(points)
        result[mask, 1] = self.traction_intensity
        return result

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """返回 Hu--Zhang 角点松弛所需的几何角点布尔标记."""
        return axis_aligned_box_corners(
            node,
            self.domain,
            self.dimension,
            self._eps,
        )

    def corner_points(self) -> tuple[tuple[float, float], ...]:
        """返回矩形设计域的四个角点坐标."""
        return axis_aligned_box_corners(
            None,
            self.domain,
            self.dimension,
            self._eps,
        )


class CantileverRightBottomEdge3d:
    """右端底边受载的三维悬臂梁设计域模型.

    左端面 ``x=xmin`` 完全固支. 右端底边 ``x=xmax, y=ymin`` 上的网格节点
    共同承受沿负 ``y`` 方向的总力 ``P``. ``LagrangeFEMAnalyzer`` 会将总力
    在被标记节点之间均匀分配.
    """

    dimension = 3
    boundary_type = "mixed"
    load_type = "concentrated"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 60.0, 0.0, 20.0, 0.0, 4.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
        plane_type: str = "3D",
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        self._P = float(P)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type

    @property
    def domain(self) -> tuple[float, ...]:
        """完整计算域 ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
        return self._domain

    @property
    def P(self) -> float:
        """右端底边承受的竖向总力."""
        return self._P

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
        """三维本构假设标识."""
        return self._plane_type

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """返回零体力."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左端固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支面 ``x=xmin``."""
        return bm.abs(points[..., 0] - self.domain[0]) < self._eps

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable, Callable]:
        """返回三个平动分量共享的左端面边界标记."""
        return (
            self._on_left_boundary,
            self._on_left_boundary,
            self._on_left_boundary,
        )

    @cartesian
    def _concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """返回沿负 ``y`` 方向的总力值."""
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P)

    def concentrate_load_bc(self) -> list[Callable]:
        """返回右端底边的总力值函数."""
        return [self._concentrate_load_bc]

    @cartesian
    def is_concentrate_load_boundary_dof(
        self,
        points: TensorLike,
    ) -> TensorLike:
        """标记 ``x=xmax, y=ymin`` 的整条网格边."""
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[1]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    def is_concentrate_load_boundary(self) -> list[Callable]:
        """返回与总力值函数一一对应的边界标记."""
        return [self.is_concentrate_load_boundary_dof]
