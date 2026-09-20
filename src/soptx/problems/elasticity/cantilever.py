"""二维悬臂梁线弹性与局部应力约束拓扑优化测试模型."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from soptx.problems.loads import (
    BoundaryTractionLoad,
    LineTractionLoad,
    PointForceLoad,
)

from ._base import axis_aligned_box_corners, validated_domain


class CantileverCorner2d:
    """右下角受集中载荷的二维悬臂梁设计域模型.

    默认参数对应博士论文算例 6.1: ``160 mm x 100 mm`` 矩形域,
    左端全固支, 右下角施加竖直向下的单位集中载荷.
    """

    dimension = 2
    boundary_type = "mixed"
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

    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回右下角的集中点力."""
        return (
            PointForceLoad(
                point=(self.domain[1], self.domain[2]),
                vector=(0.0, self.P),
            ),
        )


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
        # 实体保留单元掩码由装配层按算例配置注入 (见 set_passive_element_mask),
        # 模型本身不含半径参数: 垫片尺寸是数值处置的选择, 不是物理问题的属性.
        self._passive_element_mask: Optional[TensorLike] = None

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
    def traction_patch_endpoints(self) -> tuple[tuple[float, float], ...]:
        """受载区间两个端点的坐标 ``((x, y_min_patch), (x, y_max_patch))``.

        右端面上的牵引在这两点处发生跳变, 是混合边界间断点, 其应力奇异性不能
        由载荷分布化消除, 也不能由密度设计消除. 局部应力豁免以这两点为中心,
        半径由算例配置给定, 掩码构造见 ``soptx.topology.constraints.exemption``.
        """
        y_min_patch, y_max_patch = self.traction_patch
        level = self.traction_level
        return ((level, y_min_patch), (level, y_max_patch))

    @property
    def traction_level(self) -> float:
        """受载边所在的横坐标, 即右边界的 ``x``."""
        return self._domain[1]

    @property
    def clamped_corner_points(self) -> tuple[tuple[float, float], ...]:
        """固支边两端角点的坐标 ``((x_left, y_min), (x_left, y_max))``.

        左端面施加位移边界条件, 上下两边为自由边, 故这两点是 Dirichlet--Neumann
        混合边界角点. 该处应力按 ``r^(lambda - 1)`` 奇异, ``lambda`` 为 Williams
        特征值, 只依赖楔角与泊松比 (与载荷大小和分布方式无关): 直角楔在平面应力
        ``nu = 0.25`` 时 ``lambda = 0.78107``, 即 ``sigma ~ r^(-0.2189)``.

        与 ``traction_patch_endpoints`` 的区别在于奇异性的来源: 牵引间断端点由
        数据的集中产生, 可由载荷分布化削弱 (集中力改为分布牵引后退化为对数型);
        固支角点由边界条件类型的改变产生, 数据再光滑也不消失, 因此没有"分布化"
        这一步可做, 局部应力豁免与实体保留是仅有的两步处置. 半径由算例配置给定,
        且不应沿用载荷侧半径 —— 幂律奇点的污染区比对数型宽.

        Notes
        -----
        本属性服务于密度侧的豁免掩码, 与 ``corner_points`` 无关: 后者返回计算域
        的四个角点, 供 Hu--Zhang 应力空间的顶点自由度松弛使用, 作用在离散自由度
        上而非单元密度上.
        """
        x_left = self._domain[0]
        return ((x_left, self._domain[2]), (x_left, self._domain[3]))

    def set_passive_element_mask(self, mask: Optional[TensorLike]) -> None:
        """注入实体保留 (passive solid) 单元掩码.

        引入垫片的密度不是设计结果而是硬约束: 这些单元的物理密度被钉为 1,
        且不参与设计更新. 掩码按单元几何判定, 由装配层用 ``build_exemption_mask``
        以 ``traction_patch_endpoints`` (载荷侧) 与 ``clamped_corner_points``
        (支撑侧) 为中心分别构造后取并集, 两侧半径各自独立;
        与应力约束豁免共用同一批单元 —— 只豁免不保留时, 优化器会把该处减料
        以换体积, 反而制造出不受约束的过应力.

        Parameters
        ----------
        mask : TensorLike, optional
            形状 ``(NC,)`` 的布尔张量; None 表示无实体保留区.
        """
        self._passive_element_mask = mask

    def get_passive_element_mask(self, mesh=None) -> Optional[TensorLike]:
        """返回实体保留单元掩码, 供优化器固定这些设计变量.

        优化器按 ``hasattr(pde, 'get_passive_element_mask')`` 探测该接口, 因此
        未注入掩码时返回 None 即可让优化器走无保留区的原路径.

        Parameters
        ----------
        mesh : HomogeneousMesh, optional
            设计变量所在网格; 本实现的掩码在注入时已绑定网格, 故忽略该参数,
            仅保留以匹配优化器的调用约定.

        Returns
        -------
        TensorLike or None
            形状 ``(NC,)`` 的布尔张量, 或 None.
        """
        return self._passive_element_mask

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
        """标记右端中点加载区 ``y in [y_mid - l/2, y_mid + l/2]``."""
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

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """Hu--Zhang 混合有限元的本质牵引边界条件标记 (右、顶、底边界)."""
        on_right = self._on_right_boundary(points)
        on_top = self._on_top_boundary(points)
        on_bottom = self._on_bottom_boundary(points)
        return bm.logical_or(on_right, bm.logical_or(on_top, on_bottom))

    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回右端加载区的边界牵引."""
        if self._traction is not None:
            return self._traction(points)
        return self._step_traction(points)

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回右端局部加载区的边界牵引."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )

    @cartesian
    def _step_traction(self, points: TensorLike) -> TensorLike:
        """右端局部加载区内的常值均布牵引力 (向下为负).

        点集必须按边 (面) 成批传入 ``(..., NP, GD)``, 做整边选取: 以边中心
        是否落在载荷区内决定整条边取常值强度 ``P/load_width`` 还是全零.
        载荷区端点与网格节点对齐时 (本仓库全部算例如此), 插值/积分后的
        合力严格等于 ``P``.

        不接受纯点集 ``(NP, GD)``: 该输入下只能退化为逐点阶跃判断, 会在载荷区
        端点顶点处向相邻边泄漏二次插值尾巴, 使 Hu--Zhang 本质边界的有效合力
        偏大 ``2|t|h/6`` (80x40 基准算例约 +5.6%). 同一个 ``BoundaryTraction``
        对象在两种输入下给出合力不同的载荷, 是静默的精度损失, 因此显式报错。
        需要逐点牵引值的调用方应改用 ``soptx.fem.boundary_loads`` 中的 P1 迹
        投影, 它在两条路径上给出同一个离散载荷泛函。
        """
        if points.ndim < 3:
            raise ValueError(
                "_step_traction 只支持按边 (面) 成批的求值点 (..., NP, GD); "
                f"实际输入形状 {tuple(points.shape)} 为纯点集, 整边选取语义无法"
                "定义。逐点牵引请使用 soptx.fem.boundary_loads 的 P1 迹投影."
            )

        result = bm.zeros(points.shape, **bm.context(points))
        centers = bm.mean(points, axis=-2)
        edge_mask = self._in_traction_patch(centers)
        mask = bm.broadcast_to(edge_mask[..., None], points.shape[:-1])
        loaded = bm.where(
            mask,
            bm.array(self.traction_intensity, **bm.context(points)),
            bm.zeros(mask.shape, **bm.context(points)),
        )
        result = bm.set_at(result, (..., 1), loaded)
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
    def _is_load_line(
        self,
        points: TensorLike,
    ) -> TensorLike:
        """标记 ``x=xmax, y=ymin`` 的整条网格边."""
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[1]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    @cartesian
    def _line_traction(self, points: TensorLike) -> TensorLike:
        """返回右端底边单位长度上的均布牵引."""
        value = bm.zeros(points.shape, **bm.context(points))
        line_length = self.domain[5] - self.domain[4]
        return bm.set_at(value, (..., 1), self.P / line_length)

    def loads(self) -> tuple[LineTractionLoad, ...]:
        """返回右端底边的均布线载荷."""
        return (
            LineTractionLoad(
                dimension=self.dimension,
                marker=self._is_load_line,
                value=self._line_traction,
            ),
        )
