"""悬臂梁工程基准问题."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import (
    BoundaryTractionLoad,
    LineTractionLoad,
    PointForceLoad,
)

from ._base import ElasticityBenchmark


class CantileverCornerPoint2d(ElasticityBenchmark):
    """右下角受集中力的二维悬臂梁.

    左端 ``x = x0`` 完全固支 ``u_x = u_y = 0``; 右下角 ``(x1, y0)`` 受竖直集中力 ``P``,
    节点分配由有限元装配层完成. 集中力不能按应力迹强施加, 本问题只用于 Lagrange
    位移元. 默认参数对应博士论文算例 6.1.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 160, 0, 100)``.
    P : float, optional
        集中力合力, 负值向下, 默认 ``-1``.
    E : float, optional
        实体材料的 Young 模量, 默认 ``1``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.3``.
    plane_type : str, optional
        平面假设, ``'plane_stress'`` (默认) 或 ``'plane_strain'``.
    """

    dimension = 2

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 160.0, 0.0, 100.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
        plane_type: str = "plane_stress",
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=plane_type)

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左端固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左端."""
        return self._on_left_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 左端."""
        return self._on_left_boundary(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], ...]:
        """返回 Lagrange 位移分量的边界标记函数."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    # ------------------------------------------------------------ 边界标记
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支边界 ``x = x0``."""
        return self._on_x(points, self.domain[0])

    # ---------------------------------------------------------------- 载荷
    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回右下角的集中点力."""
        return (
            PointForceLoad(
                point=(self.domain[1], self.domain[2]),
                vector=(0.0, self.P),
            ),
        )


class CantileverMiddlePatch2d(ElasticityBenchmark):
    """右端中点局部牵引受载的二维悬臂梁.

    左端 ``x = x0`` 完全固支 ``u_x = u_y = 0``; 顶、底边自由, 牵引为零; 右端中点宽
    ``load_width`` 的区间施加竖直均布牵引, 合力 ``P``, 强度 ``P / load_width``.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 80, 0, 40)``.
    P : float, optional
        局部牵引的竖向合力, 负值向下, 默认 ``-400``.
    load_width : float, optional
        右端中点受载区间的长度, 默认 ``6``; 端点须与网格节点对齐.
    E : float, optional
        实体材料的 Young 模量, 默认 ``70000``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.25``.
    plane_type : str, optional
        平面假设, ``'plane_stress'`` (默认) 或 ``'plane_strain'``.
    traction : Callable, optional
        替代默认阶跃牵引的边界牵引函数 (如 P1 迹投影载荷), 输入输出均为
        ``(..., GD)`` 张量.

    Raises
    ------
    ValueError
        ``load_width`` 不为正数, 或超过右边界长度 ``y1 - y0``.
    """

    dimension = 2

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 80.0, 0.0, 40.0),
        *,
        P: float = -400.0,
        load_width: float = 6.0,
        E: float = 70000.0,
        nu: float = 0.25,
        plane_type: str = "plane_stress",
        traction: Optional[Callable[[TensorLike], TensorLike]] = None,
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=plane_type)
        self._load_width = float(load_width)
        self._traction = traction
        if self._load_width <= 0.0:
            raise ValueError("load_width 必须为正数.")
        if self._load_width > self._domain[3] - self._domain[2]:
            raise ValueError("载荷区宽度 load_width 不得超过右边界长度 y1 - y0.")

    # ------------------------------------------------------------------ 载荷
    @property
    def load_width(self) -> float:
        """右端中点局部受载区间的长度."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """右端面上受载区间的起止纵坐标 ``(y_min_patch, y_max_patch)``."""
        y_mid = (self._domain[2] + self._domain[3]) / 2.0
        half_width = self._load_width / 2.0
        return (y_mid - half_width, y_mid + half_width)

    @property
    def traction_axis(self) -> int:
        """受载边所沿的坐标轴下标, 与 ``project_patch_traction_to_p1_trace`` 的 ``axis`` 同义.

        右边界沿 ``y`` 轴, 取 ``1``; ``traction_patch`` 是该轴上的区间.
        """
        return 1

    @property
    def traction_level(self) -> float:
        """受载边在垂直于 ``traction_axis`` 方向上的坐标, 即右边界的 ``x``."""
        return self._domain[1]

    @property
    def uses_step_traction(self) -> bool:
        """是否按整边选取施加载荷区牵引, 即构造时未注入 ``traction``.

        为 True 时载荷区端点必须是网格节点, 分析器构造时由
        ``soptx.fem.check_traction_patch_alignment`` 检查.
        """
        return self._traction is None

    @property
    def traction_intensity(self) -> float:
        """受载区间内的常值竖向牵引强度 ``P / load_width``."""
        return self._P / self._load_width

    # -------------------------------------------------------------- 应力奇点
    @property
    def traction_patch_endpoints(self) -> tuple[tuple[float, float], ...]:
        """牵引发生跳变的两个端点 ``((x1, y_min_patch), (x1, y_max_patch))``.

        局部应力豁免与实体保留以这两点为中心, 半径由算例配置给定, 掩码构造见
        ``soptx.topology.constraints.exemption``.
        """
        y_min_patch, y_max_patch = self.traction_patch
        level = self.traction_level
        return ((level, y_min_patch), (level, y_max_patch))

    @property
    def support_corner_points(self) -> tuple[tuple[float, float], ...]:
        """固支边两端的角点 ``((x0, y0), (x0, y1))``.

        固支边与上下自由边在此交界, 是 Dirichlet--Neumann 混合边界角点, 应力奇异.
        局部应力豁免与实体保留以这两点为中心, 半径由算例配置给定, 与载荷侧半径相互独立.
        """
        x_left = self._domain[0]
        return ((x_left, self._domain[2]), (x_left, self._domain[3]))

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左端固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左端."""
        return self._on_left_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 左端."""
        return self._on_left_boundary(points)

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
        """标记混合形式中弱施加位移的边界: 左端."""
        return self._on_left_boundary(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中强施加牵引的边界: 右、顶、底边."""
        return (
            self._on_x(points, self.domain[1])
            | self._on_y(points, self.domain[2])
            | self._on_y(points, self.domain[3])
        )

    # ------------------------------------------------------------ 边界标记
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支边界 ``x = x0``."""
        return self._on_x(points, self.domain[0])

    def _in_traction_patch(self, points: TensorLike) -> TensorLike:
        """标记右端载荷区 ``x = x1, y in traction_patch``."""
        y = points[..., 1]
        y_min_patch, y_max_patch = self.traction_patch
        in_y = (y >= y_min_patch - self._eps) & (y <= y_max_patch + self._eps)
        return self._on_x(points, self.traction_level) & in_y

    # ---------------------------------------------------------------- 牵引
    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回右、顶、底边的牵引数据; 构造时注入了 ``traction`` 则直接转发."""
        if self._traction is not None:
            return self._traction(points)
        return self._step_traction(points)

    @cartesian
    def _step_traction(self, points: TensorLike) -> TensorLike:
        """右端载荷区内的常值均布牵引 (向下为负), 其余牵引边界为零.

        点集必须按边成批传入 ``(..., NP, GD)``, 做整边选取: 以边中心是否落在载荷区内
        决定整条边取常值强度 ``P / load_width`` 还是全零. 载荷区端点与网格节点对齐时
        (本仓库全部算例如此), 插值/积分后的合力严格等于 ``P``.

        不接受纯点集 ``(NP, GD)``: 该输入下只能退化为逐点阶跃判断, 会在载荷区端点
        顶点处向相邻边泄漏二次插值尾巴, 使 Hu--Zhang 本质边界的有效合力偏大
        ``2|t|h/6`` (80x40 基准算例约 +5.6%). 同一个 ``BoundaryTraction`` 对象在两种
        输入下给出合力不同的载荷, 是静默的精度损失, 因此显式报错. 需要逐点牵引值的
        调用方应改用 ``soptx.fem.boundary_loads`` 中的 P1 迹投影, 它在两条路径上给出
        同一个离散载荷泛函.

        Raises
        ------
        ValueError
            ``points`` 不是按边成批的三维以上张量.
        """
        if points.ndim < 3:
            raise ValueError(
                "_step_traction 只支持按边成批的求值点 (..., NP, GD); "
                f"实际输入形状 {tuple(points.shape)} 为纯点集, 整边选取语义无法"
                "定义. 逐点牵引请使用 soptx.fem.boundary_loads 的 P1 迹投影."
            )

        centers = bm.mean(points, axis=-2)
        edge_mask = self._in_traction_patch(centers)
        mask = bm.broadcast_to(edge_mask[..., None], points.shape[:-1])
        traction = bm.zeros(points.shape, **bm.context(points))
        traction = bm.set_at(traction, (..., 1), self.traction_intensity)
        return bm.where(bm.expand_dims(mask, axis=-1), traction, bm.zeros_like(traction))

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回右端载荷区的边界牵引."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )


class CantileverBottomEdgeLine3d(ElasticityBenchmark):
    """右端底棱受均布线载荷的三维悬臂梁.

    左端面 ``x = x0`` 完全固支 ``u_x = u_y = u_z = 0``; 右端底棱 ``(x = x1, y = y0)``
    施加竖直均布线载荷, 合力 ``P``, 强度 ``P / (z1 - z0)``. 本问题只用于 Lagrange
    位移元.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1, z0, z1)``, 默认 ``(0, 60, 0, 20, 0, 4)``.
    P : float, optional
        线载荷合力, 负值向下, 默认 ``-1``.
    E : float, optional
        实体材料的 Young 模量, 默认 ``1``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.3``.
    """

    dimension = 3
    plane_type = "3D"

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 60.0, 0.0, 20.0, 0.0, 4.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=self.plane_type)

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左端固支的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左端面."""
        return self._on_left_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 左端面."""
        return self._on_left_boundary(points)

    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        """标记 ``u_z`` 的 Dirichlet 边界: 左端面."""
        return self._on_left_boundary(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], ...]:
        """返回 Lagrange 位移分量的边界标记函数."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
            self.is_dirichlet_boundary_dof_z,
        )

    # ------------------------------------------------------------ 边界标记
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支面 ``x = x0``."""
        return self._on_x(points, self.domain[0])

    # ---------------------------------------------------------------- 载荷
    @cartesian
    def _is_load_line(self, points: TensorLike) -> TensorLike:
        """标记右端底棱 ``(x = x1, y = y0)``."""
        return self._on_x(points, self.domain[1]) & self._on_y(points, self.domain[2])

    @cartesian
    def _line_traction(self, points: TensorLike) -> TensorLike:
        """返回载荷线单位长度上的竖直牵引 ``P / (z1 - z0)``."""
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P / (self.domain[5] - self.domain[4]))

    def loads(self) -> tuple[LineTractionLoad, ...]:
        """返回右端底棱的均布线载荷."""
        return (
            LineTractionLoad(
                dimension=self.dimension,
                marker=self._is_load_line,
                value=self._line_traction,
            ),
        )
