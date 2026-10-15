"""两端固支梁工程基准问题."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad

from ._base import ElasticityBenchmark


class FixedFixedBeamPatch2d(ElasticityBenchmark):
    """底边中点局部牵引受载的两端固支梁全域问题.

    左右两端 ``x = x0``, ``x = x1`` 完全固支 ``u_x = u_y = 0``; 顶边自由, 牵引为零;
    底边中点宽 ``load_width`` 的区间施加竖直均布牵引, 合力 ``P``, 强度 ``P / load_width``.

    Lagrange 位移元通过 Neumann 边界弱施加牵引, Hu--Zhang 混合元通过应力自由度强施加
    同一牵引. 默认牵引在载荷区边缘不连续, 两条路径的离散误差不同; 受控比较时用
    ``traction`` 注入等价的连续牵引 (如 ``soptx.fem.project_patch_traction_to_p1_trace``
    投影出的 P1 迹载荷), 几何、材料与边界标记全部不变.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 160, 0, 20)``.
    P : float, optional
        局部牵引的竖向合力, 负值向下, 默认 ``-3``.
    load_width : float, optional
        底边中点受载区间的长度, 默认 ``1``.
    E : float, optional
        实体材料的 Young 模量, 默认 ``30``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.4``.
    plane_type : str, optional
        平面假设, ``'plane_stress'`` (默认) 或 ``'plane_strain'``.
    traction : Callable, optional
        替代默认阶跃牵引的边界牵引函数, 输入输出均为 ``(..., GD)`` 张量.

    Raises
    ------
    ValueError
        ``load_width`` 不为正数.
    """

    dimension = 2

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
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=plane_type)
        self._load_width = float(load_width)
        self._traction = traction
        if self._load_width <= 0.0:
            raise ValueError("load_width 必须为正数.")

    # ------------------------------------------------------------------ 载荷
    @property
    def load_width(self) -> float:
        """底边中点局部受载区间的长度."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """底边上受载区间的起止横坐标."""
        x_mid = (self._domain[0] + self._domain[1]) / 2.0
        half_width = self._load_width / 2.0
        return (x_mid - half_width, x_mid + half_width)

    @property
    def traction_axis(self) -> int:
        """受载边所沿的坐标轴下标, 与 ``project_patch_traction_to_p1_trace`` 的 ``axis`` 同义.

        底边沿 ``x`` 轴, 取 ``0``; ``traction_patch`` 是该轴上的区间.
        """
        return 0

    @property
    def traction_level(self) -> float:
        """受载边在垂直于 ``traction_axis`` 方向上的坐标, 即底边的 ``y``."""
        return self._domain[2]

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

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回左右固支端的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左右固支端."""
        return self._on_clamped_ends(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 左右固支端."""
        return self._on_clamped_ends(points)

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
        """标记混合形式中弱施加位移的边界: 左右固支端."""
        return self._on_clamped_ends(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中强施加牵引的边界: 顶、底边 (不含固支端的角点)."""
        on_horizontal = self._on_y(points, self.domain[2]) | self._on_y(points, self.domain[3])
        return on_horizontal & ~self._on_clamped_ends(points)

    # ------------------------------------------------------------ 边界标记
    def _on_clamped_ends(self, points: TensorLike) -> TensorLike:
        """标记左右两条固支边界 ``x = x0`` 与 ``x = x1``."""
        return self._on_x(points, self.domain[0]) | self._on_x(points, self.domain[1])

    # ---------------------------------------------------------------- 牵引
    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回顶、底边的牵引数据.

        顶边始终为零牵引. 底边中点长度 ``load_width`` 的区间内施加常值
        ``P / load_width`` 的竖向牵引, 其余位置为零, 按点逐个判断, 因此连续合力
        严格等于 ``P``. 构造时若注入了 ``traction``, 则直接转发给它.
        """
        if self._traction is not None:
            return self._traction(points)

        x = points[..., 0]
        left, right = self.traction_patch
        x_mid = (left + right) / 2.0
        half_width = self._load_width / 2.0
        within_patch = bm.abs(x - x_mid) <= half_width + self._eps
        in_patch = self._on_y(points, self.traction_level) & within_patch
        traction = bm.zeros(points.shape, **bm.context(points))
        traction = bm.set_at(traction, (..., 1), self.traction_intensity)
        return bm.where(bm.expand_dims(in_patch, axis=-1), traction, bm.zeros_like(traction))

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回底边局部牵引."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )


class HalfFixedFixedBeamPatch2d(ElasticityBenchmark):
    """两端固支梁中点受载问题的左半域对称降维问题.

    完整域关于竖直中线对称, 本类只取左半域: 左端 ``x = x0`` 完全固支, 对称面
    ``x = x1`` 施加对称约束 (法向位移 ``u_x = 0`` 与切向牵引 ``sigma_xy = 0``), 底边
    对称面底端施加局部牵引. 完整域合力 ``P`` 由左右两半对称分担, 左半域合力为 ``P/2``,
    左半域柔顺度为完整域的一半, 报告完整结构柔顺度时应乘以 2.

    参数与 ``FixedFixedBeamPatch2d`` 一致, 但 ``domain`` 默认取左半域; 受控比较时
    同样用 ``traction`` 注入 P1 迹载荷.

    Parameters
    ----------
    domain : Sequence[float], optional
        左半计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 80, 0, 20)``, 对称面位于 ``x = x1``.
    P : float, optional
        完整域局部牵引的竖向合力, 负值向下, 默认 ``-3``; 半域合力为 ``P/2``.
    load_width : float, optional
        完整域底边中点受载区间的名义长度, 默认 ``1``; 半域只保留其左半边.
    E : float, optional
        实体材料的 Young 模量, 默认 ``30``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.4``.
    plane_type : str, optional
        平面假设, ``'plane_stress'`` (默认) 或 ``'plane_strain'``.
    traction : Callable, optional
        替代默认阶跃牵引的边界牵引函数, 输入输出均为 ``(..., GD)`` 张量.

    Raises
    ------
    ValueError
        ``load_width`` 不为正数, 或载荷区左端超出左半域.
    """

    dimension = 2

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
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=plane_type)
        self._load_width = float(load_width)
        self._traction = traction
        if self._load_width <= 0.0:
            raise ValueError("load_width 必须为正数.")
        if self._domain[1] - self._load_width / 2.0 < self._domain[0]:
            raise ValueError("载荷区左端超出左半域, 请检查 load_width 与 domain.")

    # ------------------------------------------------------------------ 载荷
    @property
    def load_width(self) -> float:
        """完整域底边中点局部受载区间的名义长度."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """左半域内受载区间的起止横坐标.

        载荷名义区间 ``[x1 - load_width/2, x1 + load_width/2]`` 关于对称面 ``x1``
        对称, 左半域只保留其左半边, 故区间宽 ``load_width/2``, 合力自动为完整域的
        一半 ``P/2``.
        """
        x_load = self._domain[1]
        return (x_load - self._load_width / 2.0, x_load)

    @property
    def traction_axis(self) -> int:
        """受载边所沿的坐标轴下标, 与 ``project_patch_traction_to_p1_trace`` 的 ``axis`` 同义.

        底边沿 ``x`` 轴, 取 ``0``; ``traction_patch`` 是该轴上的区间.
        """
        return 0

    @property
    def traction_level(self) -> float:
        """受载边在垂直于 ``traction_axis`` 方向上的坐标, 即底边的 ``y``."""
        return self._domain[2]

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

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回固支端与对称面的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左端固支与对称面."""
        return self._on_left_boundary(points) | self._on_symmetry_plane(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 左端固支."""
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
        """标记混合形式中弱施加位移的边界: 左端固支."""
        return self._on_left_boundary(points)

    @cartesian
    def is_symmetry_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中的对称面 ``x = x1``.

        对称面上切向牵引 ``sigma_nt = 0`` 需强施加, 法向牵引 ``sigma_nn`` 自由;
        法向位移 ``u_n = 0`` 在切向牵引固定后自然满足, 位移边界项贡献为零.
        """
        return self._on_symmetry_plane(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中强施加牵引的边界: 顶、底边 (不含固支端与对称面的角点)."""
        on_horizontal = self._on_y(points, self.domain[2]) | self._on_y(points, self.domain[3])
        return on_horizontal & ~self._on_left_boundary(points) & ~self._on_symmetry_plane(points)

    # ------------------------------------------------------------ 边界标记
    def _on_left_boundary(self, points: TensorLike) -> TensorLike:
        """标记左端固支边界 ``x = x0``."""
        return self._on_x(points, self.domain[0])

    def _on_symmetry_plane(self, points: TensorLike) -> TensorLike:
        """标记对称面 ``x = x1``."""
        return self._on_x(points, self.domain[1])

    # ---------------------------------------------------------------- 牵引
    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回顶、底边的牵引数据.

        底边对称面底端 ``x = x1`` 附近施加常值竖向牵引 ``P / load_width``, 其余位置
        为零, 按点逐个判断; 顶边始终为零牵引. 载荷名义区间关于对称面对称, 左半域只
        保留其左半边, 合力为完整域的一半 ``P/2``. 构造时若注入了 ``traction``, 则直接
        转发给它.
        """
        if self._traction is not None:
            return self._traction(points)

        x = points[..., 0]
        x_load = self._domain[1]
        half_width = self._load_width / 2.0
        within_patch = bm.abs(x - x_load) <= half_width + self._eps
        in_patch = self._on_y(points, self.traction_level) & within_patch
        traction = bm.zeros(points.shape, **bm.context(points))
        traction = bm.set_at(traction, (..., 1), self.traction_intensity)
        return bm.where(bm.expand_dims(in_patch, axis=-1), traction, bm.zeros_like(traction))

    def loads(self) -> tuple[BoundaryTractionLoad, ...]:
        """返回半域局部牵引, 合力为 ``P/2``."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )
