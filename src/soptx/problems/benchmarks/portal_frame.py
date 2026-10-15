"""门式框架工程基准问题."""

from __future__ import annotations

from math import isfinite
from typing import Callable, Optional

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import BoundaryTractionLoad

from ._base import ElasticityBenchmark


class PortalFramePatch2d(ElasticityBenchmark):
    """顶边中点局部牵引受载的门式框架完整域问题.

    设计域为 ``L x H`` 矩形去掉底边中央以 ``(b, 0)``, ``(L/2, h_c)``, ``(L - b, 0)``
    为顶点的三角形缺口, 为七边形 (见 ``vertices``); ``domain`` 是其包围盒 ``(0, L, 0, H)``.
    左支座段 ``y = 0, x in [0, b]`` 为固定铰 ``u_x = u_y = 0``, 右支座段
    ``y = 0, x in [L - b, L]`` 为滚动铰 ``u_y = 0``; 其余边界自由, 仅顶边中点宽
    ``load_width`` 的区间施加竖直均布牵引, 合力 ``P``, 强度 ``P / load_width``. 几何与
    载荷关于 ``x = L/2`` 对称而支座不对称, 须求全域.

    Parameters
    ----------
    L : float, optional
        域宽度, 默认 ``12``.
    H : float, optional
        域高度, 默认 ``6``.
    P : float, optional
        载荷区的竖向合力, 负值向下, 默认 ``-300``.
    load_width : float, optional
        顶边载荷区宽度 ``l``, 默认 ``1``; 端点须与网格节点对齐.
    b : float, optional
        底边支座段宽度, 默认 ``0.55``; 须满足 ``0 < b < L/2``.
    h_c : float, optional
        凹角顶点高度, 默认 ``3.5``; 须满足 ``0 < h_c < H``.
    E : float, optional
        实体材料的 Young 模量, 默认 ``100000``.
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
        几何参数不满足上述约束.
    """

    dimension = 2

    def __init__(
        self,
        *,
        L: float = 12.0,
        H: float = 6.0,
        P: float = -300.0,
        load_width: float = 1.0,
        b: float = 0.55,
        h_c: float = 3.5,
        E: float = 100000.0,
        nu: float = 0.25,
        plane_type: str = "plane_stress",
        traction: Optional[Callable[[TensorLike], TensorLike]] = None,
    ) -> None:
        super().__init__((0.0, L, 0.0, H), P=P, E=E, nu=nu, plane_type=plane_type)
        self._L, self._H = float(L), float(H)
        self._load_width, self._b, self._h_c = float(load_width), float(b), float(h_c)
        self._traction = traction
        for name, value in (("L", self._L), ("H", self._H), ("load_width", self._load_width),
                            ("b", self._b), ("h_c", self._h_c)):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须为正的有限数, 实际为 {value!r}.")
        if self._b >= self._L / 2.0:
            raise ValueError("支座段宽度 b 必须小于 L/2.")
        if self._h_c >= self._H:
            raise ValueError("凹角顶点高度 h_c 必须小于 H.")
        if self._load_width > self._L:
            raise ValueError("载荷区宽度 load_width 不得超过 L.")

    # ------------------------------------------------------------------ 几何
    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        """七边形顶点, 逆时针: ``(0,0), (b,0), (L/2,h_c), (L-b,0), (L,0), (L,H), (0,H)``."""
        return (
            (0.0, 0.0),
            (self._b, 0.0),
            (self._L / 2.0, self._h_c),
            (self._L - self._b, 0.0),
            (self._L, 0.0),
            (self._L, self._H),
            (0.0, self._H),
        )

    @property
    def L(self) -> float:
        """域宽度."""
        return self._L

    @property
    def H(self) -> float:
        """域高度."""
        return self._H

    @property
    def b(self) -> float:
        """底边支座段宽度."""
        return self._b

    @property
    def h_c(self) -> float:
        """凹角顶点高度."""
        return self._h_c

    @property
    def reentrant_corner_point(self) -> tuple[float, float]:
        """凹角顶点 ``(L/2, h_c)``."""
        return (self._L / 2.0, self._h_c)

    @property
    def symmetry_axis(self) -> float:
        """几何与载荷的对称轴横坐标 ``L/2``; 供网格镜像与设计对称化使用."""
        return self._L / 2.0

    # ------------------------------------------------------------------ 载荷
    @property
    def load_width(self) -> float:
        """顶边载荷区的宽度 ``l``."""
        return self._load_width

    @property
    def traction_patch(self) -> tuple[float, float]:
        """受载区间的起止横坐标 ``(L/2 - l/2, L/2 + l/2)``."""
        x_load = self._L / 2.0
        return (x_load - self._load_width / 2.0, x_load + self._load_width / 2.0)

    @property
    def traction_axis(self) -> int:
        """受载边所沿的坐标轴下标, 与 ``project_patch_traction_to_p1_trace`` 的 ``axis`` 同义.

        顶边沿 ``x`` 轴, 取 ``0``; ``traction_patch`` 是该轴上的区间.
        """
        return 0

    @property
    def traction_level(self) -> float:
        """受载边在垂直于 ``traction_axis`` 方向上的坐标, 即顶边的 ``y = H``."""
        return self._H

    @property
    def uses_step_traction(self) -> bool:
        """是否按整边选取施加载荷区牵引, 即构造时未注入 ``traction``.

        为 True 时载荷区端点必须是网格节点, 分析器构造时由
        ``soptx.fem.check_traction_patch_alignment`` 检查.
        """
        return self._traction is None

    @property
    def traction_intensity(self) -> float:
        """受载区间内的常值竖向牵引强度 ``P / l``."""
        return self._P / self._load_width

    # -------------------------------------------------------------- 应力奇点
    @property
    def traction_patch_endpoints(self) -> tuple[tuple[float, float], ...]:
        """牵引发生跳变的两个端点 ``((L/2 - l/2, H), (L/2 + l/2, H))``.

        局部应力豁免与实体保留以这两点为中心, 半径由算例配置给定, 掩码构造见
        ``soptx.topology.constraints.exemption``.
        """
        left, right = self.traction_patch
        return ((left, self._H), (right, self._H))

    @property
    def support_corner_points(self) -> tuple[tuple[float, float], ...]:
        """两段支座的四个端点 ``(0, 0)``, ``(b, 0)``, ``(L - b, 0)``, ``(L, 0)``.

        支座段施加位移边界条件, 与之相接的左右边和两条斜边为自由边, 故这四点是
        Dirichlet--Neumann 混合边界角点, 应力按幂律奇异; 奇异性由边界条件类型的
        改变产生, 不能靠载荷分布化削弱, 局部应力豁免与实体保留是仅有的处置. 半径
        由算例配置给定, 对应 ``CantileverMiddlePatch2d.support_corner_points``.

        凹角顶点 ``(L/2, h_c)`` 是几何奇点, 不在此列: 优化从凹角应力集中区移除材料
        正是本问题的预期结果, 不应钉为实体.
        """
        return ((0.0, 0.0), (self._b, 0.0), (self._L - self._b, 0.0), (self._L, 0.0))

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回位移边界上的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左支座段."""
        return self._on_left_support(points)

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 两段支座."""
        return self._on_left_support(points) | self._on_right_support(points)

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
        """标记混合形式中弱施加全位移的边界: 左支座段."""
        return self._on_left_support(points)

    @cartesian
    def is_symmetry_boundary(self, points: TensorLike) -> TensorLike:
        """标记只强加切向牵引 ``sigma_nt = 0`` 的边界: 右支座段 (滚动支承).

        名称沿用分析器的接口约定 (对称面与滚动支承的强施加条件相同); 本问题
        没有对称面. 法向位移 ``u_n = 0`` 在切向牵引固定后自然满足.
        """
        return self._on_right_support(points)

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记混合形式中强施加全牵引的边界: 左边、右边、顶边与两条斜边."""
        return (
            self._on_x(points, 0.0)
            | self._on_x(points, self._L)
            | self._on_y(points, self._H)
            | self._on_left_slant_edge(points)
            | self._on_right_slant_edge(points)
        )

    # ------------------------------------------------------------ 边界标记
    def _on_bottom_segment(self, points: TensorLike, x0: float, x1: float) -> TensorLike:
        """标记底边 ``y = 0`` 上 ``x in [x0, x1]`` 的点."""
        x = points[..., 0]
        return self._on_y(points, 0.0) & (x >= x0 - self._eps) & (x <= x1 + self._eps)

    def _on_segment(self, points: TensorLike, start: tuple[float, float],
                    end: tuple[float, float]) -> TensorLike:
        """标记线段 ``start -- end`` 上的点 (点到直线距离与投影参数判定)."""
        x, y = points[..., 0], points[..., 1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        px, py = x - start[0], y - start[1]
        distance = bm.abs(dx * py - dy * px) / length
        parameter = (dx * px + dy * py) / (length * length)
        # 投影参数无量纲, 长度容差须除以线段长度换算到参数空间
        parameter_tol = self._eps / length
        return (distance < self._eps) & (parameter >= -parameter_tol) & (parameter <= 1.0 + parameter_tol)

    def _on_left_support(self, points: TensorLike) -> TensorLike:
        """标记左支座段 ``y = 0``, ``x in [0, b]``."""
        return self._on_bottom_segment(points, 0.0, self._b)

    def _on_right_support(self, points: TensorLike) -> TensorLike:
        """标记右支座段 ``y = 0``, ``x in [L - b, L]``."""
        return self._on_bottom_segment(points, self._L - self._b, self._L)

    def _on_left_slant_edge(self, points: TensorLike) -> TensorLike:
        """标记左斜边 ``(b, 0) -- (L/2, h_c)`` 上的点."""
        return self._on_segment(points, (self._b, 0.0), self.reentrant_corner_point)

    def _on_right_slant_edge(self, points: TensorLike) -> TensorLike:
        """标记右斜边 ``(L - b, 0) -- (L/2, h_c)`` 上的点."""
        return self._on_segment(points, (self._L - self._b, 0.0), self.reentrant_corner_point)

    def _in_traction_patch(self, points: TensorLike) -> TensorLike:
        """标记顶边载荷区 ``y = H``, ``x`` 落在 ``traction_patch`` 内的点."""
        x = points[..., 0]
        left, right = self.traction_patch
        return self._on_y(points, self._H) & (x >= left - self._eps) & (x <= right + self._eps)

    # ---------------------------------------------------------------- 牵引
    @cartesian
    def _boundary_traction(self, points: TensorLike) -> TensorLike:
        """返回牵引边界的牵引数据; 构造时注入了 ``traction`` 则直接转发."""
        if self._traction is not None:
            return self._traction(points)
        return self._step_traction(points)

    @cartesian
    def _step_traction(self, points: TensorLike) -> TensorLike:
        """顶边载荷区内的常值均布牵引 (向下为负), 其余牵引边界为零.

        点集必须按边成批传入 ``(..., NP, GD)``, 做整边选取: 以边中心是否落在
        载荷区内决定整条边取常值强度 ``P / l`` 还是全零. 载荷区端点与网格节点
        对齐时, 插值/积分后的合力严格等于 ``P``.

        不接受纯点集 ``(NP, GD)``: 逐点阶跃判断会在载荷区端点顶点处向相邻边
        泄漏插值尾巴, 使 Hu--Zhang 本质边界的有效合力偏大, 因此显式报错.
        逐点牵引请改用 ``soptx.fem.boundary_loads`` 的 P1 迹投影.

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
        """返回顶边载荷区的边界牵引."""
        return (
            BoundaryTractionLoad(
                dimension=self.dimension,
                marker=self.is_traction_boundary,
                value=self._boundary_traction,
            ),
        )
