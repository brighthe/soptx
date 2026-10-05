"""二维 MBB 梁工程基准问题."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import PointForceLoad

from ._base import validated_domain


class HalfMBBBeamRight2d:
    """对称右半域二维 MBB 梁.

    问题定义只包含区域、材料参数、体力和边界条件; 网格由调用方显式创建.
    左边界施加 ``u_x = 0``, 右下角施加 ``u_y = 0``, 左上角承受竖直
    集中力 ``P``.
    """

    dimension = 2
    boundary_type = "mixed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 60.0, 0.0, 20.0),
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
        return self._domain

    @property
    def P(self) -> float:
        return self._P

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def plane_type(self) -> str:
        return self._plane_type

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return bm.abs(points[..., 0] - self.domain[0]) < self._eps

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[1]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回左上角的集中点力."""
        return (
            PointForceLoad(
                point=(self.domain[0], self.domain[3]),
                vector=(0.0, self.P),
            ),
        )


class HalfMBBBeamRight3d:
    """对称右半域三维 MBB 梁.

    问题定义包含区域、材料参数、体力和边界条件; 网格由调用方显式创建.
    左对称面 (x=0) 施加 ``u_x = 0``, 右下角底线 (x=Lx, y=0) 施加 ``u_y = 0``,
    底面中心平面 (y=0, z=Lz/2) 施加 ``u_z = 0``(防止刚体运动),
    左上角顶线 (x=0, y=Ly, z=Lz/2) 承受竖直集中力 ``P``.
    """

    dimension = 3
    boundary_type = "mixed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 60.0, 0.0, 20.0, 0.0, 20.0),
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
        return self._domain

    @property
    def P(self) -> float:
        return self._P

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def plane_type(self) -> str:
        return self._plane_type

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return bm.abs(points[..., 0] - self.domain[0]) < self._eps

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[1]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_z(
        self,
        points: TensorLike,
    ) -> TensorLike:
        y, z = points[..., 1], points[..., 2]
        z_mid = (self.domain[4] + self.domain[5]) / 2.0
        return (
            (bm.abs(y - self.domain[2]) < self._eps)
            & (bm.abs(z - z_mid) < self._eps)
        )

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable, Optional[Callable]]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
            self.is_dirichlet_boundary_dof_z,
        )

    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回左上方中心位置的集中点力."""
        return (
            PointForceLoad(
                point=(
                    self.domain[0],
                    self.domain[3],
                    (self.domain[4] + self.domain[5]) / 2.0,
                ),
                vector=(0.0, self.P, 0.0),
            ),
        )


class FullMBBBeam3d:
    """完整三维 MBB 梁.

    不利用对称性简化, 对整体模型求解. 坐标轴 x, y, z 分别沿长、高、宽, y 向上;
    顶面中心 (x = xm, y = y1, z = zm) 受竖直集中力 P. 支承由 ``support`` 选择:

    - ``'centerline'``: 左端底边 (x = x0, y = y0) 约束 u_x = u_y = 0, 右端底边
      (x = x1, y = y0) 约束 u_y = 0, 底面中线 (y = y0, z = zm) 约束 u_z = 0 以消除
      刚体运动;
    - ``'end_lines'``: 左端底边铰支, u_x = u_y = u_z = 0; 右端底边滚支, u_y = 0.

    Parameters
    ----------
    domain : 区域 (x0, x1, y0, y1, z0, z1).
    P : 集中力合力, 负值向下.
    E : 杨氏模量.
    nu : 泊松比.
    plane_type : 本构假设, 三维固定为 ``'3D'``.
    support : 支承方式, 取 ``'centerline'`` (默认) 或 ``'end_lines'``.
    load_subdivisions : 调用方网格在 x, z 向的剖分数 (nx, nz), 用于把集中力离散
        到节点上, 须与实际网格一致; 为 None (默认) 时返回作用于几何中心的单个
        点力.

    Raises
    ------
    ValueError
        ``support`` 取值未知, ``load_subdivisions`` 不是两个正整数, 或在 z 向剖分数为
        奇数时选用 ``'centerline'``.

    Notes
    -----
    ``'centerline'`` 的 u_z 约束取底面上距中线最近的全部节点. z 向剖分数为偶数
    时这些节点在中面上, 对称解在那里本就有 u_z = 0, 约束不改变解; 为奇数时中线
    不落在节点上, 约束落在偏离中面的两排节点上, 改变问题本身. 因此给出
    ``load_subdivisions`` 且 z 向剖分数为奇数时, ``'centerline'`` 直接报错, 须改用
    ``'end_lines'``; 不给出时剖分数未知, 不做检查. 支承属于问题定义, 不随网格自动
    切换, 否则不同网格求解的是不同的边值问题.

    几何中心的点力只有在中心恰为网格节点时才能按 ``mode='exact'`` 投影 (LFEM
    分析器的做法). 给出 ``load_subdivisions`` 后, 某方向剖分数为偶数时中心落在
    节点上, 为奇数时落在单元棱中点, 合力均分给两侧节点, 即线性单元下该点力的
    一致节点载荷; ``loads()`` 因此返回 1、2 或 4 个落在节点上的点力, 合力恒为 P.
    """

    dimension = 3
    boundary_type = "mixed"
    _eps = 1.0e-12
    _supports = ("centerline", "end_lines")

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 20.0, 0.0, 20.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
        plane_type: str = "3D",
        support: Literal["centerline", "end_lines"] = "centerline",
        load_subdivisions: Optional[tuple[int, int]] = None,
    ) -> None:
        if support not in self._supports:
            raise ValueError(f"support 须为 {self._supports} 之一, 实际为 {support!r}")
        if load_subdivisions is not None:
            values = tuple(load_subdivisions)
            if len(values) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            ):
                raise ValueError(
                    f"load_subdivisions 须为两个正整数 (nx, nz), 实际为 {load_subdivisions!r}"
                )
            load_subdivisions = values
            if support == "centerline" and values[1] % 2 == 1:
                raise ValueError(
                    f"z 向剖分数 nz = {values[1]} 为奇数, 中线不落在节点上, 'centerline' 的 "
                    "u_z 约束会改变问题本身; 请改用 support='end_lines'"
                )
        self._domain = validated_domain(domain, self.dimension)
        self._P = float(P)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = plane_type
        self._support = support
        self._load_subdivisions = load_subdivisions

    @property
    def domain(self) -> tuple[float, ...]:
        return self._domain

    @property
    def P(self) -> float:
        return self._P

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def plane_type(self) -> str:
        return self._plane_type

    @property
    def support(self) -> str:
        """支承方式"""
        return self._support

    @property
    def load_subdivisions(self) -> Optional[tuple[int, int]]:
        """集中力离散所依据的 (nx, nz), None 表示作用于几何中心"""
        return self._load_subdivisions

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

    def _on_bottom_edge(self, points: TensorLike, x_value: float) -> TensorLike:
        """标记位于底面 y = y0 且 x = x_value 那条棱上的点."""
        x, y = points[..., 0], points[..., 1]
        return (bm.abs(x - x_value) < self._eps) & (bm.abs(y - self.domain[2]) < self._eps)

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self._on_bottom_edge(points, self.domain[0])

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return (self._on_bottom_edge(points, self.domain[0])
                | self._on_bottom_edge(points, self.domain[1]))

    @cartesian
    def is_dirichlet_boundary_dof_z(
        self,
        points: TensorLike,
    ) -> TensorLike:
        if self._support == "end_lines":
            return self._on_bottom_edge(points, self.domain[0])

        y, z = points[..., 1], points[..., 2]
        z_mid = (self.domain[4] + self.domain[5]) / 2.0
        on_bottom = bm.abs(y - self.domain[2]) < self._eps
        return self._nearest_candidate_mask(
            distance=bm.abs(z - z_mid),
            candidates=on_bottom,
        )

    def _nearest_candidate_mask(
        self,
        distance: TensorLike,
        candidates: TensorLike,
    ) -> TensorLike:
        """在候选节点中选择到目标几何位置距离最小的全部对称节点."""
        infinity = bm.full(
            distance.shape,
            float("inf"),
            **bm.context(distance),
        )
        nearest = bm.min(bm.where(candidates, distance, infinity))
        return candidates & (bm.abs(distance - nearest) < self._eps)

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
            self.is_dirichlet_boundary_dof_z,
        )

    @staticmethod
    def _center_nodes(lower: float, upper: float, n: int) -> tuple[tuple[float, float], ...]:
        """区间中心在 n 等分网格上的节点坐标及其载荷权重.

        Returns
        -------
        nodes : n 为偶数时是 ((中心, 1), ), 为奇数时是中心两侧两个节点各带权重 1/2.
        """
        h = (upper - lower) / n
        if n % 2 == 0:
            return ((lower + (n // 2) * h, 1.0), )
        return ((lower + (n // 2) * h, 0.5), (lower + (n // 2 + 1) * h, 0.5))

    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回顶面中心的集中力; 给出 ``load_subdivisions`` 时离散到节点上, 合力为 P."""
        x0, x1, _, y1, z0, z1 = self.domain
        if self._load_subdivisions is None:
            return (
                PointForceLoad(
                    point=((x0 + x1) / 2.0, y1, (z0 + z1) / 2.0),
                    vector=(0.0, self.P, 0.0),
                ),
            )

        nx, nz = self._load_subdivisions
        return tuple(
            PointForceLoad(point=(x, y1, z), vector=(0.0, wx * wz * self.P, 0.0))
            for x, wx in self._center_nodes(x0, x1, nx)
            for z, wz in self._center_nodes(z0, z1, nz)
        )


class FullMBBBeam2d:
    """完整二维 MBB 梁 (1-to-1 完全精确对齐整体 MBB 梁结构, 对应 Huang 2023 第 4.1 节).

    物理问题未利用对称性简化, 是对整体全尺寸 2D 模型求解:
      - 左下角 (x=0, y=0): 铰支座 (u_x = 0, u_y = 0)
      - 右下角 (x=Lx, y=0): 滚轴支座 (u_y = 0)
      - 顶面中心点 (x=Lx/2, y=Ly): 竖直向下集中荷载 P
    """

    dimension = 2
    boundary_type = "mixed"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 20.0),
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
        return self._domain

    @property
    def P(self) -> float:
        return self._P

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def plane_type(self) -> str:
        return self._plane_type

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[0]) < self._eps)
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (
            ((bm.abs(x - self.domain[0]) < self._eps) | (bm.abs(x - self.domain[1]) < self._eps))
            & (bm.abs(y - self.domain[2]) < self._eps)
        )

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回顶边中心的集中点力."""
        return (
            PointForceLoad(
                point=(
                    (self.domain[0] + self.domain[1]) / 2.0,
                    self.domain[3],
                ),
                vector=(0.0, self.P),
            ),
        )

    def get_load_dof(self, total_fine_x: int, total_fine_y: int) -> int:
        """根据细网格分割维度 (nx, ny) 导出顶面中心集中荷载 P 作用点自由度 (y 向 DOF)."""
        top_center_node = (total_fine_x // 2) * (total_fine_y + 1) + total_fine_y
        return 2 * top_center_node + 1
