"""MBB 梁工程基准问题."""

from __future__ import annotations

from typing import Callable, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike

from soptx.problems.loads import LineTractionLoad, PointForceLoad

from ._base import ElasticityBenchmark


class HalfMBBBeamPoint2d(ElasticityBenchmark):
    """左上角受集中力的二维 MBB 梁对称右半域.

    左对称面 ``x = x0`` 约束 ``u_x = 0``, 右下角 ``(x1, y0)`` 滚支 ``u_y = 0``; 左上角
    ``(x0, y1)`` 受竖直集中力 ``P``, 节点分配由有限元装配层完成. 集中力不能按应力迹
    强施加, 本问题只用于 Lagrange 位移元.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 60, 0, 20)``.
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
        domain: Sequence[float] = (0.0, 60.0, 0.0, 20.0),
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
        """返回支承上的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左对称面 ``x = x0``."""
        return self._on_x(points, self.domain[0])

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 右下角 ``(x1, y0)``."""
        return self._on_x(points, self.domain[1]) & self._on_y(points, self.domain[2])

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable[[TensorLike], TensorLike], ...]:
        """返回 Lagrange 位移分量的边界标记函数."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )

    # ---------------------------------------------------------------- 载荷
    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回左上角的集中点力."""
        return (
            PointForceLoad(
                point=(self.domain[0], self.domain[3]),
                vector=(0.0, self.P),
            ),
        )


class HalfMBBBeamLine3d(ElasticityBenchmark):
    """对称面顶棱受均布线载荷的三维 MBB 梁对称右半域, 即 ``HalfMBBBeamPoint2d`` 沿宽度 z 方向的拉伸.

    坐标轴 x, y, z 分别沿长、高、宽, y 向上. 左对称面 ``x = x0`` 约束 ``u_x = 0``,
    右端底棱 ``(x = x1, y = y0)`` 约束 ``u_y = 0``; 拉伸后唯一剩余的刚体运动是 z 向
    平移, 由角点 ``(x0, y0, z0)`` 处的 ``u_z = 0`` 消除, 该约束不改变应力与柔度.
    二维的左上角集中力拉伸为对称面顶棱 ``(x = x0, y = y1)`` 上的竖直均布线载荷,
    强度 ``P / (z1 - z0)``. 本问题只用于 Lagrange 位移元.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1, z0, z1)``, 默认 ``(0, 60, 0, 20, 0, 20)``.
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
        domain: Sequence[float] = (0.0, 60.0, 0.0, 20.0, 0.0, 20.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=self.plane_type)

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回支承上的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左对称面 ``x = x0``."""
        return self._on_x(points, self.domain[0])

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 右端底棱 ``(x = x1, y = y0)``."""
        return self._on_bottom_edge(points, self.domain[1])

    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        """标记 ``u_z`` 的 Dirichlet 边界: 角点 ``(x0, y0, z0)``, 只消除 z 向平移."""
        return self._on_bottom_edge(points, self.domain[0]) & self._on_z(points, self.domain[4])

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
    def _on_bottom_edge(self, points: TensorLike, x_value: float) -> TensorLike:
        """标记底面 ``y = y0`` 上 ``x = x_value`` 那条棱."""
        return self._on_x(points, x_value) & self._on_y(points, self.domain[2])

    # ---------------------------------------------------------------- 载荷
    @cartesian
    def _is_load_line(self, points: TensorLike) -> TensorLike:
        """标记对称面顶棱 ``(x = x0, y = y1)``."""
        return self._on_x(points, self.domain[0]) & self._on_y(points, self.domain[3])

    @cartesian
    def _line_traction(self, points: TensorLike) -> TensorLike:
        """返回载荷线单位长度上的竖直牵引 ``P / (z1 - z0)``."""
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P / (self.domain[5] - self.domain[4]))

    def loads(self) -> tuple[LineTractionLoad, ...]:
        """返回对称面顶棱上的均布线载荷."""
        return (
            LineTractionLoad(
                dimension=self.dimension,
                marker=self._is_load_line,
                value=self._line_traction,
            ),
        )


class FullMBBBeamPoint2d(ElasticityBenchmark):
    """顶边中点受集中力的完整二维 MBB 梁.

    左下角 ``(x0, y0)`` 铰支 ``u_x = u_y = 0``, 右下角 ``(x1, y0)`` 滚支 ``u_y = 0``,
    恰好消除全部刚体运动. 顶边中点 ``((x0 + x1)/2, y1)`` 受竖直集中力 ``P``, 节点分配
    由有限元装配层完成. 集中力不能按应力迹强施加, 本问题只用于 Lagrange 位移元.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1)``, 默认 ``(0, 120, 0, 20)``.
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
        domain: Sequence[float] = (0.0, 120.0, 0.0, 20.0),
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
        """返回支承上的齐次位移数据."""
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

    # ---------------------------------------------------------------- 载荷
    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回顶边中点的集中点力."""
        x0, x1, _, y1 = self.domain
        return (PointForceLoad(point=((x0 + x1) / 2.0, y1), vector=(0.0, self.P)),)


class FullMBBBeamLine3d(ElasticityBenchmark):
    """顶面中线受均布线载荷的完整三维 MBB 梁, 即 ``FullMBBBeamPoint2d`` 沿宽度 z 方向的拉伸.

    坐标轴 x, y, z 分别沿长、高、宽, y 向上. 左端底棱 ``(x = x0, y = y0)`` 铰支
    ``u_x = u_y = 0``, 右端底棱 ``(x = x1, y = y0)`` 滚支 ``u_y = 0``; 拉伸后唯一剩余的
    刚体运动是 z 向平移, 由角点 ``(x0, y0, z0)`` 处的 ``u_z = 0`` 消除, 该约束不改变
    应力与柔度. 二维的顶边中点集中力拉伸为顶面中线 ``(x = (x0 + x1)/2, y = y1)`` 上的
    竖直均布线载荷, 强度 ``P / (z1 - z0)``; 中线须落在网格节点上 (x 向剖分数为偶数).
    本问题只用于 Lagrange 位移元.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1, z0, z1)``, 默认 ``(0, 120, 0, 20, 0, 20)``.
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
        domain: Sequence[float] = (0.0, 120.0, 0.0, 20.0, 0.0, 20.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=self.plane_type)

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回支承上的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左端底棱."""
        return self._on_bottom_edge(points, self.domain[0])

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 两端底棱."""
        return (
            self._on_bottom_edge(points, self.domain[0])
            | self._on_bottom_edge(points, self.domain[1])
        )

    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        """标记 ``u_z`` 的 Dirichlet 边界: 角点 ``(x0, y0, z0)``, 只消除 z 向平移."""
        return self._on_bottom_edge(points, self.domain[0]) & self._on_z(points, self.domain[4])

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
    def _on_bottom_edge(self, points: TensorLike, x_value: float) -> TensorLike:
        """标记底面 ``y = y0`` 上 ``x = x_value`` 那条棱."""
        return self._on_x(points, x_value) & self._on_y(points, self.domain[2])

    # ---------------------------------------------------------------- 载荷
    @cartesian
    def _is_load_line(self, points: TensorLike) -> TensorLike:
        """标记顶面中线 ``(x = (x0 + x1)/2, y = y1)``."""
        x_mid = (self.domain[0] + self.domain[1]) / 2.0
        return self._on_x(points, x_mid) & self._on_y(points, self.domain[3])

    @cartesian
    def _line_traction(self, points: TensorLike) -> TensorLike:
        """返回载荷线单位长度上的竖直牵引 ``P / (z1 - z0)``."""
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P / (self.domain[5] - self.domain[4]))

    def loads(self) -> tuple[LineTractionLoad, ...]:
        """返回顶面中线上的均布线载荷."""
        return (
            LineTractionLoad(
                dimension=self.dimension,
                marker=self._is_load_line,
                value=self._line_traction,
            ),
        )


class FullMBBBeamPoint3d(ElasticityBenchmark):
    """两端底角点支承、顶面中心受集中力的完整三维 MBB 梁.

    坐标轴 x, y, z 分别沿长、高、宽, y 向上. 左端底棱的两个角点 ``(x0, y0, z0)``,
    ``(x0, y0, z1)`` 铰支 ``u_x = u_y = u_z = 0``; 右端底棱的两个角点 ``(x1, y0, z0)``,
    ``(x1, y0, z1)`` 滚支, 沿 x 可滚动, ``u_y = u_z = 0``. 共 10 个约束, 消除全部 6 个
    刚体运动 (另有 4 个冗余); 两端的 ``u_y``, ``u_z`` 约束相同, 左端的 ``u_x`` 约束在竖直
    载荷下反力为零, 位移场关于 z 中面和跨中截面都对称. 顶面中心
    ``((x0 + x1)/2, y1, (z0 + z1)/2)`` 受竖直集中力 ``P``, 节点分配由有限元装配层完成.

    Parameters
    ----------
    domain : Sequence[float], optional
        计算域 ``(x0, x1, y0, y1, z0, z1)``, 默认 ``(0, 120, 0, 20, 0, 20)``.
    P : float, optional
        集中力合力, 负值向下, 默认 ``-1``.
    E : float, optional
        实体材料的 Young 模量, 默认 ``1``.
    nu : float, optional
        实体材料的 Poisson 比, 默认 ``0.3``.
    """

    dimension = 3
    plane_type = "3D"

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 20.0, 0.0, 20.0),
        *,
        P: float = -1.0,
        E: float = 1.0,
        nu: float = 0.3,
    ) -> None:
        super().__init__(domain, P=P, E=E, nu=nu, plane_type=self.plane_type)

    # ------------------------------------------------------------ Lagrange 层
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回支承上的齐次位移数据."""
        return bm.zeros(points.shape, **bm.context(points))

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """标记 ``u_x`` 的 Dirichlet 边界: 左端两底角点."""
        return self._on_end_corners(points, self.domain[0])

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """标记 ``u_y`` 的 Dirichlet 边界: 两端四个底角点."""
        return (
            self._on_end_corners(points, self.domain[0])
            | self._on_end_corners(points, self.domain[1])
        )

    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        """标记 ``u_z`` 的 Dirichlet 边界: 两端四个底角点."""
        return (
            self._on_end_corners(points, self.domain[0])
            | self._on_end_corners(points, self.domain[1])
        )

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
    def _on_end_corners(self, points: TensorLike, x_value: float) -> TensorLike:
        """标记 ``x = x_value`` 一端底棱 ``y = y0`` 的两个角点 ``z = z0, z1``."""
        on_z_end = self._on_z(points, self.domain[4]) | self._on_z(points, self.domain[5])
        return self._on_x(points, x_value) & self._on_y(points, self.domain[2]) & on_z_end

    # ---------------------------------------------------------------- 载荷
    def loads(self) -> tuple[PointForceLoad, ...]:
        """返回顶面中心的集中点力."""
        x0, x1, _, y1, z0, z1 = self.domain
        return (
            PointForceLoad(
                point=((x0 + x1) / 2.0, y1, (z0 + z1) / 2.0),
                vector=(0.0, self.P, 0.0),
            ),
        )
