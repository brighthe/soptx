"""二维 MBB 梁工程基准问题。"""

from __future__ import annotations

from typing import Callable, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from ._base import validated_domain


class HalfMBBBeamRight2d:
    """对称右半域二维 MBB 梁。

    问题定义只包含区域、材料参数、体力和边界条件；网格由调用方显式创建。
    左边界施加 ``u_x = 0``，右下角施加 ``u_y = 0``，左上角承受竖直
    集中力 ``P``。
    """

    dimension = 2
    boundary_type = "mixed"
    load_type = "concentrated"
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
    def body_force(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

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

    @cartesian
    def _concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P)

    def concentrate_load_bc(self) -> list[Callable]:
        """返回与集中力边界标记一一对应的载荷值函数列表。"""
        return [self._concentrate_load_bc]

    @cartesian
    def is_concentrate_load_boundary_dof(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - self.domain[0]) < self._eps)
            & (bm.abs(y - self.domain[3]) < self._eps)
        )

    def is_concentrate_load_boundary(self) -> list[Callable]:
        return [self.is_concentrate_load_boundary_dof]


class HalfMBBBeamRight3d:
    """对称右半域三维 MBB 梁。

    问题定义包含区域、材料参数、体力和边界条件；网格由调用方显式创建。
    左对称面 (x=0) 施加 ``u_x = 0``，右下角底线 (x=Lx, y=0) 施加 ``u_y = 0``，
    底面中心平面 (y=0, z=Lz/2) 施加 ``u_z = 0``（防止刚体运动），
    左上角顶线 (x=0, y=Ly, z=Lz/2) 承受竖直集中力 ``P``。
    """

    dimension = 3
    boundary_type = "mixed"
    load_type = "concentrated"
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
    def body_force(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

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

    @cartesian
    def _concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P)

    def concentrate_load_bc(self) -> list[Callable]:
        return [self._concentrate_load_bc]

    @cartesian
    def is_concentrate_load_boundary_dof(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        z_mid = (self.domain[4] + self.domain[5]) / 2.0
        return (
            (bm.abs(x - self.domain[0]) < self._eps)
            & (bm.abs(y - self.domain[3]) < self._eps)
            & (bm.abs(z - z_mid) < self._eps)
        )

    def is_concentrate_load_boundary(self) -> list[Callable]:
        return [self.is_concentrate_load_boundary_dof]


class FullMBBBeam3d:
    """完整三维 MBB 梁 (1-to-1 完全精确对齐 Huang2023 论文 4.1 节)。

    物理问题未利用对称性简化，而是对整体全尺寸 3D 模型求解：
      - 左侧底部 (x=0, y=0): 铰支座 (u_x = 0, u_y = 0)
      - 右侧底部 (x=Lx, y=0): 滚轴支座 (u_y = 0)
      - 底面中心线 (y=0, z=Lz/2): uz = 0（防止刚体运动）
      - 顶面中心点 (x=Lx/2, y=Ly, z=Lz/2): 竖直向下集中荷载 P
    """

    dimension = 3
    boundary_type = "mixed"
    load_type = "concentrated"
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 120.0, 0.0, 20.0, 0.0, 20.0),
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
    def body_force(self, points: TensorLike) -> TensorLike:
        return bm.zeros(points.shape, **bm.context(points))

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

    @cartesian
    def _concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        value = bm.zeros(points.shape, **bm.context(points))
        return bm.set_at(value, (..., 1), self.P)

    def concentrate_load_bc(self) -> list[Callable]:
        return [self._concentrate_load_bc]

    @cartesian
    def is_concentrate_load_boundary_dof(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        xc = (self.domain[0] + self.domain[1]) / 2.0
        z_mid = (self.domain[4] + self.domain[5]) / 2.0
        return (
            (bm.abs(x - xc) < self._eps)
            & (bm.abs(y - self.domain[3]) < self._eps)
            & (bm.abs(z - z_mid) < self._eps)
        )

    def is_concentrate_load_boundary(self) -> list[Callable]:
        return [self.is_concentrate_load_boundary_dof]


