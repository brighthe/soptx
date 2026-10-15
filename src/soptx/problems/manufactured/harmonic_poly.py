"""二维与三维调和多项式制造解: 位移由非齐次边界驱动, 体力为零.

完整方程与推导见 ``docs/problems/manufactured-elasticity.md``.
"""

from __future__ import annotations

from math import isfinite
from typing import Callable, Sequence

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.typing import TensorLike
from soptx.core.numerics import GEOMETRIC_TOL

from ._all_dirichlet import AllDisplacementBoundaryMixin, _AllDirichletElasticity2D


class HarmonicPoly2D(_AllDirichletElasticity2D):
    r"""二维无体力调和多项式制造解线弹性问题.

    位移场由 4 次调和流函数求导得到, 在任意弹性参数下严格满足
    :math:`\Delta u = 0, \nabla \cdot u = 0 \implies b \equiv 0`.
    边界为非齐次全 Dirichlet 约束 :math:`u|_{\partial\Omega} = u_\mathrm{exact}`.
    数学推导见 ``docs/problems/manufactured-elasticity.md``.
    """

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 0.5,
    ) -> None:
        r"""初始化二维调和多项式线弹性问题.

        参数:
            domain: 轴对齐盒形区域边界 ``(x_min, x_max, y_min, y_max)``.
            lame_lambda: Lamé 第一参数 :math:`\lambda`.
            shear_modulus: 剪切模量 :math:`\mu`.
        """
        super().__init__(domain)
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda 必须是有界实数.")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus 必须是有界正实数.")
        self._lam = float(lame_lambda)
        self._mu = float(shear_modulus)

    @property
    def lam(self) -> float:
        r"""返回 Lamé 第一参数 :math:`\lambda`."""
        return self._lam

    @property
    def mu(self) -> float:
        r"""返回剪切模量 :math:`\mu`."""
        return self._mu

    @property
    def E(self) -> float:
        """返回等效杨氏模量."""
        return self._mu * (3 * self._lam + 2 * self._mu) / (self._lam + self._mu)

    @property
    def nu(self) -> float:
        """返回等效泊松比."""
        return self._lam / (2 * (self._lam + self._mu))

    @cartesian
    def _body_force(self, points: TensorLike) -> TensorLike:
        """返回连续体力场, 调和多项式下恒为零向量."""
        return bm.zeros_like(points)

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        """返回精确位移场, 形状 ``(..., 2)``."""
        x, y = points[..., 0], points[..., 1]
        u_x = -12.0 * x**2 * y + 4.0 * y**3
        u_y = -4.0 * x**3 + 12.0 * x * y**2
        return bm.stack([u_x, u_y], axis=-1)

    @cartesian
    def disp_solution_gradient(self, points: TensorLike) -> TensorLike:
        """返回精确位移梯度场, 形状 ``(..., 2, 2)``."""
        x, y = points[..., 0], points[..., 1]
        du_x_dx = -24.0 * x * y
        du_x_dy = -12.0 * x**2 + 12.0 * y**2
        du_y_dx = -12.0 * x**2 + 12.0 * y**2
        du_y_dy = 24.0 * x * y
        return bm.stack(
            [
                bm.stack([du_x_dx, du_x_dy], axis=-1),
                bm.stack([du_y_dx, du_y_dy], axis=-1),
            ],
            axis=-2,
        )

    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        """返回精确位移梯度场, 形状 ``(..., 2, 2)``."""
        return self.disp_solution_gradient(points)

    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        """返回 Voigt 记号下的精确应力张量 ``(sigma_xx, sigma_xy, sigma_yy)``."""
        x, y = points[..., 0], points[..., 1]
        mu = self._mu
        sigma_xx = -48.0 * mu * x * y
        sigma_yy = 48.0 * mu * x * y
        sigma_xy = 24.0 * mu * (-x**2 + y**2)
        return bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)

    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        """返回应力张量的散度场, 恒为零向量."""
        return bm.zeros_like(points)


class HarmonicPoly3D(AllDisplacementBoundaryMixin):
    r"""三维无体力调和多项式制造解线弹性问题.

    位移场严格满足 :math:`\Delta u = 0, \nabla \cdot u = 0 \implies b \equiv 0`.
    边界为非齐次全 Dirichlet 约束 :math:`u|_{\partial\Omega} = u_\mathrm{exact}`.
    数学推导见 ``docs/problems/manufactured-elasticity.md``.
    """

    dimension = 3
    plane_type = "3D"
    boundary_type = "dirichlet"

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 0.5,
    ) -> None:
        r"""初始化三维调和多项式线弹性问题.

        参数:
            domain: 轴对齐盒形区域边界 ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
            lame_lambda: Lamé 第一参数 :math:`\lambda`.
            shear_modulus: 剪切模量 :math:`\mu`.
        """
        self._domain = tuple(float(value) for value in domain)
        self._eps = GEOMETRIC_TOL
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda 必须是有界实数.")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus 必须是有界正实数.")
        self._lam = float(lame_lambda)
        self._mu = float(shear_modulus)

    @property
    def domain(self) -> tuple[float, ...]:
        """返回求解域边界元组."""
        return self._domain

    @property
    def lam(self) -> float:
        r"""返回 Lamé 第一参数 :math:`\lambda`."""
        return self._lam

    @property
    def mu(self) -> float:
        r"""返回剪切模量 :math:`\mu`."""
        return self._mu

    @property
    def E(self) -> float:
        """返回等效杨氏模量."""
        return self._mu * (3 * self._lam + 2 * self._mu) / (self._lam + self._mu)

    @property
    def nu(self) -> float:
        """返回等效泊松比."""
        return self._lam / (2 * (self._lam + self._mu))

    @cartesian
    def _body_force(self, points: TensorLike) -> TensorLike:
        """返回连续体力场, 调和多项式下恒为零向量."""
        return bm.zeros_like(points)

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        """返回精确位移场, 形状 ``(..., 3)``."""
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        u_x = -12.0 * x**2 * y + 4.0 * y**3
        u_y = 4.0 * z**3 - 12.0 * x**2 * z - 4.0 * x**3 + 12.0 * x * y**2
        u_z = bm.zeros_like(x)
        return bm.stack([u_x, u_y, u_z], axis=-1)

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        """返回精确位移梯度场, 形状 ``(..., 3, 3)``."""
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        zero = bm.zeros_like(x)
        row0 = bm.stack([-24.0 * x * y, -12.0 * x**2 + 12.0 * y**2, zero], axis=-1)
        row1 = bm.stack(
            [-24.0 * x * z - 12.0 * x**2 + 12.0 * y**2, 24.0 * x * y, 12.0 * z**2 - 12.0 * x**2],
            axis=-1,
        )
        row2 = bm.stack([zero, zero, zero], axis=-1)
        return bm.stack([row0, row1, row2], axis=-2)

    def disp_solution_gradient(self, points: TensorLike) -> TensorLike:
        """返回精确位移梯度场, 形状 ``(..., 3, 3)``."""
        return self.grad_disp_solution(points)

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """返回 Dirichlet 本质边界条件位移值."""
        return self.disp_solution(points)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """判定点是否在 x 方向位移本质边界上."""
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        domain = self.domain
        return (
            (bm.abs(x - domain[0]) < self._eps)
            | (bm.abs(x - domain[1]) < self._eps)
            | (bm.abs(y - domain[2]) < self._eps)
            | (bm.abs(y - domain[3]) < self._eps)
            | (bm.abs(z - domain[4]) < self._eps)
            | (bm.abs(z - domain[5]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """判定点是否在 y 方向位移本质边界上."""
        return self.is_dirichlet_boundary_dof_x(points)

    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        """判定点是否在 z 方向位移本质边界上."""
        return self.is_dirichlet_boundary_dof_x(points)

    def is_dirichlet_boundary(self) -> tuple[Callable, Callable, Callable]:
        """返回各自由度方向的 Dirichlet 边界判定函数元组."""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
            self.is_dirichlet_boundary_dof_z,
        )


# 保留旧公开名称, 两个名称指向同一个类.
HarmonicPolynomialElasticity2D = HarmonicPoly2D
HarmonicPolynomialElasticity3D = HarmonicPoly3D
