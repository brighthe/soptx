"""三维制造解线弹性问题."""

from __future__ import annotations

from math import isfinite
from typing import Callable, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from ._base import AllDisplacementBoundaryMixin, validated_domain


class DivergenceFreePolynomialElasticity3D(AllDisplacementBoundaryMixin):
    r"""多项式全 Dirichlet 制造解弹性问题.

    精确位移是无散的 (divergence-free), 因此体力只依赖剪切模量。
    见 ``docs/models/manufactured-elasticity.md``。
    """

    dimension = 3
    plane_type = "3D"
    boundary_type = "dirichlet"
    load_type = None
    _eps = 1.0e-12

    def __init__(
        self,
        domain: Sequence[float] = (
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ),
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 1.0,
    ) -> None:
        self._domain = validated_domain(domain, self.dimension)
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda must be finite")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be finite and positive")
        self._lam = float(lame_lambda)
        self._mu = float(shear_modulus)

    @property
    def domain(self) -> tuple[float, ...]:
        return self._domain

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        mu = self.mu
        f_x = -400 * mu * (2 * y - 1) * (2 * z - 1) * (
            3 * (x**2 - x) ** 2 * (y**2 - y + z**2 - z)
            + (1 - 6 * x + 6 * x**2)
            * (y**2 - y)
            * (z**2 - z)
        )
        f_y = 200 * mu * (2 * x - 1) * (2 * z - 1) * (
            3 * (y**2 - y) ** 2 * (x**2 - x + z**2 - z)
            + (1 - 6 * y + 6 * y**2)
            * (x**2 - x)
            * (z**2 - z)
        )
        f_z = 200 * mu * (2 * x - 1) * (2 * y - 1) * (
            3 * (z**2 - z) ** 2 * (x**2 - x + y**2 - y)
            + (1 - 6 * z + 6 * z**2)
            * (x**2 - x)
            * (y**2 - y)
        )
        return bm.stack([f_x, f_y, f_z], axis=-1)

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        mu = self.mu
        u_x = (
            200
            * mu
            * (x - x**2) ** 2
            * (2 * y**3 - 3 * y**2 + y)
            * (2 * z**3 - 3 * z**2 + z)
        )
        u_y = (
            -100
            * mu
            * (y - y**2) ** 2
            * (2 * x**3 - 3 * x**2 + x)
            * (2 * z**3 - 3 * z**2 + z)
        )
        u_z = (
            -100
            * mu
            * (z - z**2) ** 2
            * (2 * x**3 - 3 * x**2 + x)
            * (2 * y**3 - 3 * y**2 + y)
        )
        return bm.stack([u_x, u_y, u_z], axis=-1)

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        mu = self.mu

        def phi(value):
            return (value - value**2) ** 2

        def dphi(value):
            return 2 * (value - value**2) * (1 - 2 * value)

        def psi(value):
            return 2 * value**3 - 3 * value**2 + value

        def dpsi(value):
            return 6 * value**2 - 6 * value + 1

        phx, phy, phz = phi(x), phi(y), phi(z)
        dphx, dphy, dphz = dphi(x), dphi(y), dphi(z)
        psx, psy, psz = psi(x), psi(y), psi(z)
        dpsx, dpsy, dpsz = dpsi(x), dpsi(y), dpsi(z)

        return bm.stack(
            [
                bm.stack(
                    [
                        200 * mu * dphx * psy * psz,
                        200 * mu * phx * dpsy * psz,
                        200 * mu * phx * psy * dpsz,
                    ],
                    axis=-1,
                ),
                bm.stack(
                    [
                        -100 * mu * phy * dpsx * psz,
                        -100 * mu * dphy * psx * psz,
                        -100 * mu * phy * psx * dpsz,
                    ],
                    axis=-1,
                ),
                bm.stack(
                    [
                        -100 * mu * phz * dpsx * psy,
                        -100 * mu * phz * psx * dpsy,
                        -100 * mu * dphz * psx * psy,
                    ],
                    axis=-1,
                ),
            ],
            axis=-2,
        )

    def disp_solution_gradient(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.grad_disp_solution(points)

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return self.disp_solution(points)

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
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
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.is_dirichlet_boundary_dof_x(points)

    @cartesian
    def is_dirichlet_boundary_dof_z(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.is_dirichlet_boundary_dof_x(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable, Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
            self.is_dirichlet_boundary_dof_z,
        )
