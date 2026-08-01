"""Two-dimensional manufactured linear-elasticity problems.

The full equations and derivations are documented in
``docs/models/manufactured-elasticity.md``.
"""

from __future__ import annotations

from math import isfinite
from typing import Callable, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from ._base import AllDisplacementBoundaryMixin, validated_domain


class _AllDirichletElasticity2D(AllDisplacementBoundaryMixin):
    dimension = 2
    plane_type = "plane_strain"
    boundary_type = "dirichlet"
    load_type = None
    _eps = 1.0e-12

    def __init__(self, domain: Sequence[float]) -> None:
        self._domain = validated_domain(domain, self.dimension)

    @property
    def domain(self) -> tuple[float, ...]:
        return self._domain

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return self.disp_solution(points)

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        domain = self.domain
        return (
            (bm.abs(x - domain[0]) < self._eps)
            | (bm.abs(x - domain[1]) < self._eps)
            | (bm.abs(y - domain[2]) < self._eps)
            | (bm.abs(y - domain[3]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.is_dirichlet_boundary_dof_x(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )


class ExponentialSineManufacturedElasticity2D(
    _AllDirichletElasticity2D
):
    r"""Plane-strain manufactured problem with exponential/sine displacement.

    .. math:: -\nabla\cdot\sigma(u)=b,\quad
       u=(e^{x-y}x(1-x)y(1-y),\sin(\pi x)\sin(\pi y)).
    """

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 0.5,
    ) -> None:
        super().__init__(domain)
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda must be finite")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be finite and positive")
        self._lam = float(lame_lambda)
        self._mu = float(shear_modulus)

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        return -self.div_stress_solution(points)

    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi
        deps_xx_dx = (
            exp_xy * y * (1 - y) * (-x**2 - 3 * x)
        )
        deps_xx_dy = (
            exp_xy
            * (1 - x - x**2)
            * (1 - 3 * y + y**2)
        )
        deps_yy_dx = pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        deps_yy_dy = -pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        deps_xy_dx = 0.5 * (
            exp_xy
            * (1 - 3 * y + y**2)
            * (1 - x - x**2)
            - pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        )
        deps_xy_dy = 0.5 * (
            exp_xy * x * (1 - x) * (-4 + 5 * y - y**2)
            + pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        )
        div_x = (
            (self.lam + 2 * self.mu) * deps_xx_dx
            + self.lam * deps_yy_dx
            + 2 * self.mu * deps_xy_dy
        )
        div_y = (
            2 * self.mu * deps_xy_dx
            + self.lam * deps_xx_dy
            + (self.lam + 2 * self.mu) * deps_yy_dy
        )
        return bm.stack([div_x, div_y], axis=-1)

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return bm.stack(
            [
                bm.exp(x - y) * x * (1 - x) * y * (1 - y),
                bm.sin(bm.pi * x) * bm.sin(bm.pi * y),
            ],
            axis=-1,
        )

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        du1_dx = exp_xy * y * (1 - y) * (1 - x - x**2)
        du1_dy = exp_xy * x * (1 - x) * (1 - 3 * y + y**2)
        du2_dx = (
            bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        )
        du2_dy = (
            bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        )
        return bm.stack(
            [
                bm.stack([du1_dx, du1_dy], axis=-1),
                bm.stack([du2_dx, du2_dy], axis=-1),
            ],
            axis=-2,
        )

    def disp_solution_gradient(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.grad_disp_solution(points)

    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        gradient = self.grad_disp_solution(points)
        eps_xx = gradient[..., 0, 0]
        eps_yy = gradient[..., 1, 1]
        eps_xy = 0.5 * (
            gradient[..., 0, 1] + gradient[..., 1, 0]
        )
        trace = eps_xx + eps_yy
        return bm.stack(
            [
                self.lam * trace + 2 * self.mu * eps_xx,
                2 * self.mu * eps_xy,
                self.lam * trace + 2 * self.mu * eps_yy,
            ],
            axis=-1,
        )


class MixedBoundaryExponentialSineElasticity2D(
    ExponentialSineManufacturedElasticity2D
):
    """Mixed-boundary view of the exponential/sine manufactured problem.

    The exact displacement vanishes on the unit-square boundary.  Three sides
    are treated as displacement boundaries and the right side as a nonzero
    traction boundary, so the same exact fields exercise strong traction data
    without introducing a second formula source.
    """

    boundary_type = "mixed"
    load_type = "distributed"

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        domain = self.domain
        return (
            (bm.abs(x - domain[0]) < self._eps)
            | (bm.abs(y - domain[2]) < self._eps)
            | (bm.abs(y - domain[3]) < self._eps)
        )

    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        return bm.abs(points[..., 0] - self.domain[1]) < self._eps

    def traction_bc(self, points: TensorLike) -> TensorLike:
        return self.stress_solution(points)


class SinusoidalPlaneStrainElasticity2D(
    _AllDirichletElasticity2D
):
    r"""Unit-square plane-strain manufactured displacement problem.

    .. math:: u=(\sin(\pi x)\sin(\pi y),0),\quad
       -\nabla\cdot\sigma(u)=b.
    """

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        *,
        youngs_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
    ) -> None:
        super().__init__(domain)
        youngs_modulus = float(youngs_modulus)
        poisson_ratio = float(poisson_ratio)
        if not isfinite(youngs_modulus) or youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be finite and positive")
        if (
            not isfinite(poisson_ratio)
            or not -1.0 < poisson_ratio < 0.5
        ):
            raise ValueError(
                "poisson_ratio must be finite and satisfy -1 < nu < 0.5"
            )
        self._E = youngs_modulus
        self._nu = poisson_ratio
        self._mu = youngs_modulus / (2 * (1 + poisson_ratio))
        self._lam = (
            youngs_modulus
            * poisson_ratio
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        return bm.stack(
            [
                (self.lam + 3 * self.mu)
                * pi**2
                * bm.sin(pi * x)
                * bm.sin(pi * y),
                -(self.lam + self.mu)
                * pi**2
                * bm.cos(pi * x)
                * bm.cos(pi * y),
            ],
            axis=-1,
        )

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return bm.stack(
            [
                bm.sin(bm.pi * x) * bm.sin(bm.pi * y),
                bm.zeros_like(x),
            ],
            axis=-1,
        )

    @cartesian
    def disp_solution_gradient(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        du_x_dx = (
            bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        )
        du_x_dy = (
            bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        )
        zero = bm.zeros_like(x)
        return bm.stack(
            [
                bm.stack([du_x_dx, du_x_dy], axis=-1),
                bm.stack([zero, zero], axis=-1),
            ],
            axis=-2,
        )

    def grad_disp_solution(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.disp_solution_gradient(points)
