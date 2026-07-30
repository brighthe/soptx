from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Protocol, Sequence

from fealpy.mesh import TetrahedronMesh, TriangleMesh
from fealpy.typing import TensorLike

import layout

if str(layout.SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(layout.SOURCE_ROOT))

from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
)


class DirichletElasticityProblem(Protocol):
    @property
    def domain(self) -> Sequence[float]: ...

    @property
    def lam(self) -> float: ...

    @property
    def mu(self) -> float: ...

    def body_force(self, points: TensorLike) -> TensorLike: ...
    def disp_solution(self, points: TensorLike) -> TensorLike: ...
    def grad_disp_solution(self, points: TensorLike) -> TensorLike: ...
    def dirichlet_bc(self, points: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self): ...


@dataclass(frozen=True)
class MaterialSpec:
    """Dimension-aware isotropic material data independent of the PDE object."""

    hypothesis: str
    lame_lambda: float
    shear_modulus: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "lame_lambda": self.lame_lambda,
            "shear_modulus": self.shear_modulus,
        }


@dataclass(frozen=True)
class ElasticityCase:
    """All dimension-dependent choices for one manufactured PINN problem."""

    dimension: int
    name: str
    domain: tuple[float, ...]
    problem: DirichletElasticityProblem
    material: MaterialSpec
    diagnostic_mesh_size: int

    def validate(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError(
                f"PINN elasticity supports dimension 2 or 3, received {self.dimension}."
            )
        if len(self.domain) != 2 * self.dimension:
            raise ValueError(
                f"Dimension {self.dimension} requires {2 * self.dimension} domain "
                f"bounds, received {len(self.domain)}."
            )
        for axis in range(self.dimension):
            lower = self.domain[2 * axis]
            upper = self.domain[2 * axis + 1]
            if not lower < upper:
                raise ValueError(
                    f"Domain axis {axis} must have lower < upper, received "
                    f"({lower}, {upper})."
                )

        problem_domain = tuple(float(value) for value in self.problem.domain)
        if problem_domain != self.domain:
            raise ValueError(
                f"Case domain {self.domain} does not match PDE domain {problem_domain}."
            )
        if not math.isclose(
            float(self.problem.lam),
            self.material.lame_lambda,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("Case Lamé lambda does not match the PDE data.")
        if not math.isclose(
            float(self.problem.mu),
            self.material.shear_modulus,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError("Case shear modulus does not match the PDE data.")
        if getattr(self.problem, "boundary_type", None) != "dirichlet":
            raise ValueError(
                "Only all-Dirichlet elasticity problems are supported in this baseline."
            )
        if self.dimension == 2:
            if self.material.hypothesis != "plane_strain":
                raise ValueError("The 2D PINN case requires plane_strain material data.")
            if getattr(self.problem, "plane_type", None) != "plane_strain":
                raise ValueError(
                    "Only 2D PDE data with plane_type='plane_strain' are supported."
                )
        elif self.material.hypothesis != "3D":
            raise ValueError("The 3D PINN case requires hypothesis='3D'.")

    def validate_problem_values(self, points: TensorLike) -> None:
        expected = (*points.shape[:-1], self.dimension)
        displacement = self.problem.disp_solution(points)
        body_force = self.problem.body_force(points)
        boundary = self.problem.dirichlet_bc(points)
        for name, value in (
            ("disp_solution", displacement),
            ("body_force", body_force),
            ("dirichlet_bc", boundary),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must return shape {expected}, got {tuple(value.shape)}."
                )

        gradient = self.exact_displacement_gradient(points)
        gradient_expected = (*points.shape[:-1], self.dimension, self.dimension)
        if tuple(gradient.shape) != gradient_expected:
            raise ValueError(
                "The exact displacement gradient must return shape "
                f"{gradient_expected}, got {tuple(gradient.shape)}."
            )

    def exact_displacement_gradient(self, points: TensorLike) -> TensorLike:
        gradient = getattr(self.problem, "grad_disp_solution", None)
        if gradient is None:
            gradient = getattr(self.problem, "disp_solution_gradient", None)
        if gradient is None:
            raise ValueError(
                "The manufactured problem must provide an exact "
                "displacement-gradient function."
            )
        return gradient(points)

    def create_diagnostic_mesh(self, mesh_size: int | None = None, *, device=None):
        size = self.diagnostic_mesh_size if mesh_size is None else int(mesh_size)
        if size < 2:
            raise ValueError("'mesh_size' must be at least two.")
        kwargs = {
            "nx": size - 1,
            "ny": size - 1,
        }
        if self.dimension == 3:
            kwargs["nz"] = size - 1
        if device is not None:
            kwargs["device"] = device
        if self.dimension == 2:
            return TriangleMesh.from_box(list(self.domain), **kwargs)
        return TetrahedronMesh.from_box(list(self.domain), **kwargs)


def create_case(dimension: int) -> ElasticityCase:
    """Create the supported 2D or 3D all-Dirichlet manufactured case."""

    if dimension == 2:
        domain = (0.0, 1.0, 0.0, 1.0)
        lame_lambda = 1.0
        shear_modulus = 0.5
        case = ElasticityCase(
            dimension=2,
            name="exponential-sine-plane-strain",
            domain=domain,
            problem=ExponentialSineManufacturedElasticity2D(
                domain=domain,
                lame_lambda=lame_lambda,
                shear_modulus=shear_modulus,
            ),
            material=MaterialSpec(
                hypothesis="plane_strain",
                lame_lambda=lame_lambda,
                shear_modulus=shear_modulus,
            ),
            diagnostic_mesh_size=30,
        )
    elif dimension == 3:
        domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        lame_lambda = 1.0
        shear_modulus = 1.0
        case = ElasticityCase(
            dimension=3,
            name="divergence-free-polynomial-3d",
            domain=domain,
            problem=DivergenceFreePolynomialElasticity3D(
                domain=domain,
                lame_lambda=lame_lambda,
                shear_modulus=shear_modulus,
            ),
            material=MaterialSpec(
                hypothesis="3D",
                lame_lambda=lame_lambda,
                shear_modulus=shear_modulus,
            ),
            diagnostic_mesh_size=8,
        )
    else:
        raise ValueError(f"Dimension must be 2 or 3, received {dimension}.")

    case.validate()
    return case
