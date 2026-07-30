"""Public structural protocols."""

from __future__ import annotations

from typing import Protocol, Sequence

from fealpy.typing import TensorLike


class ElasticityProblem(Protocol):
    """Mathematical contract for an all-Dirichlet elasticity problem.

    A Problem owns equations and boundary data only.  It must not create a
    mesh or own a material object.
    """

    dimension: int

    @property
    def domain(self) -> Sequence[float]: ...

    def body_force(self, points: TensorLike) -> TensorLike: ...

    def disp_solution(self, points: TensorLike) -> TensorLike: ...

    def dirichlet_bc(self, points: TensorLike) -> TensorLike: ...

    def is_dirichlet_boundary(self): ...


class MaterialInterpolation(Protocol):
    """Interface consumed by FEM analyzers for density interpolation."""

    density_location: str
    n_sub: int

    def interpolate_material(self, *args, **kwargs): ...

    def interpolate_material_derivative(self, *args, **kwargs): ...
