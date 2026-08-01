"""Public structural protocols.

Each protocol describes what one consumer actually requires, so a Problem that
satisfies a protocol can be handed to the analyzer annotated with it.  Adding a
new ``pde`` member to an analyzer therefore requires extending the matching
protocol here; ``tests/unit/test_problem_protocol_conformance.py`` enforces
that.

The protocols are ``runtime_checkable``, which validates member *presence*
only -- never signatures.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from fealpy.typing import TensorLike


@runtime_checkable
class ElasticityProblem(Protocol):
    """What every elasticity analyzer may assume unconditionally.

    A Problem owns equations and boundary data only.  It must not create a
    mesh or own a material object.
    """

    dimension: int

    @property
    def domain(self) -> Sequence[float]: ...

    def body_force(self, points: TensorLike) -> TensorLike: ...


@runtime_checkable
class DirichletElasticityProblem(ElasticityProblem, Protocol):
    """Contract consumed by ``LagrangeFEMAnalyzer``.

    The analyzer dispatches on ``boundary_type`` and ``load_type``, and each
    branch requires further members that are deliberately not modelled here
    yet:

    * ``load_type == 'distributed'``: ``neumann_bc``, ``is_neumann_boundary``
      and the optional ``set_equivalent_traction`` hook;
    * ``load_type == 'concentrated'``: ``concentrate_load_bc`` and
      ``is_concentrate_load_boundary``;
    * adjoint right-hand sides: ``adjoint_load_bc`` and
      ``is_adjoint_load_boundary``;
    * spring supports: ``k_in``, ``k_out`` and ``is_spring_boundary``.

    Modelling that tagged union is follow-up work; until then the conformance
    test carries these names in its allow list.
    """

    boundary_type: str
    load_type: Optional[str]

    def dirichlet_bc(self, points: TensorLike) -> TensorLike: ...

    def is_dirichlet_boundary(self): ...


@runtime_checkable
class MixedBoundaryElasticityProblem(ElasticityProblem, Protocol):
    """Contract consumed by ``HuZhangMFEMAnalyzer``.

    The mixed formulation splits the boundary the other way round from the
    primal one: displacement data is imposed weakly (natural) and traction
    data strongly (essential), so the two predicates must partition the
    boundary explicitly.  ``mark_corners`` feeds the Hu-Zhang corner
    relaxation.
    """

    def mark_corners(self, node: TensorLike) -> TensorLike: ...

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike: ...

    def is_traction_boundary(self, points: TensorLike) -> TensorLike: ...

    def traction_bc(self, points: TensorLike) -> TensorLike: ...


@runtime_checkable
class MaterialInterpolation(Protocol):
    """Interface consumed by FEM analyzers for density interpolation."""

    density_location: str
    n_sub: int

    def interpolate_material(self, *args, **kwargs): ...

    def interpolate_material_derivative(self, *args, **kwargs): ...
