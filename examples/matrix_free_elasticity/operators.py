from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm
from fealpy.fem import (
    BilinearForm,
    DirichletBC,
    DirichletBCOperator,
    LinearForm,
)

from soptx.fem.integrators import LinearElasticIntegrator, SourceIntegrator
from cases import MaterialSpec
from distributed import OverlapOperator


@dataclass
class PreparedLinearSystem:
    """A boundary-conditioned linear system ready for CG."""

    operator: object
    load: object
    initial: object
    prescribed: object
    boundary_dofs: object
    raw_operator: object | None = None


def make_operator(
    space,
    material: MaterialSpec,
    degree: int,
    *,
    cache_elements: bool,
) -> BilinearForm:
    form = BilinearForm(space)
    form.dtype = bm.float64
    integrator = LinearElasticIntegrator(
        material=material.create(device=bm.get_device(space.mesh)),
        q=degree + 3,
        method="standard",
    )
    if cache_elements:
        integrator = integrator.const(space)
    form.add_integrator(integrator)
    return form


def make_load(space, problem, degree: int):
    form = LinearForm(space)
    form.dtype = bm.float64
    integrator = SourceIntegrator(
        source=problem.body_force,
        q=degree + 3,
    ).const(space)
    form.add_integrator(integrator)
    return form.assembly()


def prepare_ea_problem(
    space,
    dof_comm,
    problem,
    material: MaterialSpec,
    degree: int,
) -> PreparedLinearSystem:
    """Create the overlap-aware EA operator and apply Dirichlet rows."""

    local_form = make_operator(
        space,
        material,
        degree,
        cache_elements=True,
    )
    load = dof_comm.sync_add(make_load(space, problem, degree))
    boundary_dofs = space.is_boundary_dof(
        threshold=problem.is_dirichlet_boundary(),
        method="interp",
    )
    overlap_form = OverlapOperator(local_form, dof_comm)
    operator = DirichletBCOperator(
        overlap_form,
        gd=problem.dirichlet_bc,
        isDDof=boundary_dofs,
    )
    initial = operator.init_solution(dtype=bm.float64)
    prescribed = bm.asarray(initial, copy=True)
    initial[~boundary_dofs] = 0.0
    load = operator.apply(load, initial)
    return PreparedLinearSystem(
        operator=operator,
        load=load,
        initial=initial,
        prescribed=prescribed,
        boundary_dofs=boundary_dofs,
    )


def prepare_serial_fa_problem(
    space,
    problem,
    material: MaterialSpec,
    degree: int,
) -> PreparedLinearSystem:
    """Assemble the single-rank FA CSR system and apply Dirichlet rows."""

    assembled_form = make_operator(
        space,
        material,
        degree,
        cache_elements=False,
    )
    raw_operator = assembled_form.assembly(format="csr")
    boundary_threshold = problem.is_dirichlet_boundary()
    boundary_dofs = space.is_boundary_dof(
        threshold=boundary_threshold,
        method="interp",
    )
    boundary = DirichletBC(
        space,
        gd=problem.dirichlet_bc,
        threshold=boundary_threshold,
        method="interp",
    )
    initial = space.function(dtype=bm.float64)
    load = boundary.apply_vector(
        make_load(space, problem, degree),
        raw_operator,
        uh=initial,
    )
    operator = boundary.apply_matrix(raw_operator)
    prescribed = bm.asarray(initial, copy=True)
    initial[~boundary_dofs] = 0.0
    return PreparedLinearSystem(
        operator=operator,
        load=load,
        initial=initial,
        prescribed=prescribed,
        boundary_dofs=boundary_dofs,
        raw_operator=raw_operator,
    )


def prepare_problem(
    space,
    problem,
    material: MaterialSpec,
    degree: int,
    *,
    operator_level: str,
    dof_comm,
) -> PreparedLinearSystem:
    """Dispatch to the EA or FA system builder for one operator level."""

    if operator_level == "fa":
        return prepare_serial_fa_problem(space, problem, material, degree)
    if operator_level == "ea":
        return prepare_ea_problem(
            space,
            dof_comm,
            problem,
            material,
            degree,
        )
    raise ValueError(f"unknown operator level: {operator_level!r}")
