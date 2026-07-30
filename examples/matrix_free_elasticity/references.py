from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fem import DirichletBCOperator
from fealpy.solver import spsolve

import contract
from operators import make_operator, prepare_serial_fa_problem


def relative_difference(left, right) -> tuple[float, float]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    absolute = float(np.linalg.norm(left_array - right_array))
    relative = absolute / max(
        float(np.linalg.norm(right_array)),
        contract.NORM_FLOOR,
    )
    return absolute, relative


def serial_references(space, problem, material, degree: int):
    """Build correctness-only EA/FA and direct-solve references."""

    element_form = make_operator(
        space,
        material,
        degree,
        cache_elements=True,
    )
    fa_system = prepare_serial_fa_problem(
        space,
        problem,
        material,
        degree,
    )

    random = np.random.default_rng(contract.REFERENCE_RANDOM_SEED)
    first = bm.asarray(
        random.standard_normal(space.number_of_global_dofs()),
        dtype=bm.float64,
    )
    raw_absolute, raw_relative = relative_difference(
        element_form @ first,
        fa_system.raw_operator @ first,
    )

    element_boundary_operator = DirichletBCOperator(
        element_form,
        gd=problem.dirichlet_bc,
        isDDof=fa_system.boundary_dofs,
    )
    boundary_absolute, boundary_relative = relative_difference(
        element_boundary_operator @ first,
        fa_system.operator @ first,
    )

    second = bm.asarray(
        random.standard_normal(space.number_of_global_dofs()),
        dtype=bm.float64,
    )
    first_action = element_boundary_operator @ first
    second_action = element_boundary_operator @ second
    first_pairing = float(bm.sum(first * second_action))
    second_pairing = float(bm.sum(second * first_action))
    symmetry_relative = abs(first_pairing - second_pairing) / max(
        abs(first_pairing),
        abs(second_pairing),
        contract.NORM_FLOOR,
    )
    energy = float(bm.sum(first * first_action))

    direct_solution = spsolve(
        fa_system.operator,
        fa_system.load,
        solver="scipy",
    )
    return (
        {
            "raw_absolute_error": raw_absolute,
            "raw_relative_error": raw_relative,
            "dirichlet_absolute_error": boundary_absolute,
            "dirichlet_relative_error": boundary_relative,
            "symmetry_relative_error": symmetry_relative,
            "random_vector_energy": energy,
        },
        np.asarray(direct_solution),
    )
