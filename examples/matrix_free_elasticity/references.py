from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fem import DirichletBCOperator
from fealpy.solver import spsolve

import contract
from analyzer import build_serial_analyzer


def relative_difference(left, right) -> tuple[float, float]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    absolute = float(np.linalg.norm(left_array - right_array))
    relative = absolute / max(
        float(np.linalg.norm(right_array)),
        contract.NORM_FLOOR,
    )
    return absolute, relative


def serial_references(space, case, degree: int):
    """Build correctness-only EA/FA and direct-solve references.

    Single-rank only, so plain serial analyzers are used. Both operator levels
    come from the same class, so any discrepancy below is a genuine difference
    between the two assembly strategies rather than between two separate
    implementations.
    """

    ea = build_serial_analyzer(space, case, degree, "ea")
    fa = build_serial_analyzer(space, case, degree, "fa")

    element_form = ea.assemble_stiff_matrix()
    fa_matrix = fa.assemble_stiff_matrix()
    fa_operator, fa_load = fa.apply_bc(fa_matrix, fa.assemble_body_force_vector())

    boundary_dofs = space.is_boundary_dof(
        threshold=case.problem.is_dirichlet_boundary(),
        method="interp",
    )

    random = np.random.default_rng(contract.REFERENCE_RANDOM_SEED)
    first = bm.asarray(
        random.standard_normal(space.number_of_global_dofs()),
        dtype=bm.float64,
    )
    raw_absolute, raw_relative = relative_difference(
        element_form @ first,
        fa_matrix @ first,
    )

    element_boundary_operator = DirichletBCOperator(
        element_form,
        gd=case.problem.dirichlet_bc,
        isDDof=boundary_dofs,
    )
    boundary_absolute, boundary_relative = relative_difference(
        element_boundary_operator @ first,
        fa_operator @ first,
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

    direct_solution = spsolve(fa_operator, fa_load, solver="scipy")
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
