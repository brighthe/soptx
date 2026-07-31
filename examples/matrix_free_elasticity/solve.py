from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm

import contract
from cg import OverlapInnerProduct


@dataclass
class PreparedLinearSystem:
    """A boundary-conditioned linear system, as returned by the analyzer."""

    operator: object
    load: object
    prescribed: object
    boundary_dofs: object


def weighted_norm(vector, dof_comm) -> float:
    return OverlapInnerProduct(dof_comm, vector.shape[0]).norm(vector)


def solver_diagnostics(
    system: PreparedLinearSystem,
    solution,
    dof_comm,
    cg_info: dict,
) -> dict:
    residual = system.operator @ solution - system.load
    residual_norm = weighted_norm(residual, dof_comm)
    load_norm = weighted_norm(system.load, dof_comm)
    boundary_error = bm.where(
        system.boundary_dofs,
        solution - system.prescribed,
        bm.zeros_like(solution),
    )
    boundary_reference = bm.where(
        system.boundary_dofs,
        system.prescribed,
        bm.zeros_like(system.prescribed),
    )
    boundary_absolute = weighted_norm(boundary_error, dof_comm)
    boundary_reference_norm = weighted_norm(
        boundary_reference,
        dof_comm,
    )
    return {
        "name": "unpreconditioned-cg",
        "converged": bool(cg_info["converged"]),
        "iterations": int(cg_info["iterations"]),
        "reported_residual": float(cg_info["true_residual"]),
        "recursive_residual": float(cg_info["recursive_residual"]),
        "true_absolute_residual": residual_norm,
        "rhs_norm": load_norm,
        "true_relative_residual": (
            residual_norm / max(load_norm, contract.NORM_FLOOR)
        ),
        "boundary_absolute_error": boundary_absolute,
        "boundary_relative_error": (
            boundary_absolute
            / max(boundary_reference_norm, contract.NORM_FLOOR)
        ),
        "breakdown": cg_info["breakdown"],
    }
