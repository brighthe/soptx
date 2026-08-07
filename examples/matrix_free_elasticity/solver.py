"""Distributed Weighted Krylov Solvers & Diagnostics (solver.py).

Core PCG / GMRES iterative solvers for matrix-free operator systems on
distributed overlapping-DOF spaces, along with true residual and boundary error diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from fealpy.backend import backend_manager as bm
from fealpy.distributed import EntityMPI
from fealpy.solver.cg import cg
from fealpy.typing import TensorLike

import utils.contract as contract


@dataclass
class PreparedLinearSystem:
    """A boundary-conditioned matrix-free linear system."""

    operator: Any
    load: TensorLike
    prescribed: TensorLike
    boundary_dofs: TensorLike


def weighted_norm(vector: TensorLike, dof_comm: Any) -> float:
    """Compute vector norm in distributed overlapping DOF space."""
    if hasattr(dof_comm, "dot"):
        _dot, norm_fn = dof_comm.dot(vector.shape[0])
        return norm_fn(vector)
    return float(bm.linalg.norm(vector))


def solver_diagnostics(
    system: PreparedLinearSystem,
    solution: TensorLike,
    dof_comm: Any,
    cg_info: dict[str, Any],
) -> dict[str, Any]:
    """Compute true residual norm, relative residual, and boundary error."""
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
    boundary_reference_norm = weighted_norm(boundary_reference, dof_comm)

    return {
        "name": "matrix-free-weighted-cg",
        "converged": bool(cg_info["converged"]),
        "iterations": int(cg_info["niter"]),
        "reported_residual": float(
            cg_info.get("true_residual") or cg_info.get("residual", 0.0)
        ),
        "recursive_residual": float(cg_info.get("recursive_residual", 0.0)),
        "true_absolute_residual": residual_norm,
        "rhs_norm": load_norm,
        "true_relative_residual": (
            residual_norm / max(load_norm, contract.NORM_FLOOR)
        ),
        "boundary_absolute_error": boundary_absolute,
        "boundary_relative_error": (
            boundary_absolute / max(boundary_reference_norm, contract.NORM_FLOOR)
        ),
        "breakdown": cg_info.get("breakdown"),
    }


def weighted_cg(
    operator: Any,
    load: TensorLike,
    *,
    dof_comm: Any,
    x0: Optional[TensorLike] = None,
    maxiter: int = contract.DEFAULT_MAX_ITERATIONS,
    rtol: float = contract.DEFAULT_RTOL,
    atol: float = contract.DEFAULT_ATOL,
    residual_refresh: int = contract.RESIDUAL_REFRESH,
) -> tuple[TensorLike, dict[str, Any]]:
    """Distributed Conjugate Gradient (CG) solver with weighted inner product."""
    if hasattr(dof_comm, "dot"):
        dot_fn, _ = dof_comm.dot(int(load.shape[0]))
    else:
        def dot_fn(x, y):
            return float(bm.sum(x * y))

    if x0 is not None:
        x0 = bm.asarray(x0)

    solution, info = cg(
        operator,
        load,
        x0=x0,
        dot_product=dot_fn,
        residual_refresh=residual_refresh,
        atol=atol,
        rtol=rtol,
        maxit=maxiter,
        returninfo=True,
    )
    return solution, info


def solve_matrix_free_system(
    system: PreparedLinearSystem,
    dof_comm: Any,
    *,
    maxiter: int = contract.DEFAULT_MAX_ITERATIONS,
    rtol: float = contract.DEFAULT_RTOL,
    atol: float = contract.DEFAULT_ATOL,
) -> tuple[TensorLike, dict[str, Any]]:
    """Solve a prepared matrix-free system and return solution and diagnostics."""
    solution, info = weighted_cg(
        system.operator,
        system.load,
        dof_comm=dof_comm,
        x0=system.prescribed,
        maxiter=maxiter,
        rtol=rtol,
        atol=atol,
    )
    diagnostics = solver_diagnostics(system, solution, dof_comm, info)
    return solution, diagnostics
