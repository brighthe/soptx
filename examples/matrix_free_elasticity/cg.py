from __future__ import annotations

from mpi4py import MPI

from fealpy.backend import backend_manager as bm

from contract import NORM_FLOOR


class OverlapInnerProduct:
    """Global inner products without double-counting shared DOF copies."""

    def __init__(self, dof_comm, local_size: int) -> None:
        self.dof_comm = dof_comm
        self.comm = MPI.COMM_WORLD
        self.references = dof_comm.refs(local_size)

    def dot(self, left, right) -> float:
        local_value = bm.sum(
            bm.conj(left) * right / self.references
        )
        return float(
            self.comm.allreduce(
                float(bm.real(local_value)),
                op=MPI.SUM,
            )
        )

    def norm(self, vector) -> float:
        return max(self.dot(vector, vector), 0.0) ** 0.5


def solve_cg(
    operator,
    rhs,
    *,
    dof_comm,
    max_iterations: int,
    rtol: float,
    atol: float,
    residual_refresh: int,
    initial=None,
):
    """Unpreconditioned CG using overlap-weighted MPI reductions."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    if residual_refresh <= 0:
        raise ValueError("residual_refresh must be positive")

    inner = OverlapInnerProduct(dof_comm, rhs.shape[0])
    solution = (
        bm.zeros_like(rhs)
        if initial is None
        else bm.asarray(initial, copy=True)
    )
    residual = rhs - operator @ solution
    direction = bm.asarray(residual, copy=True)
    residual_squared = inner.dot(residual, residual)
    rhs_norm = inner.norm(rhs)
    tolerance = max(atol, rtol * rhs_norm)
    recursive_norm = max(residual_squared, 0.0) ** 0.5
    true_norm = recursive_norm

    if true_norm <= tolerance:
        return solution, {
            "converged": True,
            "iterations": 0,
            "true_residual": true_norm,
            "recursive_residual": recursive_norm,
            "breakdown": None,
        }

    converged = False
    breakdown = None
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        operator_direction = operator @ direction
        curvature = inner.dot(direction, operator_direction)
        if curvature <= 0.0:
            breakdown = (
                "CG encountered non-positive curvature: "
                f"{curvature:.16e}"
            )
            break

        step = residual_squared / curvature
        solution = solution + step * direction
        residual = residual - step * operator_direction
        next_residual_squared = inner.dot(residual, residual)
        recursive_norm = max(next_residual_squared, 0.0) ** 0.5

        refresh = (
            recursive_norm <= tolerance
            or iteration % residual_refresh == 0
            or iteration == max_iterations
        )
        if refresh:
            true_residual = rhs - operator @ solution
            true_norm = inner.norm(true_residual)
            if dof_comm.mpi_rank == 0:
                relative = true_norm / max(rhs_norm, NORM_FLOOR)
                print(
                    "CG true residual: "
                    f"iter={iteration} abs={true_norm:.16e} "
                    f"rel={relative:.16e}",
                    flush=True,
                )
            if true_norm <= tolerance:
                converged = True
                residual = true_residual
                next_residual_squared = inner.dot(residual, residual)
                break
            if recursive_norm <= tolerance:
                residual = true_residual
                next_residual_squared = inner.dot(residual, residual)
                direction = bm.asarray(residual, copy=True)
                residual_squared = next_residual_squared
                continue

        if residual_squared <= 0.0:
            breakdown = "CG encountered a non-positive residual norm"
            break
        direction = (
            residual
            + (next_residual_squared / residual_squared) * direction
        )
        residual_squared = next_residual_squared

    true_residual = rhs - operator @ solution
    true_norm = inner.norm(true_residual)
    return solution, {
        "converged": bool(converged or true_norm <= tolerance),
        "iterations": iteration,
        "true_residual": true_norm,
        "recursive_residual": recursive_norm,
        "breakdown": breakdown,
    }
