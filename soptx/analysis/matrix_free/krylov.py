from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


@dataclass
class MatrixFreeCGInfo:
    converged: bool
    iterations: int
    initial_residual: float
    final_residual: float
    elapsed_time: float
    average_matvec_time: float


class MatrixFreeCGSolver:
    """Small backend-oriented CG solver for matrix-free operators."""

    def __init__(self,
                 tol: float = 1.0e-8,
                 maxiter: int = 500,
                 preconditioner=None) -> None:
        self.tol = tol
        self.maxiter = maxiter
        self.preconditioner = preconditioner

    def _apply_preconditioner(self, r: TensorLike) -> TensorLike:
        if self.preconditioner is None:
            return r
        return self.preconditioner.apply(r)

    @staticmethod
    def _dot(x: TensorLike, y: TensorLike):
        return bm.sum(x * y)

    @staticmethod
    def _norm(x: TensorLike) -> float:
        return float(bm.linalg.norm(x))

    def solve(self, operator, rhs: TensorLike, x0: Optional[TensorLike] = None):
        start = perf_counter()
        matvec_time = 0.0

        if x0 is None:
            x = bm.zeros_like(rhs)
        else:
            x = bm.copy(x0)

        mv_start = perf_counter()
        Ax = operator.matvec(x)
        matvec_time += perf_counter() - mv_start

        r = operator.boundary.apply_rhs(rhs - Ax)
        z = self._apply_preconditioner(r)
        p = bm.copy(z)
        rz_old = self._dot(r, z)

        initial_residual = self._norm(r)
        final_residual = initial_residual
        converged = final_residual <= self.tol
        iterations = 0

        for k in range(1, self.maxiter + 1):
            if converged:
                break

            mv_start = perf_counter()
            Ap = operator.matvec(p)
            matvec_time += perf_counter() - mv_start

            alpha = rz_old / self._dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap

            final_residual = self._norm(r)
            iterations = k
            if final_residual <= self.tol:
                converged = True
                break

            z = self._apply_preconditioner(r)
            rz_new = self._dot(r, z)
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        elapsed = perf_counter() - start
        info = MatrixFreeCGInfo(
            converged=converged,
            iterations=iterations,
            initial_residual=initial_residual,
            final_residual=final_residual,
            elapsed_time=elapsed,
            average_matvec_time=matvec_time / max(iterations + 1, 1),
        )
        return x, info
