"""Single-process tests for :func:`fealpy.solver.cg` with custom dot_product.

These run under plain pytest without MPI.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.matrix_free_evidence import contract
from fealpy.solver.cg import cg


class SerialDofComm:
    """Single-rank stand-in for FEALPy's overlapping-DOF communicator."""

    mpi_rank = 0

    def refs(self, local_size: int) -> np.ndarray:
        return np.ones(local_size, dtype=np.float64)

    def dot(self, local_size: int):
        """Simulate EntityMPI.dot() for serial testing."""
        refs = self.refs(local_size)

        def _dot(x, y):
            return float(np.sum(x * y / refs))

        def _norm(x):
            return max(_dot(x, x), 0.0) ** 0.5

        return _dot, _norm


def spd_operator(size: int, seed: int = 20260730) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((size, size))
    return factor @ factor.T + size * np.eye(size)


def run(operator, rhs, **overrides):
    dof_comm = SerialDofComm()
    dot_fn, _norm_fn = dof_comm.dot(rhs.shape[0])

    keywords = {
        "A": operator,
        "b": rhs,
        "dot_product": dot_fn,
        "maxit": contract.DEFAULT_MAX_ITERATIONS,
        "rtol": contract.DEFAULT_RTOL,
        "atol": contract.DEFAULT_ATOL,
        "residual_refresh": contract.RESIDUAL_REFRESH,
        "returninfo": True,
    }
    keywords.update(overrides)
    return cg(**keywords)


def test_cg_matches_a_direct_solve():
    size = 24
    operator = spd_operator(size)
    expected = np.random.default_rng(11).standard_normal(size)
    rhs = operator @ expected

    solution, info = run(operator, rhs)

    assert info["converged"]
    assert info["breakdown"] is None
    assert 0 < info["niter"] <= size
    assert np.allclose(solution, expected, rtol=0.0, atol=1.0e-8)


def test_cg_returns_before_iterating_for_a_zero_rhs():
    operator = spd_operator(8)
    rhs = np.zeros(8)

    solution, info = run(operator, rhs)

    assert info["converged"]
    assert info["niter"] == 0
    assert np.allclose(solution, rhs)


def test_cg_flags_breakdown_when_curvature_is_non_positive():
    size = 8
    rhs = np.random.default_rng(3).standard_normal(size)

    class SignFlipOperator:
        def __matmul__(self, other):
            return -other  # negative eigenvalues

    solution, info = run(
        SignFlipOperator(), rhs,
        residual_refresh=0,  # no true residual refresh needed
    )

    assert not info["converged"]
    assert info["breakdown"] is not None
