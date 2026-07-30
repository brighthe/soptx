"""Single-process tests for the overlap-weighted CG driver.

These run under plain pytest: mpi4py initializes a size-1 ``COMM_WORLD``, so
the weighted reductions in :mod:`cg` degenerate to local sums.
"""

from __future__ import annotations

import numpy as np
import pytest

import contract
from cg import solve_cg


class SerialDofComm:
    """Single-rank stand-in for FEALPy's overlapping-DOF communicator."""

    mpi_rank = 0

    def refs(self, local_size: int) -> np.ndarray:
        return np.ones(local_size, dtype=np.float64)


def spd_operator(size: int, seed: int = 20260730) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((size, size))
    return factor @ factor.T + size * np.eye(size)


def run(operator, rhs, **overrides):
    keywords = {
        "dof_comm": SerialDofComm(),
        "max_iterations": contract.DEFAULT_MAX_ITERATIONS,
        "rtol": contract.DEFAULT_RTOL,
        "atol": contract.DEFAULT_ATOL,
        "residual_refresh": contract.RESIDUAL_REFRESH,
    }
    keywords.update(overrides)
    return solve_cg(operator, rhs, **keywords)


def test_cg_matches_a_direct_solve():
    size = 24
    operator = spd_operator(size)
    expected = np.random.default_rng(11).standard_normal(size)
    rhs = operator @ expected

    solution, info = run(operator, rhs)

    assert info["converged"]
    assert info["breakdown"] is None
    assert 0 < info["iterations"] <= size
    assert np.allclose(solution, expected, rtol=0.0, atol=1.0e-8)


def test_cg_returns_before_iterating_for_a_zero_rhs():
    operator = spd_operator(8)
    rhs = np.zeros(8)

    solution, info = run(operator, rhs)

    assert info["converged"]
    assert info["iterations"] == 0
    assert info["breakdown"] is None
    assert np.all(solution == 0.0)


def test_cg_honours_a_nonzero_initial_guess():
    size = 16
    operator = spd_operator(size)
    expected = np.random.default_rng(5).standard_normal(size)
    rhs = operator @ expected

    solution, info = run(operator, rhs, initial=expected.copy())

    assert info["converged"]
    assert info["iterations"] <= 1
    assert np.allclose(solution, expected, rtol=0.0, atol=1.0e-8)


def test_cg_refreshes_the_true_residual_every_iteration():
    size = 20
    operator = spd_operator(size)
    expected = np.random.default_rng(7).standard_normal(size)
    rhs = operator @ expected

    solution, info = run(operator, rhs, residual_refresh=1)

    assert info["converged"]
    assert np.allclose(solution, expected, rtol=0.0, atol=1.0e-8)


def test_cg_reports_breakdown_for_a_negative_definite_operator():
    operator = -spd_operator(12)
    rhs = np.random.default_rng(3).standard_normal(12)

    _, info = run(operator, rhs)

    assert not info["converged"]
    assert info["breakdown"] is not None
    assert "non-positive curvature" in info["breakdown"]


def test_cg_stops_at_the_iteration_limit_without_converging():
    size = 40
    operator = spd_operator(size, seed=99)
    rhs = np.random.default_rng(13).standard_normal(size)

    _, info = run(operator, rhs, max_iterations=1, rtol=1.0e-16, atol=0.0)

    assert info["iterations"] == 1
    assert not info["converged"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_iterations": 0},
        {"rtol": -1.0e-10},
        {"atol": -1.0e-12},
        {"residual_refresh": 0},
    ],
)
def test_cg_rejects_invalid_parameters(overrides):
    operator = spd_operator(4)
    rhs = np.ones(4)

    with pytest.raises(ValueError):
        run(operator, rhs, **overrides)
