"""Tests for :func:`soptx.solvers.spsolve` — the direct layer ported from fealpy.

The scipy path always runs; the MUMPS path (including the fork-only ``sym``
flag) is skipped when PyMUMPS is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fealpy.backend import backend_manager as bm

from soptx.solvers import spsolve

bm.set_backend("numpy")

pymumps_missing = False
try:
    from mumps import DMumpsContext  # noqa: F401
except ImportError:
    pymumps_missing = True

needs_mumps = pytest.mark.skipif(pymumps_missing, reason="PyMUMPS 未安装")


def spd_system(n: int = 40) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    A = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    return csr_matrix(A), b, np.linalg.solve(A, b)


def test_scipy_path_accepts_scipy_matrix() -> None:
    A, b, reference = spd_system()
    np.testing.assert_allclose(spsolve(A, b, solver="scipy"), reference,
                               atol=1e-10)


def test_unknown_solver_rejected() -> None:
    A, b, _ = spd_system(5)
    with pytest.raises(ValueError, match="Unknown solver"):
        spsolve(A, b, solver="cupy")


@needs_mumps
def test_mumps_path_accepts_scipy_matrix() -> None:
    from soptx.core.mpi_runtime import ensure_mpi_initialized
    ensure_mpi_initialized()

    A, b, reference = spd_system()
    for sym in (0, 1, 2):
        np.testing.assert_allclose(
            spsolve(A, b, solver="mumps", sym=sym), reference, atol=1e-10,
        )


@needs_mumps
def test_mumps_rejects_invalid_sym() -> None:
    from soptx.core.mpi_runtime import ensure_mpi_initialized
    ensure_mpi_initialized()

    A, b, _ = spd_system(5)
    with pytest.raises(ValueError, match="sym must be 0, 1 or 2"):
        spsolve(A, b, solver="mumps", sym=3)
