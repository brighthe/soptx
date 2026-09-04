"""Tests for :class:`soptx.solvers.DirectSolver`.

Covers capability negotiation (the first solver to declare ``CAP_MATRIX``),
factorization reuse across right-hand sides, and the info contract. The scipy
path always runs; the MUMPS path is skipped when PyMUMPS is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fealpy.backend import backend_manager as bm

from soptx.solvers import DirectSolver, OperatorCapabilityError

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


class MatvecOnlyOperator:
    """站位的 'ea' 层级算子: 只支持 ``@``, 给不出矩阵."""

    def __init__(self, A) -> None:
        self._A = A

    def __matmul__(self, other):
        return self._A @ other


# --------------------------------------------------------------------------
# 能力协商
# --------------------------------------------------------------------------

def test_matrix_free_operator_rejected_at_setup() -> None:
    A, _, _ = spd_system(10)
    solver = DirectSolver("scipy")
    with pytest.raises(OperatorCapabilityError, match="matrix"):
        solver.setup(MatvecOnlyOperator(A))


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="未知的直接法后端"):
        DirectSolver("cupy")


def test_invalid_sym_rejected() -> None:
    with pytest.raises(ValueError, match="sym must be 0, 1 or 2"):
        DirectSolver("mumps", sym=3)


def test_solve_before_setup_raises() -> None:
    _, b, _ = spd_system(10)
    solver = DirectSolver("scipy")
    with pytest.raises(RuntimeError, match="尚未 setup"):
        solver.solve(b)


def test_op_property_before_setup_raises() -> None:
    solver = DirectSolver("scipy")
    assert solver.is_setup is False
    with pytest.raises(RuntimeError, match="尚未 setup"):
        _ = solver.op


# --------------------------------------------------------------------------
# scipy 路径
# --------------------------------------------------------------------------

def test_scipy_solve_matches_reference() -> None:
    A, b, reference = spd_system()
    x, info = DirectSolver("scipy").setup(A).solve(b)
    np.testing.assert_allclose(x, reference, atol=1e-10)
    assert info["niter"] == 1
    assert info["converged"] is True
    assert info["relres"] < 1e-10


def test_setup_returns_self_for_chaining() -> None:
    A, _, _ = spd_system(10)
    solver = DirectSolver("scipy")
    assert solver.setup(A) is solver
    assert solver.is_setup is True


def test_factorization_reused_across_right_hand_sides() -> None:
    """一次 setup 解多个右端 —— 状态解与伴随解共用同一个分解."""
    A, b, reference = spd_system()
    solver = DirectSolver("scipy").setup(A)

    x1, _ = solver.solve(b)
    np.testing.assert_allclose(x1, reference, atol=1e-10)

    b2 = np.ones_like(b)
    x2, _ = solver.solve(b2)
    np.testing.assert_allclose(x2, np.linalg.solve(A.toarray(), b2),
                               atol=1e-10)


def test_solve_does_not_destroy_caller_matrix() -> None:
    """SuperLU 会原地改写输入; setup 内部的复制必须挡住这一点."""
    A, b, _ = spd_system()
    before = A.toarray().copy()
    DirectSolver("scipy").setup(A).solve(b)
    np.testing.assert_array_equal(A.toarray(), before)


def test_scipy_accepts_multiple_right_hand_sides() -> None:
    A, b, reference = spd_system()
    B = np.stack([b, np.ones_like(b)], axis=1)
    x, info = DirectSolver("scipy").setup(A).solve(B)
    assert x.shape == B.shape
    np.testing.assert_allclose(x[:, 0], reference, atol=1e-10)
    assert info["converged"] is True


def test_matmul_gives_preconditioner_mode() -> None:
    """求解器与预条件子是同一个类型: ``M @ b`` 应与 solve 一致."""
    A, b, reference = spd_system()
    solver = DirectSolver("scipy").setup(A)
    np.testing.assert_allclose(solver @ b, reference, atol=1e-10)
    np.testing.assert_allclose(solver.apply(b), reference, atol=1e-10)


def test_zero_right_hand_side_reports_zero_residual() -> None:
    A, b, _ = spd_system(10)
    x, info = DirectSolver("scipy").setup(A).solve(np.zeros_like(b))
    np.testing.assert_allclose(x, np.zeros_like(b), atol=1e-12)
    assert info["relres"] == 0.0
    assert info["converged"] is True


def test_resetup_rebinds_operator() -> None:
    A1, b, _ = spd_system(10)
    solver = DirectSolver("scipy").setup(A1)
    A2 = csr_matrix(3.0 * np.eye(10))
    solver.setup(A2)
    x, _ = solver.solve(b)
    np.testing.assert_allclose(x, b / 3.0, atol=1e-12)


# --------------------------------------------------------------------------
# MUMPS 路径
# --------------------------------------------------------------------------

@needs_mumps
def test_mumps_solve_matches_reference_for_all_sym() -> None:
    A, b, reference = spd_system()
    for sym in (0, 1, 2):
        with DirectSolver("mumps", sym=sym) as solver:
            x, info = solver.setup(A).solve(b)
            np.testing.assert_allclose(x, reference, atol=1e-10)
            assert info["converged"] is True


@needs_mumps
def test_mumps_factorization_reused_across_right_hand_sides() -> None:
    A, b, reference = spd_system()
    with DirectSolver("mumps", sym=1) as solver:
        solver.setup(A)
        x1, _ = solver.solve(b)
        np.testing.assert_allclose(x1, reference, atol=1e-10)

        b2 = np.ones_like(b)
        x2, _ = solver.solve(b2)
        np.testing.assert_allclose(x2, np.linalg.solve(A.toarray(), b2),
                                   atol=1e-10)


@needs_mumps
def test_mumps_rejects_multiple_right_hand_sides() -> None:
    A, b, _ = spd_system(10)
    B = np.stack([b, b], axis=1)
    with DirectSolver("mumps") as solver:
        solver.setup(A)
        with pytest.raises(NotImplementedError, match="一维右端项"):
            solver.solve(B)


@needs_mumps
def test_mumps_close_releases_context() -> None:
    A, b, _ = spd_system(10)
    solver = DirectSolver("mumps").setup(A)
    solver.solve(b)
    solver.close()
    assert solver.is_setup is False
    with pytest.raises(RuntimeError, match="尚未 setup"):
        solver.solve(b)
