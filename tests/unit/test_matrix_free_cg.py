"""Single-process tests for the overlapping-DOF path: :func:`soptx.solvers.cg`
with a custom dot_product, and the :mod:`soptx.solvers.overlap` adapter that
injects it.

These run under plain pytest without MPI.
"""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from tools.matrix_free_evidence import contract
from soptx.solvers import cg
from soptx.solvers.cg import NORM_TYPES
from soptx.solvers.overlap import weighted_cg, weighted_norm

# TensorLike 的 isinstance 判定依赖已注册的 backend
bm.set_backend("numpy")


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
# --------------------------------------------------------------------------
# weighted_cg: 把 dof_comm 的重叠加权内积注入 CGSolver 的 dot_product 扩展点
# --------------------------------------------------------------------------

class OverlapDofComm:
    """交界面自由度被两个 rank 各存一份, 权重取 1/2 以消除重复计数."""

    mpi_rank = 0

    def __init__(self, weights: np.ndarray) -> None:
        self._weights = weights

    def dot(self, local_size: int):
        weights = self._weights[:local_size]

        def _dot(x, y):
            return float(np.sum(weights * x * y))

        def _norm(x):
            return max(_dot(x, x), 0.0) ** 0.5

        return _dot, _norm


def overlap_setup(size: int = 60, shared: int = 8):
    """构造与重叠布局代数形态一致的一组算子/权重/右端项.

    CG 要求算子对所用内积自伴, 即 ``W @ A`` 对称. 重叠布局下"本地算子 +
    重复计数权重"的形态正是 :math:`A_w = W^{-1} S`, S 为 SPD; 拿任意 SPD
    矩阵配任意权重是不自伴的, CG 在其上本就不该收敛.
    """
    S = spd_operator(size, seed=20260901)
    weights = np.ones(size)
    weights[:shared] = 0.5
    operator = (1.0 / weights)[:, None] * S
    rhs = np.random.default_rng(31).standard_normal(size)
    return operator, weights, rhs


def test_weighted_norm_discounts_shared_dofs():
    weights = np.ones(6)
    weights[:2] = 0.5
    vector = np.ones(6)

    assert weighted_norm(vector, OverlapDofComm(weights)) == pytest.approx(
        np.sqrt(5.0))
    # dof_comm 为 None (串行) 时退化为普通 2-范数
    assert weighted_norm(vector, None) == pytest.approx(np.sqrt(6.0))


def test_weighted_cg_serial_fallback_matches_a_direct_solve():
    operator = spd_operator(30, seed=32)
    rhs = np.random.default_rng(32).standard_normal(30)

    solution, info = weighted_cg(operator, rhs, dof_comm=None, rtol=1.0e-12,
                                 maxiter=3000)

    assert info["converged"]
    assert np.allclose(solution, np.linalg.solve(operator, rhs), atol=1.0e-8)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_weighted_cg_norm_type_is_inert_without_a_preconditioner(norm_type):
    """本函数不注入 M, 三个取值数值上重合; 参数只为让上层显式声明判据."""
    operator = spd_operator(30, seed=33)
    rhs = np.random.default_rng(33).standard_normal(30)

    reference, ref_info = weighted_cg(operator, rhs, dof_comm=None,
                                      rtol=1.0e-10, maxiter=3000)
    solution, info = weighted_cg(operator, rhs, dof_comm=None, rtol=1.0e-10,
                                 maxiter=3000, norm_type=norm_type)

    assert info["niter"] == ref_info["niter"]
    assert np.array_equal(solution, reference)


def test_weighted_cg_rejects_an_unknown_norm_type():
    with pytest.raises(ValueError, match="norm_type"):
        weighted_cg(spd_operator(8), np.ones(8), dof_comm=None,
                    norm_type="bogus")


def test_weighted_cg_converges_under_the_overlap_inner_product():
    operator, weights, rhs = overlap_setup()
    dof_comm = OverlapDofComm(weights)
    # 自伴性是前提, 先钉住构造本身
    assert np.allclose(weights[:, None] * operator,
                       (weights[:, None] * operator).T)

    solution, info = weighted_cg(operator, rhs, dof_comm=dof_comm,
                                 rtol=1.0e-10, maxiter=3000)

    assert info["converged"] and info["reason"] > 0
    relerr = (np.linalg.norm(operator @ solution - rhs)
              / np.linalg.norm(rhs))
    assert relerr < 1.0e-8


def test_weighted_cg_warm_start_reports_a_true_residual():
    operator, weights, rhs = overlap_setup()
    dof_comm = OverlapDofComm(weights)
    rough, _ = weighted_cg(operator, rhs, dof_comm=dof_comm, rtol=1.0e-3,
                           maxiter=3000)

    solution, info = weighted_cg(operator, rhs, dof_comm=dof_comm, x0=rough,
                                 rtol=1.0e-10, maxiter=3000,
                                 residual_refresh=25)

    assert info["converged"]
    assert info["true_residual"] is not None
    relerr = (np.linalg.norm(operator @ solution - rhs)
              / np.linalg.norm(rhs))
    assert relerr < 1.0e-8
