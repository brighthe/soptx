"""求解层 ``JacobiSmoother`` 与 ``Multigrid`` 的测试 (一维 Poisson 模型问题)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from soptx.backend import backend_manager as bm
from soptx.solvers import DirectSolver, JacobiSmoother, Multigrid, create
from soptx.sparse import CSRTensor


def _poisson(n: int) -> CSRTensor:
    """n 个内点的一维 Poisson 刚度矩阵 tridiag(-1, 2, -1)."""
    return CSRTensor.from_scipy(sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr'))


def _linear_prolongation(n_coarse: int) -> CSRTensor:
    """粗网格 n_coarse 个内点到细网格 2 n_coarse + 1 个内点的线性插值."""
    n_fine = 2 * n_coarse + 1
    rows, cols, vals = [], [], []
    for j in range(n_coarse):
        center = 2 * j + 1
        rows += [center - 1, center, center + 1]
        cols += [j, j, j]
        vals += [0.5, 1.0, 0.5]
    return CSRTensor.from_scipy(sp.csr_matrix((vals, (rows, cols)), shape=(n_fine, n_coarse)))


def _hierarchy(n_coarsest: int, num_levels: int, **kwargs) -> tuple[Multigrid, CSRTensor]:
    """由粗到细的 Galerkin 层次; 返回未 setup 的 Multigrid 与最细层矩阵."""
    sizes = [n_coarsest]
    for _ in range(num_levels - 1):
        sizes.append(2 * sizes[-1] + 1)
    fine = _poisson(sizes[-1])
    ops = [fine]
    prolongations = []
    for level in range(num_levels - 1, 0, -1):
        P = _linear_prolongation(sizes[level - 1])
        prolongations.insert(0, P)
        ops.insert(0, P.T @ ops[0] @ P)
    mg = Multigrid(coarse_solver=DirectSolver(), **kwargs)
    mg.add_level(ops[0])
    for level in range(1, num_levels):
        mg.add_level(None if level == num_levels - 1 else ops[level], JacobiSmoother(omega=0.6),
                     prolongations[level - 1])
    return mg, fine


def test_jacobi_smoother_matches_hand_iteration() -> None:
    bm.set_backend('numpy')
    A = _poisson(9)
    b = bm.tensor(np.random.default_rng(0).standard_normal(9))
    x0 = bm.tensor(np.random.default_rng(1).standard_normal(9))
    dense = A.to_scipy().toarray()
    smoother = JacobiSmoother(omega=0.6, sweeps=3).setup(A)

    expected = np.asarray(x0).copy()
    for _ in range(3):
        expected = expected + 0.6 * (np.asarray(b) - dense @ expected) / np.diag(dense)
    x, info = smoother.solve(b, x0)
    np.testing.assert_allclose(np.asarray(x), expected, rtol=0, atol=1e-14)
    assert info['niter'] == 3 and info['converged'] is False

    # 零初值的首次扫描化为 omega D^{-1} b
    x_zero, _ = smoother.solve(b)
    expected = np.zeros(9)
    for _ in range(3):
        expected = expected + 0.6 * (np.asarray(b) - dense @ expected) / np.diag(dense)
    np.testing.assert_allclose(np.asarray(x_zero), expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize('kwargs', [dict(omega=0.0), dict(sweeps=0), dict(sweeps=1.5)])
def test_jacobi_smoother_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        JacobiSmoother(**kwargs)


def test_jacobi_smoother_with_explicit_diagonal_needs_no_capability() -> None:
    bm.set_backend('numpy')
    smoother = JacobiSmoother(diag=bm.tensor([2.0, 4.0]))
    assert smoother.requires == frozenset()
    with pytest.raises(ValueError):
        JacobiSmoother(diag=bm.tensor([1.0, 0.0]))


@pytest.mark.parametrize('num_levels', [2, 3, 4])
def test_vcycle_is_symmetric(num_levels) -> None:
    bm.set_backend('numpy')
    mg, fine = _hierarchy(3, num_levels)
    mg.setup(fine)
    rng = np.random.default_rng(2)
    v = bm.tensor(rng.standard_normal(fine.shape[0]))
    w = bm.tensor(rng.standard_normal(fine.shape[0]))
    lhs, rhs = float(v @ (mg @ w)), float(w @ (mg @ v))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), 1.0)


def test_mgcg_converges_in_few_iterations_independent_of_size() -> None:
    bm.set_backend('numpy')
    iterations = []
    for num_levels in (4, 6):
        mg, fine = _hierarchy(3, num_levels)
        mg.setup(fine)
        b = bm.ones((fine.shape[0], ), dtype=bm.float64)
        solver = create('cg', M=mg, atol=0.0, rtol=1e-10, maxit=200, norm_type='unpreconditioned')
        x, info = solver.setup(fine).solve(b)
        assert info['converged']
        reference = sp.linalg.spsolve(fine.to_scipy().tocsc(), np.asarray(b))
        np.testing.assert_allclose(np.asarray(x), reference, rtol=1e-8)
        iterations.append(info['niter'])
    assert max(iterations) <= 15 and iterations[1] <= iterations[0] + 3


def test_single_level_reduces_to_coarse_solve() -> None:
    bm.set_backend('numpy')
    A = _poisson(5)
    mg = Multigrid(coarse_solver=DirectSolver()).add_level(None).setup(A)
    b = bm.ones((5, ), dtype=bm.float64)
    x, info = mg.solve(b)
    np.testing.assert_allclose(np.asarray(A @ x), np.asarray(b), atol=1e-13)
    assert info['niter'] == 1


def test_solve_from_initial_guess_is_residual_correction() -> None:
    bm.set_backend('numpy')
    mg, fine = _hierarchy(3, 3)
    mg.setup(fine)
    rng = np.random.default_rng(3)
    b = bm.tensor(rng.standard_normal(fine.shape[0]))
    x0 = bm.tensor(rng.standard_normal(fine.shape[0]))
    x, _ = mg.solve(b, x0)
    np.testing.assert_allclose(np.asarray(x), np.asarray(x0 + mg @ (b - fine @ x0)), rtol=0, atol=1e-13)


@pytest.mark.parametrize('kwargs', [dict(cycle='W'), dict(n_pre=1, n_post=2), dict(n_pre=0, n_post=0)])
def test_multigrid_rejects_unsupported_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        Multigrid(**kwargs)


def test_setup_validates_hierarchy() -> None:
    bm.set_backend('numpy')
    A = _poisson(7)
    with pytest.raises(ValueError, match='没有任何层次'):
        Multigrid(coarse_solver=DirectSolver()).setup(A)
    with pytest.raises(ValueError, match='coarse_solver'):
        Multigrid().add_level(None).setup(A)
    with pytest.raises(ValueError, match='缺少延拓算子或光滑子'):
        Multigrid(coarse_solver=DirectSolver()).add_level(_poisson(3)).add_level(None).setup(A)
    with pytest.raises(RuntimeError, match='尚未 setup'):
        Multigrid(coarse_solver=DirectSolver()).add_level(None) @ bm.ones((7, ))


def test_estimate_lambda_max_is_a_tight_lower_bound() -> None:
    bm.set_backend('numpy')
    from soptx.solvers import estimate_lambda_max
    A = _poisson(50)
    dense = A.to_scipy().toarray()
    diag = bm.tensor(np.diag(dense).copy())
    exact = float(np.max(np.linalg.eigvalsh(dense / 2.0)))      # D = 2 I
    lam = estimate_lambda_max(A, diag=diag, n_iter=60)
    assert lam <= exact * (1 + 1e-12) and lam >= 0.95 * exact
    assert estimate_lambda_max(A, n_iter=60) <= 2.0 * exact * (1 + 1e-12)


def test_jacobi_smoother_chooses_omega_from_spectrum() -> None:
    bm.set_backend('numpy')
    A = _poisson(50)
    smoother = JacobiSmoother().setup(A)
    exact = float(np.max(np.linalg.eigvalsh(A.to_scipy().toarray() / 2.0)))
    # 4 / (3 * 1.1 * lam_hat), lam_hat 不超过真值, 故 omega * lam_max <= 4 / 3.3 < 2 的条件由下界保证
    assert smoother.omega >= 4.0 / (3.0 * 1.1 * exact)
    assert smoother.omega * exact < 2.0
