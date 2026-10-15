"""右端项相对容差在热启动、批量和极小载荷下的回归用例."""

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.solvers import cg


@pytest.fixture(autouse=True)
def numpy_backend():
    """用例结束后恢复调用方后端."""
    previous = bm.backend_name
    bm.set_backend('numpy')
    yield
    bm.set_backend(previous)


def test_rhs_reference_accepts_accurate_warm_start():
    """右端项参照的目标不因热启动而收紧, 默认初始残差语义保留."""
    matrix = np.diag([1.0, 4.0])
    rhs = np.ones(2)
    guess = np.linalg.solve(matrix, rhs) * (1.0 - 1e-5)
    options = dict(atol=0.0, rtol=1e-4, norm_type='unpreconditioned',
                   verify_residual=True, returninfo=True)
    _, warm = cg(matrix, rhs, guess.copy(), rtol_reference='rhs', **options)
    _, cold = cg(matrix, rhs, rtol_reference='rhs', **options)
    _, legacy = cg(matrix, rhs, guess.copy(), **options)
    assert warm['niter'] == 0
    assert legacy['niter'] > 0
    assert warm['stopping_threshold'] == cold['stopping_threshold']
    assert warm['true_residual'] <= warm['stopping_threshold']


@pytest.mark.parametrize('batch_first', [False, True])
def test_rhs_reference_checks_each_column(batch_first):
    """不同量级的列独立达标, 不用大载荷列的阈值放过小载荷列."""
    matrix = np.diag([1.0, 4.0])
    columns = np.array([[1.0, 1e-20], [2.0, 3e-20]])
    rhs = columns.T.copy() if batch_first else columns
    solution, info = cg(matrix, rhs, rtol_reference='rhs', atol=0.0, rtol=1e-10,
                        batch_first=batch_first, verify_residual=True, returninfo=True)
    x = solution.T if batch_first else solution
    residuals = np.linalg.norm(columns - matrix @ x, axis=0)
    assert info['converged']
    assert np.all(residuals <= 1e-10 * np.linalg.norm(columns, axis=0))


def test_rhs_reference_does_not_discard_tiny_load():
    """极小但非零的右端项仍须求解."""
    rhs = np.array([1e-20, 2e-20])
    x, info = cg(np.eye(2), rhs, atol=0.0, rtol=1e-8,
                 rtol_reference='rhs', verify_residual=True, returninfo=True)
    assert info['niter'] == 1
    np.testing.assert_allclose(x, rhs, rtol=1e-8, atol=0.0)


def test_rhs_reference_absolute_tolerance_floor():
    """绝对容差可以主导停机阈值."""
    rhs = np.ones(2)
    _, info = cg(np.eye(2), rhs, x0=rhs * (1 - 1e-5),
                 atol=1e-4, rtol=1e-8, rtol_reference='rhs', returninfo=True)
    assert info['niter'] == 0
    assert info['stopping_threshold'] == pytest.approx(1e-4)


@pytest.mark.parametrize('norm_type', ['natural', 'unpreconditioned', 'preconditioned'])
def test_rhs_reference_uses_selected_norm(norm_type):
    """参照量与残差使用同一范数, 包括有预条件子的情况."""
    matrix = np.diag([1.0, 4.0])
    inverse = np.diag([1.0, 0.25])
    rhs = np.ones(2)
    z = inverse @ rhs
    expected = {'natural': np.sqrt(rhs @ z),
                'unpreconditioned': np.linalg.norm(rhs),
                'preconditioned': np.linalg.norm(z)}[norm_type]
    _, info = cg(matrix, rhs, M=inverse, norm_type=norm_type,
                 rtol_reference='rhs', atol=0.0, rtol=1e-8, returninfo=True)
    assert info['reference_norm'] == pytest.approx(expected)


def test_fem_and_interface_use_constrained_rhs():
    """固定自由度上的大载荷不能放宽自由系统的容差."""
    from scipy.sparse import csr_matrix
    from types import SimpleNamespace

    from soptx.fem.linear_solve import build_solver, run_solver
    from soptx.fem.substructure.solve import solve_constrained_system

    options = dict(atol=0.0, rtol=1e-4, rtol_reference='rhs',
                   norm_type='unpreconditioned', verify_residual=True)
    matrix = np.diag([1.0, 2.0, 4.0])
    force = np.array([1e10, 1.0, 2.0])
    rhs = force.copy()
    rhs[0] = 0.0
    guess = np.linalg.solve(matrix, rhs) * (1 - 1e-5)
    system = SimpleNamespace(stiffness=csr_matrix(matrix), global_dofs=np.arange(3))
    result = solve_constrained_system(
        system, force, csr_matrix([[1.0, 0.0, 0.0]]), solver='cg',
        precond='none', x0=guess.copy(), **options)
    solver, extra, tol = build_solver('cg', options)
    info = run_solver(solver, matrix, rhs, np.zeros(3), name='cg',
                      extra=extra, tol=tol, x0=guess.copy())
    assert result.iterations == info['niter'] == 0
    assert result.solver_info['rhs_norm'] == pytest.approx(np.linalg.norm(rhs))
    assert result.solver_info['stopping_threshold'] == info['stopping_threshold']


def test_interface_absolute_tolerance_accepts_zero_guess():
    """接口验收沿用绝对阈值, 不额外要求相对残差小于 rtol."""
    from scipy.sparse import csr_matrix
    from types import SimpleNamespace

    from soptx.fem.substructure.solve import solve_constrained_system

    system = SimpleNamespace(stiffness=csr_matrix(np.eye(2)), global_dofs=np.arange(2))
    result = solve_constrained_system(
        system, np.array([0.0, 1e-5]), csr_matrix([[1.0, 0.0]]),
        solver='cg', precond='none', atol=1e-4, rtol=0.0)
    assert result.iterations == 0
    assert result.equilibrium_relative_residual == pytest.approx(1.0)
    assert result.solver_info['true_residual'] <= result.solver_info['stopping_threshold']
