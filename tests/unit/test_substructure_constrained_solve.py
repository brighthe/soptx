"""一般线性约束子结构求解的解析解与约束一致性测试."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from soptx.backend import backend_manager as bm
from soptx.fem.substructure import solve_constrained_system
from soptx.fem.substructure.solve import (
    PreparedDirectSystem,
    prepare_constrained_system,
    solve_prepared_constrained_system,
    prepare_fixed_interface_system,
    solve_prepared_fixed_system,
)


def make_system():
    """构造两个自由度的正定系统, 供解析解对照."""
    bm.set_backend("numpy")
    return SimpleNamespace(stiffness=csr_matrix(np.diag([2.0, 4.0])),
                           global_dofs=np.arange(2))


def test_coupled_constraint_preserves_virtual_work():
    """q0=q1 时应得到共同位移 1/6, 不能把两个自由度固定为零."""
    result = solve_constrained_system(
        make_system(), np.array([1.0, 0.0]), csr_matrix([[1.0, -1.0]])
    )
    np.testing.assert_allclose(bm.to_numpy(result.displacement), [1/6, 1/6], atol=1e-12)
    assert result.constraint_rank == 1
    assert result.equilibrium_relative_residual < 1e-12
    assert result.constraint_relative_residual < 1e-12


def test_nonzero_redundant_constraints_have_same_solution():
    """一致的冗余行和零行不应改变 q0+q1=1 的约束最小能量解."""
    result = solve_constrained_system(
        make_system(), np.zeros(2),
        csr_matrix([[1., 1.], [2., 2.], [0., 0.]]),
        prescribed=np.array([1., 2., 0.]),
    )
    np.testing.assert_allclose(bm.to_numpy(result.displacement), [2/3, 1/3], atol=1e-12)
    assert result.constraint_rank == 1


@pytest.mark.parametrize("constraints, prescribed", [
    ([[1., 1.], [2., 2.]], [1., 3.]),
    ([[0., 0.]], [1.]),
])
def test_inconsistent_constraints_are_rejected(constraints, prescribed):
    """矛盾约束必须报错, 不能静默删除后输出一个解."""
    with pytest.raises(ValueError):
        solve_constrained_system(make_system(), np.zeros(2), csr_matrix(constraints),
                                 prescribed=np.asarray(prescribed))


def test_prescribed_coordinate_displacement_with_zero_load():
    """非齐次边界驱动的无外载系统应有有限且接近零的平衡残差."""
    system = make_system()
    system.stiffness = csr_matrix([[2., -1.], [-1., 2.]])
    result = solve_constrained_system(
        system, np.zeros(2), csr_matrix([[1., 0.]]), prescribed=np.array([1.])
    )
    np.testing.assert_allclose(bm.to_numpy(result.displacement), [1., 0.5], atol=1e-12)
    assert result.equilibrium_relative_residual < 1e-12


def test_prepared_fixed_system_preserves_matrix_and_solves_equilibrium():
    """掩码与右端在准备时生效, 原矩阵保留耦合项且解满足自由平衡."""
    system = make_system()
    system.stiffness = csr_matrix([[2., -1.], [-1., 2.]])
    original = system.stiffness.toarray().copy()
    force = np.array([3., 1.])
    prepared = prepare_fixed_interface_system(system, force, csr_matrix([[1., 0.]]))
    np.testing.assert_array_equal(prepared.rhs, [0., 1.])
    np.testing.assert_allclose(prepared.operator @ np.array([7., 2.]), [7., 4.])
    np.testing.assert_array_equal(system.stiffness.toarray(), original)
    np.testing.assert_array_equal(force, [3., 1.])
    result = solve_prepared_fixed_system(
        prepared, atol=0., rtol=1e-10, rtol_reference='rhs',
        norm_type='unpreconditioned', verify_residual=True,
        cg_maxiter=10, precond='jacobi', x0=np.array([9., 0.]),
    )
    np.testing.assert_allclose(result.displacement, [0., 0.5], atol=1e-12)
    assert result.equilibrium_relative_residual < 1e-10


@pytest.mark.parametrize('constraints, values, load, expected, shape', [
    (np.empty((0, 2)), [], [2., 4.], [1., 1.], (2, 2)),
    ([[1., 0.]], [1.], [0., 4.], [1., 1.], (1, 1)),
    ([[1., 0.], [0., 1.]], [1., 2.], [0., 0.], [1., 2.], (0, 0)),
    ([[1., -1.]], [0.], [1., 0.], [1/6, 1/6], (3, 3)),
    ([[1., 1.], [2., 2.], [0., 0.]], [1., 2., 0.], [0., 0.], [2/3, 1/3], (3, 3)),
])
def test_direct_preparation_is_complete_before_solve(
        monkeypatch, constraints, values, load, expected, shape):
    """无约束, 固定约束及一般约束均在准备阶段完成系统构造."""
    import soptx.fem.substructure.solve as interface_solve

    system = make_system()
    original = system.stiffness.toarray().copy()
    prepared = prepare_constrained_system(
        system, np.asarray(load), csr_matrix(constraints), prescribed=np.asarray(values),
    )
    assert isinstance(prepared, PreparedDirectSystem)
    assert prepared.operator.shape == shape
    assert prepared.rhs.shape == (shape[0],)
    np.testing.assert_array_equal(system.stiffness.toarray(), original)

    def unexpected_preparation(*args, **kwargs):
        raise AssertionError('求解阶段重复准备约束')

    for name in ('qr', 'bmat', '_interface_constraint_data', 'solve_interface_system'):
        monkeypatch.setattr(interface_solve, name, unexpected_preparation)
    result = solve_prepared_constrained_system(prepared, load=prepared.rhs.copy())
    np.testing.assert_allclose(result.displacement, expected, atol=1e-12)
    assert result.constraint_relative_residual < 1e-12
    assert result.equilibrium_relative_residual < 1e-12


def test_inconsistent_constraints_fail_during_preparation():
    """相关约束的给定值矛盾必须在准备阶段报错."""
    with pytest.raises(ValueError, match='不一致'):
        prepare_constrained_system(
            make_system(), np.zeros(2), csr_matrix([[1., 1.], [2., 2.]]),
            prescribed=np.array([1., 3.]),
        )


def test_direct_nonzero_boundary_rhs_is_prepared():
    """非齐次给定位移引起的右端修正在准备阶段完成."""
    system = make_system()
    system.stiffness = csr_matrix([[2., -1.], [-1., 2.]])
    prepared = prepare_constrained_system(
        system, np.zeros(2), csr_matrix([[1., 0.]]), prescribed=np.array([1.]),
    )
    np.testing.assert_array_equal(prepared.rhs, [1.])
    result = solve_prepared_constrained_system(prepared)
    np.testing.assert_allclose(result.displacement, [1., 0.5], atol=1e-12)
