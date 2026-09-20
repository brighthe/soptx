"""一般线性约束子结构求解的解析解与约束一致性测试."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from fealpy.backend import backend_manager as bm
from soptx.fem.substructure import solve_constrained_system


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
