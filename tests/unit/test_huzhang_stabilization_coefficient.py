"""稳定化系数默认值及密度依赖的装配回归测试."""
import numpy as np
import pytest
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from soptx.fem import HuZhangMFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import MixedBoundarySinusoidalElasticity2D


class _Interpolation:
    """用于装配检查的逐单元 SIMP 插值."""
    def interpolate_material(self, *, material, rho_val, **kwargs):
        return material.youngs_modulus * (1e-6 + (1 - 1e-6) * rho_val**3)


def _analyzer(order=2, **kwargs):
    """在小网格上创建真实分析器, 省略系数参数时走公共默认值."""
    bm.set_backend('numpy')
    problem = MixedBoundarySinusoidalElasticity2D()
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam, shear_modulus=problem.mu,
        hypothesis=problem.plane_type, enable_logging=False)
    return HuZhangMFEMAnalyzer(
        disp_mesh=mesh, pde=problem, material=material,
        interpolation_scheme=_Interpolation(), topopt_algorithm='density_based',
        space_degree=order, integration_order=2*order+2,
        solve_method='scipy', use_relaxation=False, **kwargs)


def _blocks(analyzer, rho):
    """从实际鞍点矩阵提取柔度与位移惩罚块."""
    matrix = analyzer.assemble_stiff_matrix(rho_val=np.full(8, rho)).to_scipy()
    n = analyzer.tensor_space.number_of_global_dofs()
    return matrix[:-n, :-n].toarray(), matrix[-n:, -n:].toarray()


def test_default_penalty_is_fixed_while_compliance_changes():
    """改变密度只改变柔度块, 默认惩罚块保持非零且与显式固定模式相同."""
    analyzer = _analyzer()
    assert analyzer.stabilization_coefficient == 'fixed'
    a1, j1 = _blocks(analyzer, 1.)
    a2, j2 = _blocks(analyzer, .5)
    assert not np.allclose(a1, a2)
    assert np.linalg.norm(j1) > 0
    np.testing.assert_allclose(j1, j2, rtol=0, atol=0)
    _, explicit = _blocks(_analyzer(stabilization_coefficient='fixed'), .5)
    np.testing.assert_allclose(j2, explicit, rtol=0, atol=0)


def test_density_dependent_penalty_requires_explicit_choice():
    """显式密度模式在均匀密度下按相对剪切模量缩放, 实体时与固定模式一致."""
    analyzer = _analyzer(stabilization_coefficient='density_dependent')
    _, solid = _blocks(analyzer, 1.)
    _, porous = _blocks(analyzer, .5)
    _, fixed = _blocks(_analyzer(), 1.)
    np.testing.assert_allclose(solid, fixed, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(porous, solid*(1e-6+(1-1e-6)*.5**3), rtol=1e-13, atol=1e-14)


def test_native_high_order_is_unchanged_by_coefficient_mode():
    """原生高阶不装配惩罚块, 两种模式给出相同鞍点矩阵."""
    fixed = _blocks(_analyzer(order=3), .5)
    density = _blocks(_analyzer(order=3, stabilization_coefficient='density_dependent'), .5)
    for a, b in zip(fixed, density):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert not np.any(fixed[1])


def test_invalid_coefficient_mode_is_rejected():
    """构造时拒绝未知模式, 避免拼写错误静默落到其他计算路径."""
    with pytest.raises(ValueError, match='stabilization_coefficient'):
        _analyzer(stabilization_coefficient='typo')