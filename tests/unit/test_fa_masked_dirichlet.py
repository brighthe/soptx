"""FA 下 ``apply_bc(explicit=False)`` 掩码算子、``apply_bc(inplace=True)`` 原地消元与对称消元等价的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.operators import ConstrainedOperator
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh, QuadrangleMesh
from soptx.problems import ExponentialSineManufacturedElasticity2D, FullMBBBeamLine3d
from soptx.solvers import operator_diagonal
from soptx.topology.interpolation import MaterialInterpolationScheme


CG_OPTIONS = dict(solver='cg', precond='jacobi', maxiter=2000, rtol=1e-10, atol=0.0,
                  rtol_reference='rhs', norm_type='unpreconditioned', verify_residual=True)


def _mbb_analyzer(operator_level='fa'):
    """齐次 Dirichlet 的 6x2x2 MBB 梁, 单元密度分析器, 默认求解器为直接法."""
    bm.set_backend('numpy')
    grid, domain = (6, 2, 2), (0.0, 6.0, 0.0, 2.0, 0.0, 2.0)
    mesh = HexahedronMesh.from_box(list(domain), *grid)
    problem = FullMBBBeamLine3d(domain=domain)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D')
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': ['E']})
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=2, assembly_method='fast', operator_level=operator_level,
                               solve_method='scipy', topopt_algorithm='density_based',
                               interpolation_scheme=interpolation)


def _manufactured_analyzer():
    """非齐次 Dirichlet (u_D 为解析位移) 的二维制造解, 标准 FA 分析器."""
    bm.set_backend('numpy')
    problem = ExponentialSineManufacturedElasticity2D(lame_lambda=1.0, shear_modulus=1.0)
    mesh = QuadrangleMesh.from_box(list(problem.domain), 6, 6)
    material = IsotropicLinearElasticMaterial(lame_lambda=1.0, shear_modulus=1.0, hypothesis='plane_strain')
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=3, operator_level='fa', solve_method='scipy')


def _random_density(analyzer, seed):
    """(0.1, 1) 上均匀分布的单元密度."""
    NC = analyzer.tensor_space.mesh.number_of_cells()
    return bm.tensor(np.random.default_rng(seed).uniform(0.1, 1.0, NC))


def _assert_same_operator(operator, matrix, seed):
    """掩码算子与消元矩阵的作用与对角逐位相同."""
    v = bm.tensor(np.random.default_rng(seed).standard_normal(matrix.shape[0]))
    np.testing.assert_array_equal(np.asarray(operator @ v), np.asarray(matrix @ v))
    np.testing.assert_array_equal(np.asarray(operator.diagonal()), np.asarray(operator_diagonal(matrix)))


def test_masked_operator_matches_explicit_elimination_bitwise() -> None:
    """掩码算子与对称消元矩阵的作用、对角与右端项逐位相同, 且不复制 K."""
    analyzer = _mbb_analyzer()
    stiffness = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 2))
    original = np.array(stiffness.values)
    load = analyzer.assemble_body_force_vector()

    matrix, rhs = analyzer.apply_bc(stiffness, load)
    operator, rhs_masked = analyzer.apply_bc(stiffness, load, explicit=False)

    assert isinstance(operator, ConstrainedOperator)
    assert operator.form.matrix is stiffness
    _assert_same_operator(operator, matrix, seed=3)
    np.testing.assert_array_equal(np.asarray(rhs_masked[:]), np.asarray(rhs[:]))
    np.testing.assert_array_equal(np.asarray(stiffness.values), original)


def test_explicit_true_matches_default() -> None:
    """explicit=True 与缺省一样返回对称消元矩阵."""
    analyzer = _mbb_analyzer()
    stiffness = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 4))
    load = analyzer.assemble_body_force_vector()

    default, _ = analyzer.apply_bc(stiffness, load)
    explicit, _ = analyzer.apply_bc(stiffness, load, explicit=True)

    np.testing.assert_array_equal(np.asarray(explicit.values), np.asarray(default.values))
    assert explicit.crow is stiffness.crow and explicit.col is stiffness.col


def test_masked_operator_nonhomogeneous_dirichlet() -> None:
    """非齐次 Dirichlet 下两种形式的右端项与算子作用逐位相同."""
    analyzer = _manufactured_analyzer()
    stiffness = analyzer.assemble_stiff_matrix()
    load = analyzer.assemble_body_force_vector()

    matrix, rhs = analyzer.apply_bc(stiffness, load)
    operator, rhs_masked = analyzer.apply_bc(stiffness, load, explicit=False)

    assert analyzer._dirichlet_data()[2]
    np.testing.assert_array_equal(np.asarray(rhs_masked[:]), np.asarray(rhs[:]))
    _assert_same_operator(operator, matrix, seed=5)


def test_masked_operator_cg_matches_explicit() -> None:
    """CG + Jacobi 在两种形式上迭代步数相同, 解逐位相同."""
    analyzer = _mbb_analyzer()
    stiffness = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 6))
    load = analyzer.assemble_body_force_vector()

    solutions, infos = [], []
    for explicit in (None, False):
        system, rhs = analyzer.apply_bc(stiffness, load, explicit=explicit)
        uh = analyzer.tensor_space.function()
        _, info = analyzer.solve_system(system, rhs, uh, x0=analyzer.prescribed_solution, **CG_OPTIONS)
        solutions.append(np.asarray(uh[:]).copy())
        infos.append(info)

    assert infos[0]['converged'] and infos[1]['converged']
    assert infos[0]['niter'] == infos[1]['niter']
    np.testing.assert_array_equal(solutions[1], solutions[0])


def test_masked_operator_rejects_direct_solver() -> None:
    """掩码算子交给直接法时报出指向 explicit 的错误."""
    analyzer = _mbb_analyzer()
    stiffness = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 7))
    operator, rhs = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector(), explicit=False)

    with pytest.raises(RuntimeError, match='explicit=False'):
        analyzer.solve_system(operator, rhs, analyzer.tensor_space.function())


def test_explicit_rejects_non_bool() -> None:
    """explicit 只接受 None 或 bool."""
    analyzer = _mbb_analyzer()
    stiffness = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 8))

    with pytest.raises(RuntimeError, match='explicit'):
        analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector(), explicit=0)


def test_matrix_free_rejects_explicit_true() -> None:
    """'ea' 层级下 explicit=True 报错, False 与缺省一样返回掩码算子."""
    analyzer = _mbb_analyzer(operator_level='ea')
    K = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 9))
    load = analyzer.assemble_body_force_vector()

    operator, _ = analyzer.apply_bc(K, load, explicit=False)
    assert isinstance(operator, ConstrainedOperator)
    with pytest.raises(RuntimeError, match='explicit=True'):
        analyzer.apply_bc(K, load, explicit=True)


def test_inplace_matches_default_and_overwrites_stiffness() -> None:
    """inplace=True 与缺省复制消元的矩阵与右端项逐位相同, 且改写 K 本身."""
    analyzer = _mbb_analyzer()
    density = _random_density(analyzer, 9)
    stiffness = analyzer.assemble_stiff_matrix(rho_val=density)
    load = analyzer.assemble_body_force_vector()
    default, rhs = analyzer.apply_bc(stiffness, load)
    reference = np.array(default.values)

    stiffness = analyzer.assemble_stiff_matrix(rho_val=density)
    matrix, rhs_inplace = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector(), inplace=True)

    assert matrix.values is stiffness.values
    np.testing.assert_array_equal(np.asarray(matrix.values), reference)
    np.testing.assert_array_equal(np.asarray(rhs_inplace[:]), np.asarray(rhs[:]))
    assert analyzer.stiffness_matrix is stiffness


def test_inplace_nonhomogeneous_dirichlet_uses_unconstrained_load_correction() -> None:
    """非齐次 Dirichlet 下 K u_D 在原地消元之前算好, 右端项与复制消元逐位相同."""
    analyzer = _manufactured_analyzer()
    _, rhs = analyzer.apply_bc(analyzer.assemble_stiff_matrix(), analyzer.assemble_body_force_vector())

    _, rhs_inplace = analyzer.apply_bc(analyzer.assemble_stiff_matrix(), analyzer.assemble_body_force_vector(),
                                       inplace=True)

    assert analyzer._dirichlet_data()[2]
    np.testing.assert_array_equal(np.asarray(rhs_inplace[:]), np.asarray(rhs[:]))


def test_inplace_rejects_masked_operator_and_matrix_free_level() -> None:
    """inplace=True 不能与 explicit=False 同时给出, 无矩阵层级不支持 inplace."""
    analyzer = _mbb_analyzer()
    stiffness = analyzer.assemble_stiff_matrix(rho_val=_random_density(analyzer, 10))
    with pytest.raises(RuntimeError, match='inplace'):
        analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector(), explicit=False, inplace=True)

    ea = _mbb_analyzer(operator_level='ea')
    operator = ea.assemble_stiff_matrix(rho_val=_random_density(ea, 10))
    with pytest.raises(RuntimeError, match='inplace'):
        ea.apply_bc(operator, ea.assemble_body_force_vector(), inplace=True)
