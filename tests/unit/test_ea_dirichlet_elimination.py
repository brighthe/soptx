"""无矩阵层级 ('ea') 施加 Dirichlet 条件改用缓存数据后的回归测试.

参考值按改动前的做法现算: 由 ``is_boundary_dof`` 取掩码, ``init_solution`` 插值 u_D,
``ConstrainedOperator.apply`` 消去边界贡献.
"""

from __future__ import annotations

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.operators import ConstrainedOperator
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh, QuadrangleMesh
from soptx.problems import ExponentialSineManufacturedElasticity2D, FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme


def _mbb_analyzer(operator_level='ea'):
    """齐次 Dirichlet 的 6x2x2 MBB 梁, 单元密度分析器."""
    bm.set_backend('numpy')
    grid, domain = (6, 2, 2), (0.0, 6.0, 0.0, 2.0, 0.0, 2.0)
    mesh = HexahedronMesh.from_box(list(domain), *grid)
    problem = FullMBBBeam3d(domain=domain, support='end_lines', load_subdivisions=(grid[0], grid[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D',
                                              enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': ['E']},
        enable_logging=False)
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=2, assembly_method='fast', operator_level=operator_level,
                               solve_method='cg', topopt_algorithm='density_based',
                               interpolation_scheme=interpolation, enable_logging=False)


def _manufactured_analyzer(operator_level):
    """非齐次 Dirichlet (u_D 为解析位移) 的二维制造解."""
    bm.set_backend('numpy')
    problem = ExponentialSineManufacturedElasticity2D(lame_lambda=1.0, shear_modulus=1.0)
    mesh = QuadrangleMesh.from_box(list(problem.domain), 6, 6)
    material = IsotropicLinearElasticMaterial(lame_lambda=1.0, shear_modulus=1.0, hypothesis='plane_strain',
                                              enable_logging=False)
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=3, operator_level=operator_level, solve_method='cg',
                               enable_logging=False)


def _reference(analyzer, K, F):
    """改动前的做法: 现算掩码与 u_D, 完整执行 F - K u_D."""
    space = analyzer.tensor_space
    isBdDof = space.is_boundary_dof(threshold=analyzer.pde.is_dirichlet_boundary(), method='interp')
    operator = ConstrainedOperator(analyzer.wrap_operator(K), gd=analyzer.pde.dirichlet_bc, isDDof=isBdDof)
    uh_bd = operator.init_solution(dtype=bm.float64)
    uh_bd = bm.set_at(uh_bd, ~isBdDof, 0.0)
    non_body = analyzer._non_body_loads_by_boundary_type(adjoint=False)
    F_full = F if non_body is None else F + non_body
    return operator, np.asarray(operator.apply(F_full, uh_bd)), np.asarray(uh_bd), np.asarray(isBdDof)


def test_homogeneous_dirichlet_matches_previous_construction_bitwise() -> None:
    analyzer = _mbb_analyzer()
    density = bm.tensor(np.random.default_rng(0).uniform(0.1, 1.0, analyzer.tensor_space.mesh.number_of_cells()))
    K = analyzer.assemble_stiff_matrix(rho_val=density)
    F = analyzer.assemble_body_force_vector()

    operator, rhs = analyzer.apply_bc(K, F)
    ref_operator, ref_rhs, ref_uh, ref_mask = _reference(analyzer, K, F)

    assert not analyzer._dirichlet_data()[2]
    np.testing.assert_array_equal(np.asarray(rhs), ref_rhs)
    np.testing.assert_array_equal(np.asarray(analyzer.prescribed_solution), ref_uh)
    np.testing.assert_array_equal(np.asarray(operator.is_boundary_dof), ref_mask)
    v = bm.tensor(np.random.default_rng(1).standard_normal(rhs.shape[0]))
    np.testing.assert_array_equal(np.asarray(operator @ v), np.asarray(ref_operator @ v))


def test_dirichlet_data_is_reused_and_prescribed_solution_is_a_copy() -> None:
    analyzer = _mbb_analyzer()
    density = bm.full((analyzer.tensor_space.mesh.number_of_cells(), ), 0.5, dtype=bm.float64)
    K = analyzer.assemble_stiff_matrix(rho_val=density)
    analyzer.apply_bc(K, analyzer.assemble_body_force_vector())
    cache = analyzer._dirichlet_cache
    analyzer.apply_bc(K, analyzer.assemble_body_force_vector())
    assert analyzer._dirichlet_cache is cache

    analyzer.prescribed_solution[:] = 123.0
    assert not np.any(np.asarray(analyzer._dirichlet_data()[0][:]) == 123.0)


def test_nonhomogeneous_dirichlet_still_corrects_load_and_matches_fa() -> None:
    ea = _manufactured_analyzer('ea')
    K = ea.assemble_stiff_matrix()
    F = ea.assemble_body_force_vector()
    operator, rhs = ea.apply_bc(K, F)
    _, ref_rhs, ref_uh, _ = _reference(ea, K, F)

    assert ea._dirichlet_data()[2]
    np.testing.assert_array_equal(np.asarray(rhs), ref_rhs)
    np.testing.assert_array_equal(np.asarray(ea.prescribed_solution), ref_uh)

    fa = _manufactured_analyzer('fa')
    _, fa_rhs = fa.apply_bc(fa.assemble_stiff_matrix(), fa.assemble_body_force_vector())
    np.testing.assert_allclose(np.asarray(rhs), np.asarray(fa_rhs[:]), rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(np.asarray(fa_rhs[:]))))
