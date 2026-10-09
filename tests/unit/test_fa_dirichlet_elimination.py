"""FA 保结构对称消元与 Dirichlet 数据缓存的回归测试."""

from __future__ import annotations

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh, QuadrangleMesh
from soptx.problems import ExponentialSineManufacturedElasticity2D, FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme


def _mbb_analyzer():
    """齐次 Dirichlet 的 6x2x2 MBB 梁, 单元密度 FA 分析器."""
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
                               integration_order=2, assembly_method='fast', operator_level='fa',
                               solve_method='scipy', topopt_algorithm='density_based',
                               interpolation_scheme=interpolation, enable_logging=False)


def _manufactured_analyzer():
    """非齐次 Dirichlet (u_D 为解析位移) 的二维制造解, 标准 FA 分析器."""
    bm.set_backend('numpy')
    problem = ExponentialSineManufacturedElasticity2D(lame_lambda=1.0, shear_modulus=1.0)
    mesh = QuadrangleMesh.from_box(list(problem.domain), 6, 6)
    material = IsotropicLinearElasticMaterial(lame_lambda=1.0, shear_modulus=1.0, hypothesis='plane_strain',
                                              enable_logging=False)
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=3, operator_level='fa', solve_method='scipy',
                               enable_logging=False)


def _reference_system(analyzer, stiffness, load):
    """删除行列式对称消元的稠密参考: 右端减去 K u_D, 约束行列置零、对角置 1."""
    uh_bd, is_bd = analyzer.tensor_space.boundary_interpolate(
        gd=analyzer.pde.dirichlet_bc, threshold=analyzer.pde.is_dirichlet_boundary(), method='interp')
    uh_bd, is_bd = np.asarray(uh_bd[:]), np.asarray(is_bd)
    matrix = np.asarray(stiffness.to_scipy().toarray())
    rhs = np.asarray(load[:]) - matrix @ uh_bd
    rhs[is_bd] = uh_bd[is_bd]
    matrix[is_bd, :] = 0.0
    matrix[:, is_bd] = 0.0
    matrix[is_bd, is_bd] = 1.0
    return matrix, rhs


def test_structure_preserving_elimination_matches_row_column_deletion() -> None:
    analyzer = _mbb_analyzer()
    density = bm.tensor(np.random.default_rng(0).uniform(0.1, 1.0, analyzer.tensor_space.mesh.number_of_cells()))
    stiffness = analyzer.assemble_stiff_matrix(rho_val=density)
    original = np.array(stiffness.values)
    load = analyzer.assemble_body_force_vector()

    matrix, rhs = analyzer.apply_bc(stiffness, load)
    expected_matrix, expected_rhs = _reference_system(analyzer, stiffness, load + analyzer._non_body_loads_by_boundary_type(False))

    np.testing.assert_array_equal(matrix.to_scipy().toarray(), expected_matrix)
    np.testing.assert_allclose(np.asarray(rhs[:]), expected_rhs, rtol=0.0, atol=1e-15)
    # 原刚度矩阵不被改写, 且消元后共用原骨架
    np.testing.assert_array_equal(np.asarray(stiffness.values), original)
    assert matrix.crow is stiffness.crow and matrix.col is stiffness.col


def test_elimination_slots_are_cached_across_density_updates() -> None:
    analyzer = _mbb_analyzer()
    rng = np.random.default_rng(1)
    NC = analyzer.tensor_space.mesh.number_of_cells()

    analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=bm.tensor(rng.uniform(0.1, 1.0, NC))),
                      analyzer.assemble_body_force_vector())
    cache = analyzer._elimination._cache
    stiffness = analyzer.assemble_stiff_matrix(rho_val=bm.tensor(rng.uniform(0.1, 1.0, NC)))
    load = analyzer.assemble_body_force_vector()
    matrix, _ = analyzer.apply_bc(stiffness, load)

    assert analyzer._elimination._cache is cache
    expected, _ = _reference_system(analyzer, stiffness, load)
    np.testing.assert_array_equal(matrix.to_scipy().toarray(), expected)


def test_nonhomogeneous_dirichlet_still_corrects_load() -> None:
    analyzer = _manufactured_analyzer()
    stiffness = analyzer.assemble_stiff_matrix()
    load = analyzer.assemble_body_force_vector()

    matrix, rhs = analyzer.apply_bc(stiffness, load)
    expected_matrix, expected_rhs = _reference_system(analyzer, stiffness, load)

    assert analyzer._dirichlet_data()[2]
    np.testing.assert_array_equal(matrix.to_scipy().toarray(), expected_matrix)
    np.testing.assert_allclose(np.asarray(rhs[:]), expected_rhs, rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(expected_rhs)))


def test_prescribed_solution_is_a_copy_of_the_cache() -> None:
    analyzer = _mbb_analyzer()
    density = bm.full((analyzer.tensor_space.mesh.number_of_cells(), ), 0.5, dtype=bm.float64)
    analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density), analyzer.assemble_body_force_vector())

    prescribed = analyzer._prescribed_solution
    prescribed[:] = 123.0
    assert not np.any(np.asarray(analyzer._dirichlet_data()[0][:]) == 123.0)
