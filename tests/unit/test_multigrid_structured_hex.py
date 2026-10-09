"""``soptx.fem.multigrid`` 结构化六面体多重网格层次的测试.

粗层算子与独立构造的代数 Galerkin 投影 P0^T A P0 逐元比较: P 由坐标上的三线性基函数
直接求值得到, 不经被测代码的编号算术.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.multigrid import StructuredHexGrid, StructuredHexHierarchy
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeam3d
from soptx.solvers import DiagonalPreconditioner, create
from soptx.sparse import CSRTensor
from soptx.topology.interpolation import MaterialInterpolationScheme


def _mbb_system(grid, seed=0):
    """单元密度下 MBB 梁的 FA 系统: 返回分析器, 消元后的 K (scipy) 与单元系数."""
    bm.set_backend('numpy')
    domain = (0.0, float(grid[0]), 0.0, float(grid[1]), 0.0, float(grid[2]))
    mesh = HexahedronMesh.from_box(list(domain), *grid)
    problem = FullMBBBeam3d(domain=domain, support='end_lines', load_subdivisions=(grid[0], grid[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D',
                                              enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': ['E']},
        enable_logging=False)
    analyzer = LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                                   integration_order=2, assembly_method='fast', operator_level='fa',
                                   solve_method='scipy', topopt_algorithm='density_based',
                                   interpolation_scheme=interpolation, enable_logging=False)
    density = bm.tensor(np.random.default_rng(seed).uniform(0.0, 1.0, mesh.number_of_cells()))
    K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                             analyzer.assemble_body_force_vector())
    return analyzer, K, F, np.asarray(analyzer._integrator.coef)


def _trilinear(fine: StructuredHexGrid, coarse: StructuredHexGrid) -> np.ndarray:
    """独立参考: 粗网格三线性基函数在细节点上的取值, (NN_fine, NN_coarse) 稠密阵."""
    x = fine.node_coordinates()
    X = coarse.node_coordinates()
    h = np.asarray(coarse.spacing)
    return np.prod(np.clip(1.0 - np.abs(x[:, None, :] - X[None, :, :]) / h, 0.0, None), axis=2)


def _vector(P: np.ndarray, dof_priority: bool) -> np.ndarray:
    """标量插值扩展到 3 分量, 布局同 TensorFunctionSpace."""
    return np.kron(np.eye(3), P) if dof_priority else np.kron(P, np.eye(3))


def _reference_operators(K: np.ndarray, fixed: np.ndarray, grids, dof_priority: bool):
    """逐层 P0^T A P0, 零对角自由度置 1 并在下一层屏蔽其行."""
    operators, A, mask = [], K, ~fixed
    for fine, coarse in zip(grids[:-1], grids[1:]):
        P0 = _vector(_trilinear(fine, coarse), dof_priority) * mask[:, None]
        A = P0.T @ A @ P0
        dead = np.diag(A) <= 1e-14 * np.max(np.diag(A))
        A = A + np.diag(dead.astype(float))
        operators.append(A)
        mask = ~dead
    return operators


def _permutation_to_dof_priority(num_nodes: int) -> np.ndarray:
    """分量优先编号 3 n + c 到自由度优先编号 c N + n 的置换: perm[新] = 旧."""
    node, comp = np.meshgrid(np.arange(num_nodes), np.arange(3), indexing='xy')
    return (3 * node + comp).ravel()


def test_grid_from_box_mesh_and_coarsen() -> None:
    bm.set_backend('numpy')
    mesh = HexahedronMesh.from_box([0.0, 7.0, 0.0, 3.0, 0.0, 5.0], 7, 3, 5)
    grid = StructuredHexGrid.from_mesh(mesh)
    assert grid.shape == (7, 3, 5) and np.allclose(grid.spacing, 1.0) and np.allclose(grid.origin, 0.0)
    coarse = grid.coarsen()
    assert coarse.shape == (4, 2, 3) and np.allclose(coarse.spacing, 2.0)
    assert coarse.coarsen().coarsen().shape == (1, 1, 1)


def test_grid_rejects_unstructured_numbering() -> None:
    bm.set_backend('numpy')
    mesh = HexahedronMesh.from_box([0.0, 2.0, 0.0, 1.0, 0.0, 1.0], 2, 1, 1)
    node, cell = np.asarray(mesh.entity('node')), np.asarray(mesh.entity('cell'))
    with pytest.raises(ValueError, match='单元连接'):
        StructuredHexGrid.from_mesh(HexahedronMesh(bm.tensor(node), bm.tensor(cell[::-1].copy())))
    stretched = node.copy()
    stretched[np.isclose(node[:, 0], 2.0), 0] = 3.0
    with pytest.raises(ValueError, match='节点坐标'):
        StructuredHexGrid.from_mesh(HexahedronMesh(bm.tensor(stretched), bm.tensor(cell)))


@pytest.mark.parametrize('grid', [(6, 2, 2), (7, 3, 5)])
@pytest.mark.parametrize('dof_priority', [False, True])
def test_coarse_operators_match_algebraic_galerkin(grid, dof_priority) -> None:
    analyzer, K, _, coef = _mbb_system(grid)
    space = analyzer.tensor_space
    assert space.dof_priority is False
    K_dense = K.to_scipy().toarray()
    fixed = np.asarray(analyzer._dirichlet_data()[1])
    K0 = np.asarray(analyzer._reference_stiffness_matrices()[0])

    if dof_priority:
        # 被测空间换成自由度优先编号: 全局与单元局部自由度同时置换
        perm = _permutation_to_dof_priority(K_dense.shape[0] // 3)
        local = _permutation_to_dof_priority(8)
        K_dense, fixed, K0 = K_dense[np.ix_(perm, perm)], fixed[perm], K0[np.ix_(local, local)]
        space = SimpleNamespace(dof_numel=3, p=1, dof_priority=True, mesh=space.mesh)

    hierarchy = StructuredHexHierarchy(space, bm.tensor(fixed), bm.tensor(K0), coarse_max_dofs=30)
    assert hierarchy.num_levels >= 3
    hierarchy.update(bm.tensor(coef))
    expected = _reference_operators(K_dense, fixed, hierarchy.grids, dof_priority)
    assert len(hierarchy.operators) == len(expected)
    for actual, reference in zip(hierarchy.operators, expected):
        np.testing.assert_allclose(actual.to_scipy().toarray(), reference, rtol=0.0,
                                   atol=1e-13 * np.max(np.abs(reference)))


def test_mgcg_matches_direct_solution() -> None:
    analyzer, K, F, coef = _mbb_system((24, 4, 4), seed=1)
    hierarchy = StructuredHexHierarchy(analyzer.tensor_space, analyzer._dirichlet_data()[1],
                                       analyzer._reference_stiffness_matrices()[0], coarse_max_dofs=200)
    hierarchy.update(bm.tensor(coef))
    mg = hierarchy.build_multigrid().setup(K)
    solver = create('cg', M=mg, atol=0.0, rtol=1e-10, maxit=500, norm_type='unpreconditioned')
    x, info = solver.setup(K.tocoo()).solve(F)
    reference = spsolve(K.to_scipy().tocsc(), np.asarray(F[:]))
    # 完全随机的单元系数 (跳变达 1e7) 是最难的情形; 同一系统 Jacobi-PCG 需 500 余步
    jacobi = create('cg', M=DiagonalPreconditioner(), atol=0.0, rtol=1e-10, maxit=5000,
                    norm_type='unpreconditioned')
    _, jacobi_info = jacobi.setup(K.tocoo()).solve(F)
    assert info['converged'] and info['niter'] * 5 < jacobi_info['niter']
    np.testing.assert_allclose(np.asarray(x), reference, rtol=0.0, atol=1e-8 * np.max(np.abs(reference)))


def test_update_rejects_wrong_coefficient_shape() -> None:
    analyzer, _, _, coef = _mbb_system((6, 2, 2))
    hierarchy = StructuredHexHierarchy(analyzer.tensor_space, analyzer._dirichlet_data()[1],
                                       analyzer._reference_stiffness_matrices()[0], coarse_max_dofs=30)
    with pytest.raises(ValueError, match='coef'):
        hierarchy.update(bm.tensor(coef[:-1]))
    with pytest.raises(RuntimeError, match='尚未 update'):
        hierarchy.operators


def _analyzer_with_options(grid, operator_level='fa', **solver_options):
    bm.set_backend('numpy')
    domain = (0.0, float(grid[0]), 0.0, float(grid[1]), 0.0, float(grid[2]))
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
                               solve_method='cg', solver_options=solver_options,
                               topopt_algorithm='density_based', interpolation_scheme=interpolation,
                               enable_logging=False)


def test_analyzer_mg_preconditioner_reuses_hierarchy_across_densities() -> None:
    analyzer = _analyzer_with_options((24, 4, 4), precond='mg', rtol=0.0, atol=1e-12, maxiter=500,
                                      mg_coarse_max_dofs=200)
    rng = np.random.default_rng(4)
    NC = analyzer.tensor_space.mesh.number_of_cells()
    hierarchies = []
    for _ in range(2):
        density = bm.tensor(rng.uniform(0.0, 1.0, NC))
        K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                                 analyzer.assemble_body_force_vector())
        u = analyzer.tensor_space.function()
        _, info = analyzer.solve_system(K, F, u)
        reference = spsolve(K.to_scipy().tocsc(), np.asarray(F[:]))
        assert info['converged'] and info['precond'] == 'mg' and info['niter'] < 150
        np.testing.assert_allclose(np.asarray(u[:]), reference, rtol=0.0,
                                   atol=1e-8 * np.max(np.abs(reference)))
        hierarchies.append(analyzer._mg_hierarchy)
    assert hierarchies[0] is hierarchies[1]


def test_analyzer_mg_preconditioner_on_ea_matches_fa() -> None:
    grid = (24, 4, 4)
    density = bm.tensor(np.random.default_rng(5).uniform(0.0, 1.0, grid[0] * grid[1] * grid[2]))
    solutions, iterations = {}, {}
    for level in ('fa', 'ea'):
        analyzer = _analyzer_with_options(grid, operator_level=level, precond='mg', rtol=0.0, atol=1e-12,
                                          maxiter=500, mg_coarse_max_dofs=200)
        K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                                 analyzer.assemble_body_force_vector())
        u = analyzer.tensor_space.function()
        _, info = analyzer.solve_system(K, F, u, x0=analyzer._prescribed_solution)
        assert info['converged'] and info['precond'] == 'mg'
        solutions[level], iterations[level] = np.asarray(u[:]), info['niter']
        if level == 'fa':
            reference = spsolve(K.to_scipy().tocsc(), np.asarray(F[:]))
    assert iterations['ea'] == iterations['fa']
    for level in ('fa', 'ea'):
        np.testing.assert_allclose(solutions[level], reference, rtol=0.0,
                                   atol=1e-8 * np.max(np.abs(reference)))


def test_analyzer_mg_preconditioner_requires_element_density() -> None:
    bm.set_backend('numpy')
    grid = (6, 2, 2)
    domain = (0.0, 6.0, 0.0, 2.0, 0.0, 2.0)
    mesh = HexahedronMesh.from_box(list(domain), *grid)
    problem = FullMBBBeam3d(domain=domain, support='end_lines', load_subdivisions=(grid[0], grid[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D',
                                              enable_logging=False)
    # 不做密度拓扑优化时没有单元系数, 粗层无从构造
    analyzer = LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                                   integration_order=2, assembly_method='fast', operator_level='fa',
                                   solve_method='cg', solver_options=dict(precond='mg'), enable_logging=False)
    K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(), analyzer.assemble_body_force_vector())
    with pytest.raises(RuntimeError, match="precond='mg'"):
        analyzer.solve_system(K, F, analyzer.tensor_space.function())


def test_hierarchy_coarsens_at_least_once_so_ea_has_a_factorizable_coarse_level() -> None:
    # 6x2x2 只有 189 个自由度, 远小于默认的 coarse_max_dofs; 'ea' 的最细层算子不能直接分解
    analyzer = _analyzer_with_options((6, 2, 2), operator_level='ea', precond='mg', rtol=0.0, atol=1e-13,
                                      maxiter=200)
    density = bm.tensor(np.random.default_rng(6).uniform(0.0, 1.0, 24))
    K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                             analyzer.assemble_body_force_vector())
    u = analyzer.tensor_space.function()
    _, info = analyzer.solve_system(K, F, u, x0=analyzer._prescribed_solution)
    assert analyzer._mg_hierarchy.num_levels >= 2 and info['converged']
