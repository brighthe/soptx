"""平移类参考单元矩阵: 分析器自动判定平移类数 N_k, 只存 N_k 份的回归测试."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.matrix import assemble_csr, build_csr_pattern
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh
from soptx.problems import FullMBBBeamLine3d
from soptx.problems.loads import PointForceLoad
from soptx.topology.interpolation import MaterialInterpolationScheme


class _PointLoadMBB3d(FullMBBBeamLine3d):
    """顶面中心受集中力的三维 MBB 梁: 本测试与载荷形式无关, 点力使奇数 x 向剖分可用."""

    def loads(self):
        x0, x1, _, y1, z0, z1 = self.domain
        return (PointForceLoad(point=((x0 + x1) / 2.0, y1, (z0 + z1) / 2.0), vector=(0.0, self.P, 0.0)),)

GRID = (7, 3, 5)
DOMAIN = (0.0, 7.0, 0.0, 3.0, 0.0, 5.0)


def _analyzer(operator_level='fa', mesh=None, precond='gmg'):
    bm.set_backend('numpy')
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID) if mesh is None else mesh
    problem = _PointLoadMBB3d(domain=DOMAIN)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D')
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': ['E']})
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=2, assembly_method='fast', operator_level=operator_level,
                               solve_method='cg', solver_options=dict(precond=precond, rtol=0.0, atol=1e-13,
                                                                      maxiter=5000),
                               topopt_algorithm='density_based', interpolation_scheme=interpolation)


@pytest.mark.parametrize('n_classes', [1, 3])
@pytest.mark.parametrize('with_scale', [False, True])
def test_assemble_csr_with_reference_matrices_matches_expanded(n_classes, with_scale) -> None:
    bm.set_backend('numpy')
    space = _analyzer().tensor_space
    pattern = build_csr_pattern(space)
    NC = space.mesh.number_of_cells()
    rng = np.random.default_rng(n_classes)
    reference = rng.standard_normal((n_classes, 24, 24))
    scale = bm.tensor(rng.uniform(0.1, 1.0, NC)) if with_scale else None
    expanded = reference[np.arange(NC) % n_classes]

    actual = assemble_csr(bm.tensor(reference), pattern, scale=scale).to_scipy().toarray()
    expected = assemble_csr(bm.tensor(expanded), pattern, scale=scale).to_scipy().toarray()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13 * np.max(np.abs(expected)))


def test_assemble_csr_rejects_non_divisor_reference_count() -> None:
    bm.set_backend('numpy')
    space = _analyzer().tensor_space
    with pytest.raises(ValueError, match='约数'):
        assemble_csr(bm.zeros((2, 24, 24), dtype=bm.float64), build_csr_pattern(space))


def _stretched_mesh():
    """内部一层节点沿 x 平移, 区域不变但不再满足平移类约定的六面体网格."""
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID)
    node = np.asarray(mesh.entity('node')).copy()
    node[np.isclose(node[:, 0], DOMAIN[1] - 1.0), 0] += 0.3
    return HexahedronMesh(bm.tensor(node), bm.tensor(np.asarray(mesh.entity('cell'))))


@pytest.mark.parametrize('mesh_class, expected', [(HexahedronMesh, 1), (QuadrangleMesh, 1),
                                                    (TriangleMesh, 2), (TetrahedronMesh, 6)])
def test_translation_class_count_on_box_mesh(mesh_class, expected) -> None:
    bm.set_backend('numpy')
    if mesh_class in (HexahedronMesh, TetrahedronMesh):
        mesh = mesh_class.from_box(list(DOMAIN), *GRID)
    else:
        mesh = mesh_class.from_box(list(DOMAIN[:4]), *GRID[:2])
    analyzer = object.__new__(LagrangeFEMAnalyzer)
    analyzer._mesh, analyzer._space_degree, analyzer._reference_classes = mesh, 1, None
    assert analyzer._translation_class_count() == expected


def test_translation_class_count_falls_back_on_non_uniform_mesh() -> None:
    analyzer = _analyzer(mesh=_stretched_mesh())
    assert analyzer._translation_class_count() == int(np.prod(GRID))


@pytest.mark.parametrize('operator_level', ['fa', 'ea'])
def test_single_reference_matches_per_cell(operator_level) -> None:
    density = bm.tensor(np.random.default_rng(0).uniform(0.0, 1.0, int(np.prod(GRID))))
    results = {}
    for per_cell in (True, False):
        # 逐单元参考作对照; 多重网格要求单一平移类, 对照改用 Jacobi 预条件
        analyzer = _analyzer(operator_level, precond='jacobi' if per_cell else 'gmg')
        if per_cell:
            analyzer._reference_classes = int(np.prod(GRID))
        K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                                 analyzer.assemble_body_force_vector())
        u = analyzer.tensor_space.function()
        _, info = analyzer.solve_system(K, F, u, x0=analyzer._prescribed_solution)
        assert info['converged']
        uhe = u[:][analyzer.tensor_space.cell_to_dof()]
        energy = analyzer.compute_element_energy_derivative(rho_val=density, uhe=uhe)
        results[per_cell] = (np.asarray(u[:]), np.asarray(energy))
        if not per_cell:
            # 自动判定为单一平移类: 只积分代表单元, 不形成逐单元的 K_e^0
            assert analyzer._translation_class_count() == 1
            assert tuple(analyzer._reference_stiffness_matrices().shape) == (1, 24, 24)
            assert analyzer._cached_ke0 is None
    for index in range(2):
        reference = results[True][index]
        np.testing.assert_allclose(results[False][index], reference, rtol=0.0,
                                   atol=1e-12 * np.max(np.abs(reference)))


def test_non_uniform_mesh_uses_per_cell_reference() -> None:
    analyzer = _analyzer(mesh=_stretched_mesh())
    density = bm.full((int(np.prod(GRID)), ), 0.5, dtype=bm.float64)
    analyzer.assemble_stiff_matrix(rho_val=density)
    assert tuple(analyzer._reference_stiffness_matrices().shape) == (int(np.prod(GRID)), 24, 24)


def test_gmg_rejects_non_uniform_mesh() -> None:
    analyzer = _analyzer(operator_level='ea', mesh=_stretched_mesh())
    density = bm.full((int(np.prod(GRID)), ), 0.5, dtype=bm.float64)
    K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                             analyzer.assemble_body_force_vector())
    with pytest.raises(RuntimeError, match='等距六面体网格'):
        analyzer.solve_system(K, F, analyzer.tensor_space.function())
