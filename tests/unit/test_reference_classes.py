"""``reference_classes``: 平移类参考单元矩阵只存 N_k 份的回归测试."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.matrix import assemble_csr, build_csr_pattern
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme

GRID = (7, 3, 5)
DOMAIN = (0.0, 7.0, 0.0, 3.0, 0.0, 5.0)


def _analyzer(operator_level='fa', reference_classes=None, mesh=None):
    bm.set_backend('numpy')
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID) if mesh is None else mesh
    problem = FullMBBBeam3d(domain=DOMAIN, support='end_lines', load_subdivisions=(GRID[0], GRID[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D',
                                              enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': ['E']},
        enable_logging=False)
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=2, assembly_method='fast', operator_level=operator_level,
                               solve_method='cg', solver_options=dict(precond='mg', rtol=0.0, atol=1e-13,
                                                                      maxiter=500),
                               topopt_algorithm='density_based', interpolation_scheme=interpolation,
                               enable_logging=False, reference_classes=reference_classes)


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


@pytest.mark.parametrize('operator_level', ['fa', 'ea'])
def test_single_reference_matches_per_cell(operator_level) -> None:
    density = bm.tensor(np.random.default_rng(0).uniform(0.0, 1.0, int(np.prod(GRID))))
    results = {}
    for reference_classes in (None, 1):
        analyzer = _analyzer(operator_level, reference_classes)
        K, F = analyzer.apply_bc(analyzer.assemble_stiff_matrix(rho_val=density),
                                 analyzer.assemble_body_force_vector())
        u = analyzer.tensor_space.function()
        _, info = analyzer.solve_system(K, F, u, x0=analyzer._prescribed_solution)
        assert info['converged']
        uhe = u[:][analyzer.tensor_space.cell_to_dof()]
        energy = analyzer.compute_element_energy_derivative(rho_val=density, uhe=uhe)
        results[reference_classes] = (np.asarray(u[:]), np.asarray(energy))
        if reference_classes == 1:
            # 只积分代表单元, 不形成逐单元的 K_e^0
            assert tuple(analyzer._reference_stiffness_matrices().shape) == (1, 24, 24)
            assert analyzer._cached_ke0 is None
    for index in range(2):
        reference = results[None][index]
        np.testing.assert_allclose(results[1][index], reference, rtol=0.0,
                                   atol=1e-12 * np.max(np.abs(reference)))


def test_spot_check_rejects_non_uniform_mesh() -> None:
    bm.set_backend('numpy')
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID)
    node = np.asarray(mesh.entity('node')).copy()
    node[np.isclose(node[:, 0], DOMAIN[1]), 0] += 0.5          # 最后一层单元被拉长
    stretched = HexahedronMesh(bm.tensor(node), bm.tensor(np.asarray(mesh.entity('cell'))))
    analyzer = _analyzer(reference_classes=1, mesh=stretched)
    density = bm.full((int(np.prod(GRID)), ), 0.5, dtype=bm.float64)
    with pytest.raises(RuntimeError, match='reference_classes=1 与网格不符'):
        analyzer.assemble_stiff_matrix(rho_val=density)


@pytest.mark.parametrize('value', [0, 2, 1.5, True])
def test_reference_classes_must_divide_cell_count(value) -> None:
    with pytest.raises(RuntimeError, match='reference_classes'):
        _analyzer(reference_classes=value)
