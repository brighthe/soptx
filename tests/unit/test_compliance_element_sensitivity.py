"""单元密度柔顺度灵敏度与实体单元刚度缓存的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.integrators import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective


def _setup(assembly_method, nu=0.3, target_variables=('E', )):
    """在 6x2x2 的 MBB 梁上构造单元密度 FA 分析器、柔顺度目标与一个随机密度场."""
    bm.set_backend('numpy')
    grid = (6, 2, 2)
    domain = (0.0, 6.0, 0.0, 2.0, 0.0, 2.0)
    mesh = HexahedronMesh.from_box(list(domain), *grid)
    problem = FullMBBBeam3d(domain=domain, nu=nu, support='end_lines', load_subdivisions=(grid[0], grid[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=nu, hypothesis='3D',
                                              enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': list(target_variables)},
        enable_logging=False)
    analyzer = LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                                   integration_order=2, assembly_method=assembly_method, operator_level='fa',
                                   solve_method='scipy', topopt_algorithm='density_based',
                                   interpolation_scheme=interpolation, enable_logging=False)
    objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual',
                                    enable_logging=False)
    density = bm.tensor(np.random.default_rng(0).uniform(0.1, 1.0, mesh.number_of_cells()))
    return analyzer, objective, density


def _sensitivity_with_full_derivative(analyzer, density, state):
    """构造完整导数矩阵后缩并的写法, 即修改之前的柔顺度灵敏度.

    与原代码同用 bm.einsum: 近不可压缩时 lambda / mu 很大, 换成 np.einsum 的缩并顺序会把
    舍入放大到 1e-11 量级.
    """
    uhe = state['displacement'][analyzer.tensor_space.cell_to_dof()]
    diff_ke = analyzer.compute_stiffness_matrix_derivative(rho_val=density)
    return -np.asarray(bm.to_numpy(bm.einsum('ci, cij, cj -> c', uhe, diff_ke, uhe)))


@pytest.mark.parametrize('assembly_method', ['standard', 'fast'])
def test_solid_stiffness_cache_uses_analyzer_assembly_method(assembly_method) -> None:
    analyzer, _, _ = _setup(assembly_method)
    ke0 = bm.to_numpy(analyzer.compute_solid_stiffness_matrix())
    expected = LinearElasticIntegrator(material=analyzer._material, q=2,
                                       method=assembly_method).assembly(space=analyzer.tensor_space)
    np.testing.assert_array_equal(ke0, bm.to_numpy(expected))


@pytest.mark.parametrize('assembly_method', ['standard', 'fast'])
def test_element_compliance_sensitivity_matches_full_derivative(assembly_method) -> None:
    analyzer, objective, density = _setup(assembly_method)
    state = analyzer.solve_state(rho_val=density)

    dc = np.asarray(bm.to_numpy(objective.jac(density=density, state=state)))
    expected = _sensitivity_with_full_derivative(analyzer, density, state)
    np.testing.assert_allclose(dc, expected, rtol=1e-12, atol=1e-14 * np.max(np.abs(expected)))
    assert np.all(dc <= 0.0)


def test_poisson_interpolated_sensitivity_falls_back_to_full_derivative() -> None:
    analyzer, objective, density = _setup('standard', nu=0.4999, target_variables=('E', 'nu'))
    state = analyzer.solve_state(rho_val=density)

    dc = np.asarray(bm.to_numpy(objective.jac(density=density, state=state)))
    expected = _sensitivity_with_full_derivative(analyzer, density, state)
    np.testing.assert_allclose(dc, expected, rtol=1e-12, atol=1e-14 * np.max(np.abs(expected)))
