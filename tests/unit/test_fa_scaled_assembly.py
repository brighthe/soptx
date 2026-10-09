"""单元密度下 FA 由缓存 K_e^0 缩放装配的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.levels import FullAssembly
from soptx.fem.matrix import assemble_csr, build_csr_pattern
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme


def _analyzer(assembly_method, nu=0.3, target_variables=('E', )):
    """在 6x2x2 的 MBB 梁上构造单元密度的 FA 分析器与一个随机密度场."""
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
    density = bm.tensor(np.random.default_rng(0).uniform(0.1, 1.0, mesh.number_of_cells()))
    return analyzer, density


def _dense(matrix):
    return np.asarray(matrix.to_scipy().toarray())


def _integrated(analyzer):
    """按积分子当前系数逐单元积分后装配, 即修改之前的 FA 路线."""
    return FullAssembly.build(analyzer.tensor_space, analyzer._integrator).operator


@pytest.mark.parametrize('assembly_method', ['standard', 'fast'])
def test_scaled_reference_matches_integration(assembly_method) -> None:
    analyzer, density = _analyzer(assembly_method)
    scaled = _dense(analyzer.assemble_stiff_matrix(rho_val=density))
    integrated = _dense(_integrated(analyzer))

    np.testing.assert_allclose(scaled, integrated, rtol=0.0, atol=1e-13 * np.max(np.abs(integrated)))


def test_scaled_reference_is_reused_across_density_updates() -> None:
    analyzer, density = _analyzer('fast')
    analyzer.assemble_stiff_matrix(rho_val=density)
    updated = 0.5 * density + 0.05
    scaled = _dense(analyzer.assemble_stiff_matrix(rho_val=updated))
    integrated = _dense(_integrated(analyzer))

    np.testing.assert_allclose(scaled, integrated, rtol=0.0, atol=1e-13 * np.max(np.abs(integrated)))


def test_poisson_interpolation_keeps_integration_route() -> None:
    analyzer, density = _analyzer('standard', nu=0.4999, target_variables=('E', 'nu'))
    stiffness = _dense(analyzer.assemble_stiff_matrix(rho_val=density))
    # 系数为逐单元本构矩阵, 不是单元标量: 层级由积分装配, 不带缩放系数
    assert analyzer._level.scale is None

    np.testing.assert_array_equal(stiffness, _dense(_integrated(analyzer)))


def test_assemble_csr_scale_equals_prescaled_element_matrices() -> None:
    analyzer, density = _analyzer('fast')
    ke0 = analyzer.compute_solid_stiffness_matrix()
    pattern = build_csr_pattern(analyzer.tensor_space)
    scale = bm.tensor(np.random.default_rng(1).uniform(0.0, 2.0, ke0.shape[0]))

    scaled = np.array(assemble_csr(ke0, pattern, scale=scale).values)
    prescaled = np.array(assemble_csr(scale[:, None, None] * ke0, pattern).values)
    np.testing.assert_array_equal(scaled, prescaled)
