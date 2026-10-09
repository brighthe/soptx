"""多分辨率 (density_location='element_multiresolution') 子单元刚度与刚度导数的回归测试.

两条检验都不依赖实现细节:

1. 积分可加性: 各子密度单元的实体刚度之和等于整个位移单元的 K_e^0;
2. 系数可提出: 子单元上的材料系数为常数, 故 dK_{e,n}/drho_{e,n} = (E'(rho_{e,n}) / E_0) K^0_{e,n}.

多分辨率只支持二维四边形网格 (``map_bcs_to_sub_elements`` 只接受张量积积分点). 网格取
非正方形的矩形单元, p=1 时被积函数是多项式, 积分阶 3 下两条检验都只差舍入.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import QuadrangleMesh
from soptx.problems import HalfMBBBeamRight2d
from soptx.topology.interpolation import MaterialInterpolationScheme

DOMAIN = (0.0, 6.0, 0.0, 2.0)
NX, NY = 4, 3
E0 = 2.0
# n_sub 在 4..9 之间时 compute_sub_element_stiffness_matrix 取积分阶 3, 与分析器的一致
INTEGRATION_ORDER = 3


@pytest.fixture(autouse=True)
def reset_backend():
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _analyzer(n_sub: int):
    """矩形单元上的多分辨率 FA 分析器, 已按 n_sub 建好密度分布."""
    mesh = QuadrangleMesh.from_box(list(DOMAIN), nx=NX, ny=NY)
    # 多分辨率数据重排读取 meshdata (调用方约定, 见 docs/topology/filters.md), 与各 pipeline 同口径
    mesh.meshdata = {'nx': NX, 'ny': NY,
                     'hx': (DOMAIN[1] - DOMAIN[0]) / NX, 'hy': (DOMAIN[3] - DOMAIN[2]) / NY}
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=0.3,
                                              hypothesis="plane_stress", enable_logging=False)
    scheme = MaterialInterpolationScheme(
        density_location="element_multiresolution", interpolation_method="msimp",
        options={"penalty_factor": 3.0, "void_youngs_modulus": 1e-9, "target_variables": ["E"]},
        enable_logging=False)
    refine = int(math.isqrt(n_sub))
    design_mesh = QuadrangleMesh.from_box(list(DOMAIN), nx=NX * refine, ny=NY * refine)
    scheme.setup_density_distribution(design_variable_mesh=design_mesh, displacement_mesh=mesh,
                                      sub_density_element=n_sub)
    analyzer = LagrangeFEMAnalyzer(disp_mesh=mesh, pde=HalfMBBBeamRight2d(domain=DOMAIN), material=material,
                                   space_degree=1, integration_order=INTEGRATION_ORDER,
                                   assembly_method="standard", operator_level="fa", solve_method="scipy",
                                   topopt_algorithm="density_based", interpolation_scheme=scheme,
                                   enable_logging=False)
    return analyzer


@pytest.mark.parametrize("n_sub", [4, 9])
def test_sub_element_matrices_sum_to_element_matrix(n_sub: int) -> None:
    analyzer = _analyzer(n_sub)

    ke0_sub = bm.to_numpy(analyzer.compute_sub_element_stiffness_matrix())
    ke0 = bm.to_numpy(analyzer.compute_solid_stiffness_matrix())

    NC, TLDOF = ke0.shape[0], ke0.shape[-1]
    assert ke0_sub.shape == (NC, n_sub, TLDOF, TLDOF)
    np.testing.assert_allclose(ke0_sub.sum(axis=1), ke0, rtol=0.0, atol=1e-12 * np.max(np.abs(ke0)))


@pytest.mark.parametrize("n_sub", [4, 9])
def test_stiffness_derivative_factors_out_the_coefficient(n_sub: int) -> None:
    analyzer = _analyzer(n_sub)
    NC = analyzer.disp_mesh.number_of_cells()
    rho = bm.tensor(np.random.default_rng(n_sub).uniform(0.1, 1.0, (NC, n_sub)))

    diff_ke = bm.to_numpy(analyzer.compute_stiffness_matrix_derivative(rho_val=rho))

    dE = analyzer.interpolation_scheme.interpolate_material_derivative(
        material=analyzer.material, rho_val=rho, integration_order=INTEGRATION_ORDER)
    dE = dE[0] if isinstance(dE, tuple) else dE
    scale = bm.to_numpy(dE) / E0
    ke0_sub = bm.to_numpy(analyzer.compute_sub_element_stiffness_matrix())
    expected = scale[:, :, None, None] * ke0_sub

    assert diff_ke.shape == expected.shape
    np.testing.assert_allclose(diff_ke, expected, rtol=0.0, atol=1e-12 * np.max(np.abs(expected)))
