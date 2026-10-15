"""参考子结构的 fast 单元积分与 standard 积分的数值等价性."""

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.fem.substructure.mesh import SubstructurePrototype


@pytest.mark.parametrize('degree', [1, 2])
@pytest.mark.parametrize('hypothesis,cell_size,n_fine', [
    ('plane_stress', (2.0, 3.0), (2, 2)),
    ('plane_strain', (2.0, 3.0), (2, 2)),
    ('3D', (2.0, 3.0, 4.0), (2, 2, 2)),
])
def test_prototype_stiffness_matches_standard(degree, hypothesis, cell_size, n_fine):
    """不同维数、材料假设和次数下, 单元刚度保持 standard 积分结果."""
    bm.set_backend('numpy')
    prototype = SubstructurePrototype(
        cell_size=cell_size, n_fine=n_fine, E_base=2.5, nu=0.27,
        degree=degree, integration_order=degree + 1, hypothesis=hypothesis,
    )
    standard = LinearElasticIntegrator(
        material=prototype.material, q=prototype.integration_order, method='standard',
    ).assembly(prototype.space)
    actual = bm.to_numpy(prototype.KE_unit)
    expected = bm.to_numpy(standard)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual, actual.swapaxes(-1, -2), rtol=1.0e-12, atol=1.0e-12)
