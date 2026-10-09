"""``soptx.materials.lame_split`` 的单元测试.

1. 常数 (E, ν) 下 ``elastic_matrices`` 与 ``IsotropicLinearElasticMaterial.D`` 一致 (三种假设);
2. ``lame_parameter_derivatives`` 与中心差分一致;
3. 未知假设报错.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.materials import (
    IsotropicLinearElasticMaterial,
    elastic_matrices,
    lame_parameter_derivatives,
    lame_parameters,
)

HYPOTHESES = ("3D", "plane_strain", "plane_stress")


@pytest.fixture(autouse=True)
def reset_backend():
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


@pytest.mark.parametrize("hypothesis", HYPOTHESES)
@pytest.mark.parametrize("E, nu", [(1.0, 0.3), (210.0, 0.4999)])
def test_elastic_matrices_match_material(hypothesis: str, E: float, nu: float) -> None:
    material = IsotropicLinearElasticMaterial(youngs_modulus=E, poisson_ratio=nu,
                                              hypothesis=hypothesis, enable_logging=False)
    E_c = bm.full((4, ), E, dtype=bm.float64)
    nu_c = bm.full((4, ), nu, dtype=bm.float64)

    D = bm.to_numpy(elastic_matrices(E_c, nu_c, hypothesis))
    D0 = bm.to_numpy(material.D)

    assert D.shape == (4, ) + D0.shape
    np.testing.assert_allclose(D, np.broadcast_to(D0, D.shape), rtol=1e-12, atol=1e-12 * np.max(np.abs(D0)))


@pytest.mark.parametrize("hypothesis", HYPOTHESES)
def test_derivatives_match_central_difference(hypothesis: str) -> None:
    E = bm.tensor([0.5, 1.0, 3.0], dtype=bm.float64)
    nu = bm.tensor([0.1, 0.3, 0.45], dtype=bm.float64)
    h = 1e-6

    dlam_dE, dlam_dnu, dmu_dE, dmu_dnu = (bm.to_numpy(d) for d in lame_parameter_derivatives(E, nu, hypothesis))

    lam_p, mu_p = lame_parameters(E + h, nu, hypothesis)
    lam_m, mu_m = lame_parameters(E - h, nu, hypothesis)
    np.testing.assert_allclose(dlam_dE, bm.to_numpy(lam_p - lam_m) / (2 * h), rtol=1e-7)
    np.testing.assert_allclose(dmu_dE, bm.to_numpy(mu_p - mu_m) / (2 * h), rtol=1e-7)

    lam_p, mu_p = lame_parameters(E, nu + h, hypothesis)
    lam_m, mu_m = lame_parameters(E, nu - h, hypothesis)
    np.testing.assert_allclose(dlam_dnu, bm.to_numpy(lam_p - lam_m) / (2 * h), rtol=1e-7)
    np.testing.assert_allclose(dmu_dnu, bm.to_numpy(mu_p - mu_m) / (2 * h), rtol=1e-7)


def test_unknown_hypothesis_is_rejected() -> None:
    E = bm.ones((2, ), dtype=bm.float64)
    with pytest.raises(ValueError, match="hypothesis"):
        lame_parameters(E, 0.3 * E, "axisymmetric")
