from __future__ import annotations

import soptx


def test_root_package_exports_only_version() -> None:
    assert soptx.__all__ == ["__version__"]
    assert soptx.__version__ == "1.1.0.dev0"


def test_stable_subpackage_imports() -> None:
    from soptx.fem.integrators import LinearElasticIntegrator
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems import SinusoidalPlaneStrainElasticity2D

    assert LinearElasticIntegrator.__name__ == "LinearElasticIntegrator"
    assert (
        IsotropicLinearElasticMaterial.__name__
        == "IsotropicLinearElasticMaterial"
    )
    assert (
        SinusoidalPlaneStrainElasticity2D.__name__
        == "SinusoidalPlaneStrainElasticity2D"
    )
