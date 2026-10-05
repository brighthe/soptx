from __future__ import annotations

import importlib

import pytest

import soptx


def test_root_package_exports_only_version() -> None:
    assert soptx.__all__ == ["__version__"]
    assert soptx.__version__ == "1.2.0.dev0"


def test_stable_subpackage_imports() -> None:
    from soptx.fem.integrators import LinearElasticIntegrator
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems import SinusoidalElasticity2D, SinusoidalPlaneStrainElasticity2D

    assert LinearElasticIntegrator.__name__ == "LinearElasticIntegrator"
    assert (
        IsotropicLinearElasticMaterial.__name__
        == "IsotropicLinearElasticMaterial"
    )
    assert SinusoidalElasticity2D.__name__ == "SinusoidalElasticity2D"
    assert SinusoidalPlaneStrainElasticity2D is SinusoidalElasticity2D


@pytest.mark.parametrize(
    "name", ["analysis", "interpolation", "model", "optimization", "regularization", "utils"]
)
def test_removed_compatibility_namespaces(name: str) -> None:
    """1.1.x 的兼容路径在 1.2 起删除, 不得重新出现."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"soptx.{name}")
