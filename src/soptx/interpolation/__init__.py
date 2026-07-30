"""Deprecated mixed interpolation namespace retained for SOPTX 1.1.x."""

from warnings import warn

from soptx.materials import (
    IsotropicLinearElasticMaterial,
    LinearElasticMaterial,
)
from soptx.topology.interpolation import MaterialInterpolationScheme

warn(
    "soptx.interpolation is deprecated; use soptx.materials and "
    "soptx.topology.interpolation",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "IsotropicLinearElasticMaterial",
    "LinearElasticMaterial",
    "MaterialInterpolationScheme",
]
