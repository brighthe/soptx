"""Deprecated mixed model namespace retained for SOPTX 1.1.x."""

from warnings import warn

from soptx.problems import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)

warn(
    "soptx.model is deprecated; import mathematical problems from "
    "soptx.problems",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DivergenceFreePolynomialElasticity3D",
    "ExponentialSineManufacturedElasticity2D",
    "SinusoidalPlaneStrainElasticity2D",
]
