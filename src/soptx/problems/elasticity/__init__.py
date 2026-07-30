"""Manufactured linear-elasticity problems."""

from .manufactured_2d import (
    ExponentialSineManufacturedElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)
from .manufactured_3d import DivergenceFreePolynomialElasticity3D

__all__ = [
    "DivergenceFreePolynomialElasticity3D",
    "ExponentialSineManufacturedElasticity2D",
    "SinusoidalPlaneStrainElasticity2D",
]
