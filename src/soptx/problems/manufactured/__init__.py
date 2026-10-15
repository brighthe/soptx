"""制造解问题: 有精确解, 体力由精确解反推."""

from .divergence_free_poly import DivergenceFreePolynomialElasticity3D
from .exponential_sine import (
    ExponentialSineManufacturedElasticity2D,
    MixedBoundaryExponentialSineElasticity2D,
)
from .harmonic_poly import (
    HarmonicPoly2D,
    HarmonicPoly3D,
    HarmonicPolynomialElasticity2D,
    HarmonicPolynomialElasticity3D,
)
from .sinusoidal import (
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)

__all__ = [
    "DivergenceFreePolynomialElasticity3D",
    "ExponentialSineManufacturedElasticity2D",
    "HarmonicPoly2D",
    "HarmonicPoly3D",
    "HarmonicPolynomialElasticity2D",
    "HarmonicPolynomialElasticity3D",
    "MixedBoundaryExponentialSineElasticity2D",
    "MixedBoundarySinusoidalElasticity2D",
    "SinusoidalElasticity2D",
    "SinusoidalPlaneStrainElasticity2D",
]
