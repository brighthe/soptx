"""线弹性问题。"""

from .mbb import HalfMBBBeamRight2d, HalfMBBBeamRight3d, FullMBBBeam3d

from .manufactured_2d import (
    ExponentialSineManufacturedElasticity2D,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)
from .manufactured_3d import DivergenceFreePolynomialElasticity3D

__all__ = [
    "DivergenceFreePolynomialElasticity3D",
    "ExponentialSineManufacturedElasticity2D",
    "FullMBBBeam3d",
    "HalfMBBBeamRight2d",
    "HalfMBBBeamRight3d",
    "MixedBoundaryExponentialSineElasticity2D",
    "MixedBoundarySinusoidalElasticity2D",
    "SinusoidalPlaneStrainElasticity2D",
]
