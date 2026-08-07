"""Mathematical problems independent of meshes and material instances."""

from .elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    FullMBBBeam3d,
    HalfMBBBeamRight2d,
    HalfMBBBeamRight3d,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)

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
