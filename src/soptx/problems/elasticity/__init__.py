"""线弹性问题。"""

from .mbb import HalfMBBBeamRight2d, HalfMBBBeamRight3d, FullMBBBeam2d, FullMBBBeam3d
from .fixed_fixed import FixedFixedBeamCenterLoad2d, FixedFixedBeamHalfDomain2d
from .bearing import BearingDevice2d
from .bridge import SimplySupportedBridge2d
from .cantilever import (
    CantileverCorner2d,
    CantileverMiddle2d,
    CantileverRightBottomEdge3d,
)

from .manufactured_2d import (
    ExponentialSineManufacturedElasticity2D,
    HarmonicPoly2D,
    HarmonicPolynomialElasticity2D,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)
from .manufactured_3d import (
    DivergenceFreePolynomialElasticity3D,
    HarmonicPoly3D,
    HarmonicPolynomialElasticity3D,
)

__all__ = [
    "BearingDevice2d",
    "CantileverCorner2d",
    "CantileverMiddle2d",
    "CantileverRightBottomEdge3d",
    "DivergenceFreePolynomialElasticity3D",
    "ExponentialSineManufacturedElasticity2D",
    "FullMBBBeam2d",
    "FullMBBBeam3d",
    "FixedFixedBeamCenterLoad2d",
    "FixedFixedBeamHalfDomain2d",
    "HalfMBBBeamRight2d",
    "HalfMBBBeamRight3d",
    "HarmonicPoly2D",
    "HarmonicPoly3D",
    "HarmonicPolynomialElasticity2D",
    "HarmonicPolynomialElasticity3D",
    "MixedBoundaryExponentialSineElasticity2D",
    "MixedBoundarySinusoidalElasticity2D",
    "SimplySupportedBridge2d",
    "SinusoidalElasticity2D",
    "SinusoidalPlaneStrainElasticity2D",
]
