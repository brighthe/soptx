"""Compatibility imports for mixed face-source integrators."""

from soptx.fem.integrators.face_source_integrator_mfem import (
    BoundaryFaceSourceIntegrator_mfem,
    InterFaceSourceIntegrator,
    _FaceSourceIntegrator,
)

__all__ = [
    "BoundaryFaceSourceIntegrator_mfem",
    "InterFaceSourceIntegrator",
]
