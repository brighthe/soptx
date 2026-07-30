"""Compatibility imports for Lagrange face-source integrators."""

from soptx.fem.integrators.face_source_integrator_lfem import (
    BoundaryFaceSourceIntegrator_lfem,
    InterFaceSourceIntegrator,
    _FaceSourceIntegrator,
)

__all__ = [
    "BoundaryFaceSourceIntegrator_lfem",
    "InterFaceSourceIntegrator",
]
