"""Finite-element spaces, integrators and solver workflows."""

from .integrators import LinearElasticIntegrator, SourceIntegrator
from .meshes import create_huzhang_checkerboard_mesh
from .solvers import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer
from .spaces import HuZhangFESpace

__all__ = [
    "HuZhangFESpace",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "LinearElasticIntegrator",
    "SourceIntegrator",
    "create_huzhang_checkerboard_mesh",
]
