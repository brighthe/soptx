"""Finite-element spaces, integrators and solver workflows."""

from .integrators import LinearElasticIntegrator, SourceIntegrator
from .solvers import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer
from .spaces import HuZhangFESpace, create_huzhang_checkerboard_mesh

__all__ = [
    "HuZhangFESpace",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "LinearElasticIntegrator",
    "SourceIntegrator",
    "create_huzhang_checkerboard_mesh",
]
