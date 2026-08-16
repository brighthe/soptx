"""Finite-element spaces, integrators and solver workflows.

``soptx.fem.distributed`` and ``soptx.fem.solvers.matrix_free_analyzer`` are
deliberately *not* re-exported here: both import ``mpi4py``, which is an
optional extra, so importing them eagerly would make ``import soptx.fem`` fail
on an installation without it.  Import those two by their full module path.
"""

from .boundary_loads import P1TraceLoad, project_patch_traction_to_p1_trace
from .integrators import LinearElasticIntegrator, SourceIntegrator
from .solvers import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer
from .spaces import HuZhangFESpace, create_huzhang_checkerboard_mesh

__all__ = [
    "HuZhangFESpace",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "LinearElasticIntegrator",
    "P1TraceLoad",
    "SourceIntegrator",
    "create_huzhang_checkerboard_mesh",
    "project_patch_traction_to_p1_trace",
]
