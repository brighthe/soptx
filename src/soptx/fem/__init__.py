"""Finite-element spaces, integrators, analyzers and matrix-free operators.

``soptx.fem.distributed`` and ``soptx.fem.analyzers.distributed_analyzer`` are
deliberately *not* re-exported here: both import ``mpi4py``, which is an
optional extra, so importing them eagerly would make ``import soptx.fem`` fail
on an installation without it.  Import those two by their full module path.
"""

from .analyzers import (
    FullInterfaceAnalysisResult,
    FullInterfaceSubstructureAnalyzer,
    HuZhangMFEMAnalyzer,
    LagrangeFEMAnalyzer,
)
from .bilinear_form import BilinearForm
from .boundary_loads import (
    LoadResultantReport,
    P1TraceLoad,
    boundary_load_resultant,
    check_boundary_load_resultant,
    project_patch_traction_to_p1_trace,
)
from .integrators import LinearElasticIntegrator, SourceIntegrator
from .matrix import CSRPattern, assemble_csr, build_csr_pattern
from .spaces import HuZhangFESpace, create_huzhang_checkerboard_mesh

__all__ = [
    "BilinearForm",
    "CSRPattern",
    "FullInterfaceAnalysisResult",
    "FullInterfaceSubstructureAnalyzer",
    "HuZhangFESpace",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "LinearElasticIntegrator",
    "LoadResultantReport",
    "P1TraceLoad",
    "SourceIntegrator",
    "assemble_csr",
    "boundary_load_resultant",
    "build_csr_pattern",
    "check_boundary_load_resultant",
    "create_huzhang_checkerboard_mesh",
    "project_patch_traction_to_p1_trace",
]
