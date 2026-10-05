"""Finite-element integrators, analyzers and matrix-free operators.

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
from .kernels import (
    ElementRestriction,
    GeometricFactors,
    LinearElasticQFunction,
    ReferenceBasis,
)
from .levels import (
    AssemblyLevelExtension,
    ElementAssembly,
    FullAssembly,
    PartialAssembly,
    available_levels,
    create_level,
    register_level,
)
from .linear_form import LinearForm
from .matrix import CSRPattern, assemble_csr, build_csr_pattern
from .operators import ConstrainedOperator

__all__ = [
    "AssemblyLevelExtension",
    "BilinearForm",
    "CSRPattern",
    "ConstrainedOperator",
    "ElementAssembly",
    "ElementRestriction",
    "FullAssembly",
    "GeometricFactors",
    "FullInterfaceAnalysisResult",
    "FullInterfaceSubstructureAnalyzer",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "LinearElasticIntegrator",
    "LinearElasticQFunction",
    "LinearForm",
    "LoadResultantReport",
    "P1TraceLoad",
    "PartialAssembly",
    "ReferenceBasis",
    "SourceIntegrator",
    "assemble_csr",
    "available_levels",
    "boundary_load_resultant",
    "build_csr_pattern",
    "check_boundary_load_resultant",
    "create_level",
    "project_patch_traction_to_p1_trace",
    "register_level",
]
