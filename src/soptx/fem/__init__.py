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
from .spaces import HuZhangFESpace

# 向后兼容别名: 结构网格生成器已迁至 soptx.mesh.structured_triangle,
# 此处保留旧导入路径, 新代码请直接从 soptx.mesh 导入。
from ..mesh import (
    create_huzhang_checkerboard_mesh,
    create_huzhang_symmetric_single_diagonal_mesh,
)

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
    "HuZhangFESpace",
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
    "create_huzhang_checkerboard_mesh",
    "create_huzhang_symmetric_single_diagonal_mesh",
    "create_level",
    "project_patch_traction_to_p1_trace",
    "register_level",
]
