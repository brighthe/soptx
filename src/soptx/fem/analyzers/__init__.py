"""有限元分析器: 装配、边界条件与求解流程的载体.

EA 懒装配算子门面见 soptx.fem.matrix_free.
"""

from .builders import build_serial_analyzer
from .huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from .lagrange_fem_analyzer import LagrangeFEMAnalyzer
from .substructure import (
    FullInterfaceAnalysisResult,
    FullInterfaceSubstructureAnalyzer,
)
from .substructure_analyzer import SubstructureAnalyzer

__all__ = [
    "FullInterfaceAnalysisResult",
    "FullInterfaceSubstructureAnalyzer",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "SubstructureAnalyzer",
    "build_serial_analyzer",
]
