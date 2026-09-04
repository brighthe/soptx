"""有限元分析器: 装配、边界条件与求解流程的载体.

distributed_analyzer 不在此 eager 导出: 它经 soptx.fem.distributed
触及 mpi4py (可选 extra), 需要时按完整模块路径导入
from soptx.fem.analyzers.distributed_analyzer import ....

builders 则可以安全 eager 导出: build_distributed_analyzer 把分布式实现的
导入延迟到函数体内, 因此导入本包永远不会触及 mpi4py.

Matrix-Free 算子链 (EA 懒装配算子、一步正向求解、加权 Krylov) 见
soptx.fem.matrix_free.
"""

from .builders import build_distributed_analyzer, build_serial_analyzer
from .huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from .lagrange_fem_analyzer import LagrangeFEMAnalyzer
from .substructure import (
    FullInterfaceAnalysisResult,
    FullInterfaceSubstructureAnalyzer,
)

__all__ = [
    "FullInterfaceAnalysisResult",
    "FullInterfaceSubstructureAnalyzer",
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "build_distributed_analyzer",
    "build_serial_analyzer",
]
