"""子结构局部缩聚策略的统一公共入口."""

from .base import (
    CondensationReductionAdapter,
    LocalReduction,
    LocalReductionBatchResult,
    LocalReductionResult,
    ReductionDiagnostics,
)
from .exact_schur import ExactSchurReduction
from .piml_shape import PIMLShapeReduction
from .piml_stiffness import PIMLStiffnessReduction

__all__ = [
    "CondensationReductionAdapter", "LocalReduction", "LocalReductionBatchResult",
    "LocalReductionResult", "ReductionDiagnostics", "ExactSchurReduction",
    "PIMLShapeReduction", "PIMLStiffnessReduction",
]
