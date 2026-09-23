"""PIML 缩聚刚度直接预测策略."""

from typing import Any, Optional

from ..piml_surrogate import ReducedStiffnessCondensation
from .base import CondensationReductionAdapter


class PIMLStiffnessReduction(CondensationReductionAdapter):
    """路线 B 的规范适配器：预测刚度，恢复关系仍取 Exact Schur."""

    def __init__(
        self,
        i_dofs: Any,
        b_dofs: Any,
        model: Optional[Any] = None,
        is_cholesky: bool = True,
        range_basis: Optional[Any] = None,
        rcond_min: float = 1.0e-8,
    ) -> None:
        super().__init__(
            ReducedStiffnessCondensation(
                i_dofs,
                b_dofs,
                model=model,
                is_cholesky=is_cholesky,
                range_basis=range_basis,
                rcond_min=rcond_min,
            ),
            requested_method="piml_stiffness",
            stiffness_source="piml_stiffness",
            recovery_source="exact_schur",
        )
