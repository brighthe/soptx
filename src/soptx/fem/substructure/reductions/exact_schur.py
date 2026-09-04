"""精确 Schur 补局部缩聚策略."""

from typing import Any

from ..condensation import FEAStaticCondensation
from .base import (
    CondensationReductionAdapter,
    LocalReductionBatchResult,
)


class ExactSchurReduction(CondensationReductionAdapter):
    """以组合方式适配现有精确有限元缩聚实现."""

    def __init__(self, i_dofs: Any, b_dofs: Any) -> None:
        super().__init__(
            FEAStaticCondensation(i_dofs, b_dofs),
            requested_method="exact_schur",
            stiffness_source="exact_schur",
            recovery_source="exact_schur",
        )

    def reduce_many(
        self,
        local_stiffness_batch: Any,
        density_batch: Any = None,
    ) -> LocalReductionBatchResult:
        """以一次后端批量 solve 完成全部 Exact Schur 局部缩聚."""
        if getattr(local_stiffness_batch, "ndim", 0) != 3:
            raise ValueError(
                "ExactSchurReduction.reduce_many() 要求形状为 "
                "(n_substructure, n_dof, n_dof)."
            )
        if (
            density_batch is not None
            and len(density_batch) != len(local_stiffness_batch)
        ):
            raise ValueError("density_batch 与 local_stiffness_batch 的批量长度不一致.")

        stiffness, recovery = self.legacy.condense(
            local_stiffness_batch, density_batch
        )
        diagnostics = tuple(
            self._diagnostics(None) for _ in range(len(local_stiffness_batch))
        )
        return LocalReductionBatchResult(stiffness, recovery, diagnostics)
