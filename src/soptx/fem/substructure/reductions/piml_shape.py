"""PIML 形函数预测—变分构造局部缩聚策略."""

from typing import Any, Optional

import torch
from fealpy.backend import backend_manager as bm

from ..piml_surrogate import ShapeFunctionCondensation, SurrogateContractError
from .base import (
    CondensationReductionAdapter,
    LocalReductionBatchResult,
    ReductionDiagnostics,
)


class PIMLShapeReduction(CondensationReductionAdapter):
    """路线 A 的规范适配器：预测内部延拓并按变分式构造降阶刚度.

    ``trace`` 为 ``None`` 时结果落在完整接口上, 与其余 ``LocalReduction`` 实现
    一致; 给出 ``trace`` 时结果直接落在该迹空间上。这不是事后投影: 网络的输出
    维数定义在迹空间中, 详见 ``ShapeFunctionCondensation`` 的类文档。
    """

    def __init__(
        self,
        i_dofs: Any,
        b_dofs: Any,
        model: Optional[Any] = None,
        rigid_basis: Optional[Any] = None,
        deformation_basis: Optional[Any] = None,
        rigid_interior: Optional[Any] = None,
        rigid_tol: float = 1.0e-10,
        excess_rtol: float = 2.0e-2,
        rcond_min: float = 1.0e-8,
        trace: Optional[Any] = None,
    ) -> None:
        super().__init__(
            ShapeFunctionCondensation(
                i_dofs,
                b_dofs,
                model=model,
                rigid_basis=rigid_basis,
                deformation_basis=deformation_basis,
                rigid_interior=rigid_interior,
                rigid_tol=rigid_tol,
                excess_rtol=excess_rtol,
                rcond_min=rcond_min,
                trace=trace,
            ),
            requested_method="piml_shape",
            stiffness_source="piml_shape",
            recovery_source="piml_shape",
        )

    def _exact_batch_fallback(
        self,
        local_stiffness_batch: Any,
        density_batch: Optional[Any],
        *,
        reason: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> LocalReductionBatchResult:
        """以一次 Exact Schur 批量调用回退整个输入批次, 并降到当前迹空间."""
        stiffness, recovery = self.legacy._reduce_exact(
            self.legacy.fallback_solver.condense(
                local_stiffness_batch, density_batch
            )
        )
        diagnostics = tuple(
            ReductionDiagnostics(
                requested_method="piml_shape",
                stiffness_source="exact_schur",
                recovery_source="exact_schur",
                used_fallback=True,
                fallback_reason=reason,
                metrics={} if metrics is None else metrics,
            )
            for _ in range(len(local_stiffness_batch))
        )
        return LocalReductionBatchResult(stiffness, recovery, diagnostics)

    @staticmethod
    def _successful_diagnostics(metrics: dict[str, Any]) -> ReductionDiagnostics:
        return ReductionDiagnostics(
            requested_method="piml_shape",
            stiffness_source="piml_shape",
            recovery_source="piml_shape",
            metrics=metrics,
        )

    @staticmethod
    def _fallback_diagnostics(
        reason: str,
        metrics: dict[str, Any],
    ) -> ReductionDiagnostics:
        return ReductionDiagnostics(
            requested_method="piml_shape",
            stiffness_source="exact_schur",
            recovery_source="exact_schur",
            used_fallback=True,
            fallback_reason=reason,
            metrics=metrics,
        )

    def reduce_many(
        self,
        local_stiffness_batch: Any,
        density_batch: Optional[Any] = None,
    ) -> LocalReductionBatchResult:
        """批量执行路线 A，并对未通过门禁的子结构集中 Exact 回退.

        网络只调用一次；变分式在 FEALPy 后端上向量化计算。门禁仍按子结构
        独立判断，使 diagnostics 和回退范围保持局部性。返回的刚度与延拓落在
        构造时给定的迹空间上，``trace=None`` 时即完整接口。
        """
        if getattr(local_stiffness_batch, "ndim", 0) != 3:
            raise ValueError(
                "PIMLShapeReduction.reduce_many() 要求形状为 "
                "(n_substructure, n_dof, n_dof)."
            )
        self.legacy._check_local_stiffness(local_stiffness_batch)
        batch_size = len(local_stiffness_batch)
        if batch_size == 0:
            raise ValueError("PIMLShapeReduction.reduce_many() 不接受空批次.")
        if density_batch is not None and len(density_batch) != batch_size:
            raise ValueError("density_batch 与 local_stiffness_batch 的批量长度不一致.")
        if self.legacy.model is None:
            return self._exact_batch_fallback(
                local_stiffness_batch,
                density_batch,
                reason="model_missing",
            )
        if density_batch is None:
            return self._exact_batch_fallback(
                local_stiffness_batch,
                density_batch,
                reason="density_missing",
            )

        try:
            self.legacy.model.eval()
            with torch.no_grad():
                density_numpy = bm.to_numpy(density_batch)
                model_input = torch.tensor(
                    density_numpy.reshape(batch_size, -1), dtype=torch.float32
                )
                prediction_numpy = self.legacy.model(model_input).cpu().numpy()
        except SurrogateContractError:
            raise
        except Exception as error:
            return self._exact_batch_fallback(
                local_stiffness_batch,
                density_batch,
                reason="surrogate_inference_failed",
                metrics={"error": str(error)},
            )

        expected_shape = (batch_size, self.legacy.n_output)
        if tuple(prediction_numpy.shape) != expected_shape:
            raise SurrogateContractError(
                f"代理网络批量输出形状 {tuple(prediction_numpy.shape)} 与所需的 "
                f"{expected_shape} 不符"
            )

        prediction = bm.asarray(prediction_numpy, dtype=bm.float64)
        finite_rows = bm.to_numpy(
            bm.all(bm.isfinite(prediction), axis=1)
        ).astype(bool)
        reduced_shape = (
            batch_size,
            self.legacy.n_i,
            self.legacy.n_reduced,
        )
        deformation_component = bm.reshape(prediction, reduced_shape)
        recovery = self.legacy.assemble_recovery(deformation_component)

        # 切片与变分构造都走核心的批量实现, 迹投影由 legacy 的迹矩阵承担.
        K_ii, K_ib_t, K_bb_t = self.legacy._trace_blocks(local_stiffness_batch)
        stiffness = self.legacy._variational_stiffness(
            K_ii, K_ib_t, K_bb_t, recovery
        )

        diagnostics: list[ReductionDiagnostics] = []
        fallback_indices: list[int] = []
        for index in range(batch_size):
            if not finite_rows[index]:
                diagnostics.append(
                    self._fallback_diagnostics("nonfinite_prediction", {})
                )
                fallback_indices.append(index)
                continue

            self.legacy.gate_report = {}
            try:
                self.legacy._check_gates(stiffness[index], K_bb_t[index])
                diagnostics.append(
                    self._successful_diagnostics(dict(self.legacy.gate_report))
                )
            except Exception as error:
                gate_metrics = dict(self.legacy.gate_report)
                gate_metrics["error"] = str(error)
                diagnostics.append(
                    self._fallback_diagnostics("gate_rejected", gate_metrics)
                )
                fallback_indices.append(index)

        if fallback_indices:
            fallback_index = bm.asarray(fallback_indices, dtype=bm.int64)
            fallback_density = density_batch[fallback_index]
            exact_stiffness, exact_recovery = self.legacy._reduce_exact(
                self.legacy.fallback_solver.condense(
                    local_stiffness_batch[fallback_index], fallback_density
                )
            )
            stiffness = bm.set_at(stiffness, fallback_index, exact_stiffness)
            recovery = bm.set_at(recovery, fallback_index, exact_recovery)

        return LocalReductionBatchResult(
            stiffness=stiffness,
            recovery=recovery,
            diagnostics=tuple(diagnostics),
        )
