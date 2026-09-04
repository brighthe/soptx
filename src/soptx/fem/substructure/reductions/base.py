"""子结构局部缩聚的无状态结果契约与旧实现适配器."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from fealpy.backend import backend_manager as bm


@dataclass(frozen=True)
class ReductionDiagnostics:
    """一次局部缩聚中算子、恢复关系及回退行为的来源快照."""

    requested_method: str
    stiffness_source: str
    recovery_source: str
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结门禁指标快照，避免后续旧对象状态污染既有结果."""
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class LocalReductionResult:
    """一个子结构在完整接口上的局部缩聚结果.

    ``stiffness`` 和 ``recovery`` 尚未施加 ``TraceBasis``. 接口迹投影属于
    后续分析阶段，不属于 ``LocalReduction`` 的职责。
    """

    stiffness: Any
    recovery: Optional[Any]
    diagnostics: ReductionDiagnostics

    @property
    def used_fallback(self) -> bool:
        """兼容常用判断，权威信息仍位于 ``diagnostics``."""
        return self.diagnostics.used_fallback

    def recover(self, boundary_displacement: Any) -> Any:
        """仅使用本结果快照恢复内部位移，不读取 reduction 的可变状态."""
        if self.recovery is None:
            raise RuntimeError("当前局部缩聚结果不包含内部位移恢复矩阵.")
        return bm.einsum(
            "...ij,...j->...i",
            self.recovery,
            bm.asarray(boundary_displacement),
        )


@dataclass(frozen=True)
class LocalReductionBatchResult:
    """批量局部缩聚的堆叠数值结果及逐子结构诊断.

    数值张量保留 batch 维，避免先拆分再 ``stack`` 破坏 Exact Schur 的向量化
    路径；索引或迭代时才构造轻量的单子结构结果视图。
    """

    stiffness: Any
    recovery: Optional[Any]
    diagnostics: tuple[ReductionDiagnostics, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.stiffness)
        if len(self.diagnostics) != batch_size:
            raise ValueError("diagnostics 数量必须与 stiffness 的 batch 长度一致.")
        if self.recovery is not None and len(self.recovery) != batch_size:
            raise ValueError("recovery 的 batch 长度必须与 stiffness 一致.")

    def __len__(self) -> int:
        return len(self.stiffness)

    def __getitem__(self, index: int) -> LocalReductionResult:
        recovery = None if self.recovery is None else self.recovery[index]
        return LocalReductionResult(
            stiffness=self.stiffness[index],
            recovery=recovery,
            diagnostics=self.diagnostics[index],
        )

    def __iter__(self) -> Iterator[LocalReductionResult]:
        for index in range(len(self)):
            yield self[index]

    def recover(self, boundary_displacement: Any) -> Any:
        """使用堆叠恢复矩阵批量恢复内部位移."""
        if self.recovery is None:
            raise RuntimeError("当前批量缩聚结果不包含内部位移恢复矩阵.")
        return bm.einsum(
            "...ij,...j->...i",
            self.recovery,
            bm.asarray(boundary_displacement),
        )


@runtime_checkable
class LocalReduction(Protocol):
    """精确与代理局部缩聚策略共同遵守的结构化接口."""

    def reduce(
        self,
        local_stiffness: Any,
        density: Optional[Any] = None,
    ) -> LocalReductionResult:
        """缩聚一个局部子结构；不接受批量前导维."""
        ...

    def reduce_many(
        self,
        local_stiffness_batch: Any,
        density_batch: Optional[Any] = None,
    ) -> LocalReductionBatchResult:
        """逐项缩聚一批局部子结构并保留逐子结构诊断."""
        ...


class CondensationReductionAdapter:
    """把旧 ``condense`` 对象适配为无状态结果契约.

    适配方向固定为 ``reduce -> legacy.condense``。旧类在迁移期间保持不变，
    从而避免 ``reduce`` 与 ``condense`` 互相调用产生递归，也不改变依赖
    ``K_s``、``N`` 和旧 ``recover`` 的现有装配代码。
    """

    def __init__(
        self,
        legacy: Any,
        *,
        requested_method: str,
        stiffness_source: str,
        recovery_source: str,
    ) -> None:
        self.legacy = legacy
        self.requested_method = requested_method
        self.stiffness_source = stiffness_source
        self.recovery_source = recovery_source

    @staticmethod
    def _check_single(local_stiffness: Any) -> None:
        if getattr(local_stiffness, "ndim", 0) != 2:
            raise ValueError(
                "LocalReduction.reduce() 只接受单个二维局部刚度矩阵; "
                "批量输入请调用 reduce_many()."
            )

    def _fallback_reason(self, density: Optional[Any]) -> Optional[str]:
        if not bool(getattr(self.legacy, "used_fallback", False)):
            return None
        if getattr(self.legacy, "model", object()) is None:
            return "model_missing"
        if density is None:
            return "density_missing"
        return "surrogate_failed_or_gate_rejected"

    def _diagnostics(self, density: Optional[Any]) -> ReductionDiagnostics:
        used_fallback = bool(getattr(self.legacy, "used_fallback", False))
        if used_fallback:
            stiffness_source = "exact_schur"
            recovery_source = "exact_schur"
        else:
            stiffness_source = self.stiffness_source
            recovery_source = self.recovery_source

        fallback_reason = self._fallback_reason(density)
        gate_report = getattr(self.legacy, "gate_report", None)
        metrics = (
            dict(gate_report)
            if gate_report and fallback_reason not in {"model_missing", "density_missing"}
            else {}
        )
        return ReductionDiagnostics(
            requested_method=self.requested_method,
            stiffness_source=stiffness_source,
            recovery_source=recovery_source,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            metrics=metrics,
        )

    def reduce(
        self,
        local_stiffness: Any,
        density: Optional[Any] = None,
    ) -> LocalReductionResult:
        """调用旧缩聚器恰好一次，并立即保存独立的结果与诊断引用."""
        self._check_single(local_stiffness)
        stiffness, recovery = self.legacy.condense(local_stiffness, density)
        return LocalReductionResult(
            stiffness=stiffness,
            recovery=recovery,
            diagnostics=self._diagnostics(density),
        )

    def reduce_many(
        self,
        local_stiffness_batch: Any,
        density_batch: Optional[Any] = None,
    ) -> LocalReductionBatchResult:
        """缺省沿第一维逐项缩聚，再形成统一堆叠结果."""
        if getattr(local_stiffness_batch, "ndim", 0) != 3:
            raise ValueError(
                "LocalReduction.reduce_many() 要求形状为 "
                "(n_substructure, n_dof, n_dof)."
            )
        if (
            density_batch is not None
            and len(density_batch) != len(local_stiffness_batch)
        ):
            raise ValueError("density_batch 与 local_stiffness_batch 的批量长度不一致.")

        results: list[LocalReductionResult] = []
        for index in range(len(local_stiffness_batch)):
            density = None if density_batch is None else density_batch[index]
            results.append(self.reduce(local_stiffness_batch[index], density))

        stiffness = bm.stack([result.stiffness for result in results], axis=0)
        recoveries = [result.recovery for result in results]
        if all(recovery is None for recovery in recoveries):
            recovery_batch = None
        elif any(recovery is None for recovery in recoveries):
            raise RuntimeError("同一批次不能混合包含和不包含 recovery 的结果.")
        else:
            recovery_batch = bm.stack(recoveries, axis=0)

        return LocalReductionBatchResult(
            stiffness=stiffness,
            recovery=recovery_batch,
            diagnostics=tuple(result.diagnostics for result in results),
        )
