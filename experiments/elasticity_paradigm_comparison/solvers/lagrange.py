"""Lagrange 位移有限元适配器: 2x2 定位表的 (全局解场, 精确) 象限."""

from __future__ import annotations

from typing import Any

from .base import ExperimentContext, ParadigmSolver, SolveResult


class LagrangeSolver(ParadigmSolver):
    """全装配 Lagrange 位移有限元路径, 同时充当四条路径的参考真解.

    实现时直接调用 ``soptx.fem.analyzers.LagrangeFEMAnalyzer``: 在 ``context.assembler``
    的细网格上装配全局刚度, 施加 ``context.fixed_mask`` 后求解 ``K U = F``. 本路径无
    离线阶段, ``offline_seconds`` 恒为 ``0.0``.

    其产出的全局刚度矩阵还是 ``metrics.energy_norm_relative_error`` 的输入, 需一并
    经 ``diagnostics['global_stiffness']`` 返回.

    本路径的离散正确性与载荷装配正确性已由 ``examples/lagrange_elasticity`` 建立
    (制造解的 L2 观测收敛阶与真相对残差, 以及集中力算例的载荷等效性), 本目录不重复
    验证; 适配器只负责在本实验的冻结算例上产出参考解.
    """

    NAME = "lagrange"
    CARRIER = "global_field"
    NATURE = "exact"
    READY = False
    BLOCKER = (
        "待接线, 无能力缺口: LagrangeFEMAnalyzer 已在 soptx.fem.analyzers 中, "
        "正确性已由 examples/lagrange_elasticity 建立, "
        "调用范式见 examples/piml_substructure_elasticity/verify_stiffness_route.py"
    )

    def solve(
        self,
        context: ExperimentContext,
        case: dict[str, Any],
        protocol: dict[str, Any],
    ) -> SolveResult:
        """求解全装配线弹性方程并返回参考位移场."""
        self.ensure_ready()
        raise NotImplementedError(self.BLOCKER)
