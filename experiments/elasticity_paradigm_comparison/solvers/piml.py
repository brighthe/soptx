"""PIML 子结构缩聚适配器: 2x2 定位表的 (缩聚算子, 代理) 象限."""

from __future__ import annotations

from typing import Any

from .base import ExperimentContext, ParadigmSolver, SolveResult


class PIMLSolver(ParadigmSolver):
    """PIML 代理缩聚路径.

    实现时调用 ``soptx.fem.substructure`` 的 ``PIMLSurrogateNet`` 与
    ``PIMLStaticCondensation``: 离线阶段在随机密度快照上训练 ``rho -> K_s`` (路线 B)
    或 ``rho -> N`` (路线 A) 的代理, 在线阶段推理后经特征值门禁与精确回退进入全局
    接口装配, 其余环节与 ``substructure`` 路径完全一致.

    离线训练耗时计入 ``offline_seconds``, 在线推理与接口求解计入 ``online_seconds``;
    两者的分离是摊销盈亏点 ``metrics.breakeven_count`` 的输入.

    本路径需额外返回 ``condensed_stiffness`` 与回退统计, 供
    ``metrics.min_eigenvalue``, ``metrics.rigid_mode_residual`` 评价结构保持;
    刚体模态基取自 ``context.prototype.rigid_basis``, 不在本目录重建.
    """

    NAME = "piml"
    CARRIER = "condensed_operator"
    NATURE = "surrogate"
    READY = False
    BLOCKER = (
        "训练循环待下沉: 预测器 PIMLSurrogateNet/PIMLStaticCondensation 已在 "
        "soptx.fem.substructure 中, 但数据采样与训练循环目前只存在于 "
        "examples/piml_substructure_elasticity/verify_stiffness_route.py 脚本内部"
    )

    def solve(
        self,
        context: ExperimentContext,
        case: dict[str, Any],
        protocol: dict[str, Any],
    ) -> SolveResult:
        """训练并应用 PIML 代理缩聚, 返回全场位移与结构保持诊断."""
        self.ensure_ready()
        raise NotImplementedError(self.BLOCKER)
