"""PINN 强形式适配器: 2x2 定位表的 (全局解场, 代理) 象限."""

from __future__ import annotations

from typing import Any

from .base import ExperimentContext, ParadigmSolver, SolveResult


class PINNSolver(ParadigmSolver):
    """PINN 强形式求解路径.

    实现时以配点采样与自动微分构造线弹性强形式残差, 训练完成后在 ``context`` 的细网格
    节点上采值, 映射到参考自由度序输出. 本路径无可复用的离线阶段: 边界条件或载荷一变
    就必须重训, 因此 ``offline_seconds`` 取 ``0.0``, 单次训练耗时全部计入
    ``online_seconds``, 这正是摊销曲线上其斜率不下降的原因.

    边界条件按 ``protocol['pinn_boundary_mode'] == 'hard'`` 以
    ``u = u_bar + B(x) * net(x)`` 硬施加, 不引入边界残差权重: 软约束下权重调参足以
    改变一个数量级的误差, 会把范式对比退化成调参对比.
    """

    NAME = "pinn"
    CARRIER = "global_field"
    NATURE = "surrogate"
    READY = False
    BLOCKER = (
        "求解器待下沉: PINNElasticityNet 与残差构造目前只存在于 "
        "examples/pinn_elasticity/minimal_demo.py, 且在 "
        "experiments/elasticity_paradigm_comparison/legacy/compare_piml_pinn.py 中被重复实现一次; "
        "src/soptx/ml/ 当前仅有 networks.py. 需先下沉到 src/soptx/ml/ 并消除重复"
    )

    def solve(
        self,
        context: ExperimentContext,
        case: dict[str, Any],
        protocol: dict[str, Any],
    ) -> SolveResult:
        """训练 PINN 并在参考细网格节点上采样输出位移场."""
        self.ensure_ready()
        raise NotImplementedError(self.BLOCKER)
