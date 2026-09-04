"""精确子结构静力缩聚适配器: 2x2 定位表的 (缩聚算子, 精确) 象限."""

from __future__ import annotations

from typing import Any

from .base import ExperimentContext, ParadigmSolver, SolveResult


class SubstructureSolver(ParadigmSolver):
    """精确 Schur 补静力缩聚路径.

    实现时调用 ``soptx.fem.substructure`` 的 ``FEAStaticCondensation`` 求局部
    ``K_s`` 与恢复矩阵 ``N``, 经 ``GlobalAssembler`` 装配全局接口系统, 由
    ``solve_interface_system`` 求解接口位移后回代恢复内部位移.

    本路径与 ``lagrange`` 在代数上完全等价, 因此在对比中不承担精度竞争角色, 而是
    两项职责: 其一, 与参考解的偏差应停留在机器精度量级, 构成实现正确性自检;
    其二, 为 ``piml`` 提供同载体的精确基线, 使代理误差可被单独分离出来.
    """

    NAME = "substructure"
    CARRIER = "condensed_operator"
    NATURE = "exact"
    READY = False
    BLOCKER = (
        "待接线, 无能力缺口: FEAStaticCondensation, GlobalAssembler 与 "
        "solve_interface_system 均已在 soptx.fem.substructure 中"
    )

    def solve(
        self,
        context: ExperimentContext,
        case: dict[str, Any],
        protocol: dict[str, Any],
    ) -> SolveResult:
        """执行精确静力缩聚求解并返回全场位移与批量缩聚刚度."""
        self.ensure_ready()
        raise NotImplementedError(self.BLOCKER)
