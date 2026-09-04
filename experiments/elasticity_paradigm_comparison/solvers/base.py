"""四范式求解路径的统一适配契约.

本模块只定义契约与共享上下文, 不含任何求解实现. 各适配器必须调用 ``src/soptx`` 的
公共接口, 不在本目录重写求解逻辑: 重写正是 ``examples/piml_substructure_elasticity/
compare_piml_pinn.py`` 已经发生的问题 (它自带一份 ``PINNElasticityNet``, 与
``examples/pinn_elasticity/minimal_demo.py`` 的同名实现重复).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ExperimentContext:
    """四条路径共享的算例上下文.

    由 Runner 构造一次并分发给全部适配器, 确保四条路径面对同一细网格, 同一密度场,
    同一外载与同一 Dirichlet 约束; 任何路径私自重建其中一项都会破坏受控比较.

    属性:
        problem: ``soptx.problems.elasticity`` 中的物理问题对象.
        assembler: ``GlobalAssembler``, 持有细网格, 子结构划分与材料.
        prototype: ``SubstructurePrototype``, 提供 ``rigid_basis`` 等结构信息.
        density: 按子结构组织的单元密度场, 形状 ``(M, NC)``.
        global_load: 施加 Dirichlet 条件之前的全局外载向量, 形状 ``(n_dof,)``.
        fixed_mask: 全局 Dirichlet 自由度布尔掩码, 形状 ``(n_dof,)``.
        interface_dofs: 全局接口自由度索引, 形状 ``(n_interface,)``.
        fine_mesh_shape: 全局细网格各方向单元数.
    """

    problem: Any
    assembler: Any
    prototype: Any
    density: Any
    global_load: Any
    fixed_mask: Any
    interface_dofs: Any
    fine_mesh_shape: tuple[int, ...]


@dataclass
class SolveResult:
    """单条求解路径的产出.

    全部误差度量由 ``metrics.py`` 在 Runner 中统一施加, 适配器只负责把位移场映射到
    参考细网格的自由度序并如实报告计时, 不自带误差定义.

    属性:
        method: 求解路径登记名.
        u_full: 全场位移向量, 按参考细网格自由度序排列, 形状 ``(n_dof,)``.
        offline_seconds: 一次性离线阶段耗时; 无离线阶段的路径取 ``0.0``.
        online_seconds: 求解单个边值问题的耗时.
        condensed_stiffness: 批量缩聚刚度矩阵, 形状 ``(M, n_b, n_b)``;
            非缩聚类路径取 ``None``.
        diagnostics: 路径专属诊断项, 如 PIML 的回退次数与 PINN 的末轮 loss.
    """

    method: str
    u_full: Any
    offline_seconds: float
    online_seconds: float
    condensed_stiffness: Optional[Any] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ParadigmSolver(ABC):
    """四范式求解路径适配器基类.

    子类以类属性登记自身在 2x2 定位表中的位置与就绪状态, Runner 据此分组报告并在未就绪
    时给出明确阻塞原因, 而不是让实验静默产出不可用结果.

    属性:
        NAME: 求解路径登记名, 必须与 ``config.METHODS`` 一致.
        CARRIER: 计算载体, 取 ``global_field`` 或 ``condensed_operator``.
        NATURE: 求解性质, 取 ``exact`` 或 ``surrogate``.
        READY: 该适配器是否已可运行.
        BLOCKER: ``READY`` 为 ``False`` 时的阻塞原因; 就绪时为 ``None``.
    """

    NAME: str
    CARRIER: str
    NATURE: str
    READY: bool = False
    BLOCKER: Optional[str] = None

    @abstractmethod
    def solve(
        self,
        context: ExperimentContext,
        case: dict[str, Any],
        protocol: dict[str, Any],
    ) -> SolveResult:
        """在给定上下文上执行一次求解.

        参数:
            context: 四条路径共享的算例上下文.
            case: 已校验的单个算例配置.
            protocol: 已校验的共享冻结项.

        返回:
            本路径的求解产出.

        异常:
            NotImplementedError: 当适配器尚未就绪时抛出, 消息中给出阻塞原因.
        """

    def ensure_ready(self) -> None:
        """在求解前检查适配器就绪状态.

        异常:
            NotImplementedError: 当 ``READY`` 为 ``False`` 时抛出.
        """
        if not self.READY:
            raise NotImplementedError(f"求解路径 {self.NAME} 尚未就绪: {self.BLOCKER}")
