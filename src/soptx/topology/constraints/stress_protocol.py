"""局部应力约束对象的接口协议.

增广拉格朗日目标与 AL-MMA 优化器只通过本协议使用应力约束对象, 不依赖具体离散方法;
``LagrangeStressConstraint`` 与 ``HuZhangStressConstraint`` 均满足本协议,
``tests/unit/test_stress_constraint_protocols.py`` 负责检查这项约束. 约束值的代数由
注入的松弛公式定义, 见 ``stress_formulation``.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from soptx.typing import TensorLike

from .stress_formulation import StressRelaxationFormulation


@runtime_checkable
class StressConstraintProtocol(Protocol):
    """增广拉格朗日目标与 AL-MMA 优化器所需的最小应力约束接口.

    ``state`` 是约束对象与分析器共用的求解状态缓存字典: ``fun`` 首次调用时填充应力、
    相对刚度等中间量, 其余方法读取同一份缓存.

    Notes
    -----
    ``runtime_checkable`` 只检查成员是否存在, 不检查方法签名; 签名一致性由测试保证.
    """

    @property
    def analyzer(self) -> Any:
        """负责正向与伴随求解的分析器."""
        ...

    @property
    def formulation(self) -> StressRelaxationFormulation:
        """当前注入的松弛公式."""
        ...

    @property
    def discretization_name(self) -> str:
        """用户可读的离散方法名称."""
        ...

    def fun(
        self,
        density: TensorLike,
        state: Optional[Dict] = None,
        **kwargs,
    ) -> TensorLike:
        """计算局部约束值 g, 可行域约定为 g <= 0.

        Parameters
        ----------
        density : TensorLike
            物理密度场.
        state : dict, optional
            求解状态缓存; 为 None 时新建并就地填充.
        **kwargs
            传给具体实现的附加参数.

        Returns
        -------
        TensorLike
            形状为 (NC, NQ) 或 (NC, n_sub, NQ) 的约束值, 豁免单元取严格可行的常值.
        """
        ...

    def compute_gradient_wrt_von_mises(self, state: Dict) -> TensorLike:
        """计算约束对 von Mises 应力的偏导数.

        Parameters
        ----------
        state : dict
            已由 ``fun`` 填充的求解状态缓存.

        Returns
        -------
        TensorLike
            与约束值同形的偏导数.
        """
        ...

    def compute_partial_gradient_wrt_mE(self, state: Dict) -> TensorLike:
        """计算求解器状态固定时约束对相对刚度的显式偏导数.

        Parameters
        ----------
        state : dict
            已由 ``fun`` 填充的求解状态缓存.

        Returns
        -------
        TensorLike
            与约束值同形的偏导数, 豁免单元为 0.
        """
        ...

    def compute_adjoint_load(
        self,
        dPenaldVM: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """组装伴随载荷.

        Parameters
        ----------
        dPenaldVM : TensorLike
            AL 罚项对 von Mises 应力的偏导数, 与约束值同形.
        state : dict
            已由 ``fun`` 填充的求解状态缓存.

        Returns
        -------
        TensorLike
            伴随方程的右端项.
        """
        ...

    def compute_implicit_sensitivity_term(
        self,
        adjoint_vector: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """计算关于相对刚度的隐式灵敏度项.

        Parameters
        ----------
        adjoint_vector : TensorLike
            伴随方程的解.
        state : dict
            已由 ``fun`` 填充的求解状态缓存.

        Returns
        -------
        TensorLike
            逐单元的隐式灵敏度项.
        """
        ...

    def compute_stress_measure(
        self,
        rho: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """返回用于展示的无量纲应力测度.

        Parameters
        ----------
        rho : TensorLike
            物理密度场.
        state : dict
            已由 ``fun`` 填充的求解状态缓存.

        Returns
        -------
        TensorLike
            与约束值同形的应力测度.
        """
        ...

    def compute_relative_violation(
        self,
        rho: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """返回注入公式定义的验收超限量.

        Parameters
        ----------
        rho : TensorLike
            物理密度场.
        state : dict
            已由 ``fun`` 填充的求解状态缓存.

        Returns
        -------
        TensorLike
            与约束值同形的超限量, 非正表示满足验收判据; 豁免单元取严格可行的常值.
        """
        ...
