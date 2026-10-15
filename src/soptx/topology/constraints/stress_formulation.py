"""局部应力约束的松弛公式: 公式协议与 epsilon 松弛实现.

本模块只做约束代数, 不执行有限元状态计算与伴随求解. 不同离散方法通过
``stress_representation`` 声明求解器原生应力是实体应力还是表观应力. 约束对象层的
接口见 ``stress_protocol``.
"""

import math
from typing import Literal, Optional, Protocol, runtime_checkable

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike


StressRepresentation = Literal["solid", "apparent"]


@runtime_checkable
class StressRelaxationFormulation(Protocol):
    """可注入的局部应力松弛公式协议.

    新公式只需实现本协议, 无需继承任何内置松弛模型. 下列方法共用三个参数:

    - ``stress_ratio``: 分析器原生 von Mises 应力除以许用应力, 记为 s, 形状为
      (NC, NQ) 或多分辨率下的 (NC, n_sub, NQ), 最后一维 NQ 对应应力评价点;
    - ``stiffness_ratio``: 相对刚度 m_E, 形状为 (NC,) 或 (NC, n_sub);
    - ``stress_representation``: ``'solid'`` 表示 LFEM 恢复的实体应力,
      ``'apparent'`` 表示混合元原生的表观应力.

    Notes
    -----
    ``partial_wrt_stiffness_ratio`` 在原生应力 s 固定时求偏导; 应力随状态解变化的
    隐式项由有限元适配器处理.

    ``constraint_value`` 定义 AL 使用的约束 g <= 0, ``acceptance_violation`` 定义
    验收所用的超限量, ``stress_measure`` 仅用于展示, 三者不能互相替代.
    """

    name: str

    def constraint_value(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算约束值 g, 可行域约定为 g <= 0.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见类 docstring.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的约束值.
        """
        ...

    def partial_wrt_stiffness_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算原生应力固定时的显式偏导 ``partial g / partial m_E``.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见类 docstring.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的偏导.
        """
        ...

    def gradient_wrt_stress_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算对原生应力比的偏导 ``partial g / partial s``.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见类 docstring.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的偏导.
        """
        ...

    def stress_measure(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回用于展示的无量纲应力测度.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见类 docstring.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的应力测度.
        """
        ...

    def acceptance_violation(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回该公式定义的验收超限量.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见类 docstring.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的超限量, 非正表示满足验收判据.
        """
        ...

    def threshold(self, stiffness_ratio: TensorLike) -> Optional[TensorLike]:
        """返回局部松弛阈值.

        Parameters
        ----------
        stiffness_ratio : TensorLike
            相对刚度 m_E.

        Returns
        -------
        TensorLike or None
            与 ``stiffness_ratio`` 同形的阈值; 公式没有独立阈值时返回 None.
        """
        ...


class EpsilonRelaxedStressFormulation:
    """epsilon 松弛的表观应力约束.

    以表观应力比 a 为约束量, 阈值随相对刚度在 epsilon 与 1 之间线性过渡:

    .. math::

        t(m_E) = m_E + \\varepsilon (1 - m_E), \\qquad g = a - t(m_E) \\le 0,

    其中实体应力表示下 a = m_E s, 表观应力表示下 a = s. 两种表示只要表观应力
    相同, 给出的 g 就相同, 因而 LFEM 与混合元可以共用同一约束.

    Parameters
    ----------
    epsilon : float, optional
        松弛参数, 须有限且满足 0 <= epsilon <= 1, 默认 1e-4. 取 0 时阈值为 m_E,
        空单元的表观应力须几乎为零.

    Raises
    ------
    ValueError
        ``epsilon`` 非有限或不在 [0, 1] 内.
    """

    name = "apparent"

    def __init__(self, epsilon: float = 1e-4) -> None:
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon 必须有限且满足 0 <= epsilon <= 1")
        self.epsilon = epsilon

    def threshold(self, stiffness_ratio: TensorLike) -> TensorLike:
        """返回松弛阈值 t(m_E) = m_E + epsilon (1 - m_E).

        Parameters
        ----------
        stiffness_ratio : TensorLike
            相对刚度 m_E.

        Returns
        -------
        TensorLike
            与 ``stiffness_ratio`` 同形的阈值.
        """
        return stiffness_ratio + self.epsilon * (1.0 - stiffness_ratio)

    @staticmethod
    def _apparent_ratio(
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """把原生应力比换算为表观应力比 a; 表示未知时抛 ValueError."""
        if stress_representation == "solid":
            return stiffness_ratio[..., None] * stress_ratio
        if stress_representation == "apparent":
            return stress_ratio
        raise ValueError(f"未知应力表示: {stress_representation}")

    def constraint_value(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算约束值 g = a - t(m_E).

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见 ``StressRelaxationFormulation``.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的约束值.
        """
        apparent_ratio = self._apparent_ratio(
            stress_ratio,
            stiffness_ratio,
            stress_representation,
        )
        return apparent_ratio - self.threshold(stiffness_ratio)[..., None]

    def partial_wrt_stiffness_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算原生应力固定时的偏导 ``partial g / partial m_E``.

        实体应力表示下为 s - (1 - epsilon), 表观应力表示下为 -(1 - epsilon).

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见 ``StressRelaxationFormulation``.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的偏导.
        """
        threshold_derivative = 1.0 - self.epsilon
        if stress_representation == "solid":
            return stress_ratio - threshold_derivative
        if stress_representation == "apparent":
            return bm.ones_like(stress_ratio) * (-threshold_derivative)
        raise ValueError(f"未知应力表示: {stress_representation}")

    def gradient_wrt_stress_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算偏导 ``partial g / partial s``.

        实体应力表示下为 m_E, 表观应力表示下为 1.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见 ``StressRelaxationFormulation``.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的偏导.
        """
        if stress_representation == "solid":
            return bm.ones_like(stress_ratio) * stiffness_ratio[..., None]
        if stress_representation == "apparent":
            return bm.ones_like(stress_ratio)
        raise ValueError(f"未知应力表示: {stress_representation}")

    def stress_measure(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回表观应力比 a, 仅用于展示.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见 ``StressRelaxationFormulation``.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的表观应力比.
        """
        return self._apparent_ratio(
            stress_ratio,
            stiffness_ratio,
            stress_representation,
        )

    def acceptance_violation(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回验收超限量, 即 AL 实际施加的约束值 g.

        g 已以许用应力归一化 (适配器方法名与 summary 字段中的 "relative" 即指此),
        但没有再除以阈值 t(m_E); 验收时直接与容差 delta_g 比较.

        Parameters
        ----------
        stress_ratio, stiffness_ratio, stress_representation
            见 ``StressRelaxationFormulation``.

        Returns
        -------
        TensorLike
            与 ``stress_ratio`` 同形的超限量, 非正表示满足验收判据.
        """
        return self.constraint_value(
            stress_ratio,
            stiffness_ratio,
            stress_representation,
        )
