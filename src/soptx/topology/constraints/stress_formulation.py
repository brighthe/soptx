"""局部应力约束的松弛公式与公共协议.

本模块定义约束代数与适配器协议, 不执行有限元状态计算与伴随求解. 不同离散方法通过
``stress_representation`` 声明求解器原生应力是实体应力还是表观应力.
"""

import math
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


StressRepresentation = Literal["solid", "apparent"]


@runtime_checkable
class StressRelaxationFormulation(Protocol):
    """可注入的局部应力松弛公式协议.

    Notes
    -----
    stress_ratio 为分析器原生 von Mises 应力除以 stress_limit, 形状为
    (NC, NQ) 或 (NC, n_sub, NQ). stiffness_ratio 的形状为 (NC,) 或
    (NC, n_sub), 最后一维 NQ 始终对应应力评价点.

    solid 表示 LFEM 恢复的实体应力, apparent 表示混合元原生表观应力.
    partial_wrt_stiffness_ratio 必须在原生应力变量固定时求偏导;
    应力随状态解变化的隐式项由有限元适配器处理.

    constraint_value 定义 AL 使用的 g <= 0. acceptance_violation 定义
    模型的验收尺度, stress_measure 仅用于展示, 三者不能互相替代.
    新公式只需实现本协议, 无需继承任何内置松弛模型.
    """

    name: str

    def constraint_value(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算约束值 ``g``,可行域约定为 ``g <= 0``."""

    def partial_wrt_stiffness_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算固定求解器状态时 ``partial g / partial m_E``."""

    def gradient_wrt_stress_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """计算 ``partial g / partial (sigma_vm / sigma_lim)``."""

    def stress_measure(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回用于展示的无量纲应力测度."""

    def acceptance_violation(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回该公式定义的验收超限量, 非正表示满足该模型的验收判据."""

    def threshold(self, stiffness_ratio: TensorLike) -> Optional[TensorLike]:
        """返回局部松弛阈值; 无独立阈值时返回 ``None``."""


class PolynomialVanishingStressFormulation:
    """多项式消失约束 ``g = m_E * (s**3 + s)``."""

    name = "vanishing"

    @staticmethod
    def _require_solid(stress_representation: StressRepresentation) -> None:
        if stress_representation != "solid":
            raise ValueError("多项式消失约束只支持实体应力表示")

    def constraint_value(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        self._require_solid(stress_representation)
        deviation = stress_ratio - 1.0
        return stiffness_ratio[..., None] * (deviation**3 + deviation)

    def partial_wrt_stiffness_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        self._require_solid(stress_representation)
        deviation = stress_ratio - 1.0
        return deviation**3 + deviation

    def gradient_wrt_stress_ratio(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        self._require_solid(stress_representation)
        deviation = stress_ratio - 1.0
        return stiffness_ratio[..., None] * (3.0 * deviation**2 + 1.0)

    def stress_measure(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        self._require_solid(stress_representation)
        return stiffness_ratio[..., None] * stress_ratio

    def acceptance_violation(
        self,
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
        """返回历史加权验收量 ``m_E * sigma_vm_solid / sigma_lim - 1``.

        该量沿用既有优化验收口径, 但不等于多项式约束值 ``g``.
        """
        self._require_solid(stress_representation)
        return stiffness_ratio[..., None] * stress_ratio - 1.0

    def threshold(self, stiffness_ratio: TensorLike) -> None:
        return None


class EpsilonRelaxedStressFormulation:
    """统一表观应力的 ``epsilon`` 松弛约束."""

    name = "apparent"

    def __init__(self, epsilon: float = 1e-4) -> None:
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon 必须有限且满足 0 <= epsilon <= 1")
        self.epsilon = epsilon

    def threshold(self, stiffness_ratio: TensorLike) -> TensorLike:
        return stiffness_ratio + self.epsilon * (1.0 - stiffness_ratio)

    @staticmethod
    def _apparent_ratio(
        stress_ratio: TensorLike,
        stiffness_ratio: TensorLike,
        stress_representation: StressRepresentation,
    ) -> TensorLike:
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
        """返回 AL 实际施加的绝对超限量 ``g``."""
        return self.constraint_value(
            stress_ratio,
            stiffness_ratio,
            stress_representation,
        )


class StressConstraintProtocol(Protocol):
    """增广拉格朗日目标函数所需的最小约束接口."""

    @property
    def analyzer(self) -> Any:
        """返回负责正向与伴随求解的分析器."""

    @property
    def formulation(self) -> StressRelaxationFormulation:
        """返回当前松弛公式."""

    @property
    def discretization_name(self) -> str:
        """返回用户可读的离散方法名称."""

    def fun(
        self,
        density: TensorLike,
        state: Optional[Dict] = None,
        **kwargs,
    ) -> TensorLike:
        """计算约束值."""

    def compute_gradient_wrt_von_mises(self, state: Dict) -> TensorLike:
        """计算约束对 von Mises 应力的偏导数."""

    def compute_partial_gradient_wrt_mE(self, state: Dict) -> TensorLike:
        """计算固定求解器状态时约束对相对刚度的显式偏导数."""

    def compute_adjoint_load(
        self,
        dPenaldVM: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """组装伴随载荷."""

    def compute_implicit_sensitivity_term(
        self,
        adjoint_vector: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """计算关于相对刚度的隐式灵敏度项."""

    def compute_stress_measure(
        self,
        rho: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """返回用于展示的无量纲应力测度."""

    def compute_relative_violation(
        self,
        rho: TensorLike,
        state: Dict,
    ) -> TensorLike:
        """返回该模型定义的验收超限量."""
