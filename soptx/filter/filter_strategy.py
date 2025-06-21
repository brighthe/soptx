from abc import ABC, abstractmethod
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.sparse import CSRTensor

class _FilterStrategy(ABC):
    """过滤方法的抽象基类 (内部使用)"""

    @abstractmethod
    def get_initial_density(self, x: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def filter_variables(self, x: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        pass

    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """默认不执行任何操作"""
        return change, False


class NoneStrategy(_FilterStrategy):
    """ '无操作' 策略, 当不需要过滤时使用"""
    def get_initial_density(self, x: TensorLike) -> TensorLike:
        return x

    def filter_variables(self, x: TensorLike) -> TensorLike:
        return x

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        return dobj

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        return dcons

class SensitivityStrategy(_FilterStrategy):
    """灵敏度过滤策略"""
    def __init__(self, H: CSRTensor, cell_measure: TensorLike, normalize_factor: TensorLike):
        self._H = H
        self._cell_measure = cell_measure
        self._normalize_factor = normalize_factor

    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
        xPhys = bm.set_at(xPhys, slice(None), x)
        return xPhys

    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        xPhys = bm.set_at(xPhys, slice(None), x)
        return xPhys

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        # 计算密度加权的目标函数灵敏度
        weighted_dobj = bm.einsum('c, c -> c', xPhys, dobj)
        # 应用滤波矩阵
        filtered_dobj = self._H.matmul(weighted_dobj)
        # 计算修正因子
        correction_factor = self._Hs * bm.maximum(bm.tensor(0.001, dtype=bm.float64), xPhys)
        # 过滤后的目标函数灵敏度
        # dobj[:] = filtered_dobj / correction_factor
        dobj = bm.set_at(dobj, slice(None), filtered_dobj / correction_factor)
        return dobj

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        return dcons


class DensityStrategy(_FilterStrategy):
    """密度滤波策略"""
    def __init__(self, H: CSRTensor, cell_measure: TensorLike, normalize_factor: TensorLike):
        self._H = H
        self._cell_measure = cell_measure
        self._normalize_factor = normalize_factor

    def get_initial_density(self, x: TensorLike) -> TensorLike:
        return x

    def filter_variables(self, x: TensorLike) -> TensorLike:
        weighted_x = x * self._cell_measure
        filtered_x = self._H.matmul(weighted_x)
        return filtered_x / self._normalize_factor

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        weighted_dobj = self._cell_measure * dobj
        return self._H.matmul(weighted_dobj / self._normalize_factor)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        weighted_dcons = self._cell_measure * dcons
        return self._H.matmul(weighted_dcons / self._normalize_factor)

# SensitivityStrategy 和 HeavisideStrategy 的实现类似
# ... 这里为了简洁省略，完整代码将在下面统一给出 ...

class HeavisideStrategy(_FilterStrategy):
    """Heaviside 投影滤波策略"""
    def __init__(self, H: CSRTensor, cell_measure: TensorLike, normalize_factor: TensorLike,
                 beta: float, max_beta: float, continuation_iter: int):
        self._H = H
        self._cell_measure = cell_measure
        self._normalize_factor = normalize_factor
        
        self.beta = beta
        self.max_beta = max_beta
        self.continuation_iter = continuation_iter
        self._xTilde = None
        self._beta_iter = 0

    def _apply_projection(self, x_tilde: TensorLike) -> TensorLike:
        """应用 Heaviside 投影"""
        self._xTilde = x_tilde
        return (bm.tanh(self.beta / 2) + bm.tanh(self.beta * (x_tilde - 0.5))) / (2 * bm.tanh(self.beta / 2))

    def get_initial_density(self, x: TensorLike) -> TensorLike:
        return self._apply_projection(x)

    def filter_variables(self, x: TensorLike) -> TensorLike:
        weighted_x = self._cell_measure * x
        filtered_x = self._H.matmul(weighted_x)
        x_tilde = filtered_x / self._normalize_factor
        return self._apply_projection(x_tilde)

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        # Heaviside 投影的导数
        projection_deriv = self.beta * (1 - bm.tanh(self.beta * (self._xTilde - 0.5))**2) / (2 * bm.tanh(self.beta/2))
        weighted_dobj = dobj * projection_deriv * self._cell_measure
        return self._H.matmul(weighted_dobj / self._normalize_factor)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        projection_deriv = self.beta * (1 - bm.tanh(self.beta * (self._xTilde - 0.5))**2) / (2 * bm.tanh(self.beta/2))
        weighted_dcons = dcons * projection_deriv * self._cell_measure
        return self._H.matmul(weighted_dcons / self._normalize_factor)

    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """执行 beta continuation"""
        self._beta_iter += 1
        if self.beta < self.max_beta and (self._beta_iter >= self.continuation_iter or change <= 0.01):
            self.beta *= 2
            self._beta_iter = 0
            print(f"Beta increased to {self.beta}")
            return 1.0, True
        return change, False