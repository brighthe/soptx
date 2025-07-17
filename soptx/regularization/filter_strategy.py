import math
from abc import ABC, abstractmethod
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.sparse import CSRTensor

class _FilterStrategy(ABC):
    """过滤方法的抽象基类 (内部使用)"""
    @abstractmethod
    def get_initial_density(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def filter_variablies(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def filter_objective_sensitivities(self, 
                                    rho_Phys: TensorLike, 
                                    obj_grad: TensorLike
                                ) -> TensorLike:
        pass

    @abstractmethod
    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        pass


class NoneStrategy(_FilterStrategy):
    """ '无操作' 策略, 当不需要过滤时使用"""
    def get_initial_density(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)

        return rho_Phys

    def filter_variables(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)

        return rho_Phys

    def filter_objective_sensitivities(self, 
                                       rho_Phys: TensorLike, 
                                       obj_grad: TensorLike
                                    ) -> TensorLike:
        return obj_grad

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        return dcons

class SensitivityStrategy(_FilterStrategy):
    """灵敏度过滤策略"""
    def __init__(self, H: CSRTensor, Hs: TensorLike, cell_measure: TensorLike) -> None:
        self._H = H
        self._Hs = Hs
        self._cell_measure = cell_measure

    def get_initial_density(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)
        
        return rho_Phys

    def filter_variables(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)

        return rho_Phys

    def filter_objective_sensitivities(self, 
                                    rho_Phys: TensorLike, 
                                    obj_grad: TensorLike
                                ) -> TensorLike:

        weighted_obj_grad = rho_Phys * obj_grad / self._cell_measure
        numerator = self._H.matmul(weighted_obj_grad)

        epsilon = 1e-3
        stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), rho_Phys)
        denominator = (stability_factor / self._cell_measure) * self._Hs

        obj_grad = bm.set_at(obj_grad, slice(None), numerator / denominator)

        return obj_grad

    def filter_constraint_sensitivities(self, 
                                    rho_Phys: TensorLike, 
                                    cons_grad: TensorLike
                                ) -> TensorLike:

        return cons_grad


class DensityStrategy(_FilterStrategy):
    """密度过滤策略"""
    def __init__(self, H: CSRTensor, cell_measure: TensorLike):
        self._H = H
        self._cell_measure = cell_measure

    def get_initial_density(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)
        
        return rho_Phys

    def filter_variables(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        weighted_rho = rho * self._cell_measure
        numerator = self._H.matmul(weighted_rho)
        
        denominator = self._H.matmul(self._cell_measure)

        rho_Phys = bm.set_at(rho_Phys, slice(None), numerator / denominator)

        return rho_Phys

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        weighted_dobj = self._cell_measure * dobj
        return self._H.matmul(weighted_dobj / self._normalize_factor)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        weighted_dcons = self._cell_measure * dcons
        return self._H.matmul(weighted_dcons / self._normalize_factor)

# SensitivityStrategy 和 HeavisideStrategy 的实现类似
# ... 这里为了简洁省略，完整代码将在下面统一给出 ...

class HeavisideDensityStrategy(_FilterStrategy):
    """Heaviside 密度过滤策略"""
    def __init__(self, H: CSRTensor, cell_measure: TensorLike, normalize_factor: TensorLike,
                 beta: float = 1.0, max_beta: float = 512, continuation_iter: int = 50):
        self._H = H
        self._cell_measure = cell_measure
        self._normalize_factor = normalize_factor
        
        self.beta = beta
        self.max_beta = max_beta
        self.continuation_iter = continuation_iter
        self._xTilde = None
        self._beta_iter = 0

    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
        self._xTilde = x 
        projected_values = (1 - bm.exp(-self.beta * self._xTilde) + 
                    self._xTilde * math.exp(-self.beta)) + 2
        
        xPhys = bm.set_at(xPhys, slice(None), projected_values)

        return xPhys


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