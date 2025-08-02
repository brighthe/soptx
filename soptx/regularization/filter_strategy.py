import math
from abc import ABC, abstractmethod
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import Function
from fealpy.typing import TensorLike
from fealpy.sparse import CSRTensor

class _FilterStrategy(ABC):
    """过滤方法的抽象基类 (内部使用)"""
    @abstractmethod
    def get_initial_density(self, rho: Function, rho_Phys: Function) -> Function:
        pass

    @abstractmethod
    def filter_variables(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def filter_objective_sensitivities(self, 
                                    rho_Phys: TensorLike, 
                                    obj_grad: TensorLike
                                    ) -> TensorLike:
        pass

    @abstractmethod
    def filter_constraint_sensitivities(self, 
                                        rho_Phys: TensorLike, 
                                        con_grad: TensorLike
                                    ) -> TensorLike:
        pass


class NoneStrategy(_FilterStrategy):
    """ '无操作' 策略, 当不需要过滤时使用"""
    def get_initial_density(self, rho: Function, rho_Phys: Function) -> Function:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)

        return rho_Phys

    def filter_variables(self, rho: Function, rho_Phys: Function) -> TensorLike:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)

        return rho_Phys

    def filter_objective_sensitivities(self, 
                                       rho_Phys: Function, 
                                       obj_grad: TensorLike
                                    ) -> TensorLike:
        return obj_grad

    def filter_constraint_sensitivities(self, 
                                        rho_Phys: Function, 
                                        con_grad: TensorLike) -> TensorLike:

        return con_grad

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
                                    con_grad: TensorLike
                                ) -> TensorLike:

        return con_grad


class DensityStrategy(_FilterStrategy):
    """密度过滤策略"""
    def __init__(self, H: CSRTensor, cell_measure: TensorLike):
        self._H = H
        self._cell_measure = cell_measure

    def get_initial_density(self, rho: Function, rho_Phys: Function) -> Function:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)
        
        return rho_Phys

    def filter_variables(self, rho: Function, rho_Phys: Function) -> Function:

        # 单元密度情况
        # weighted_rho = rho[:] * self._cell_measure
        # numerator = self._H.matmul(weighted_rho)
        
        # denominator = self._H.matmul(self._cell_measure)

        # rho_Phys[:] = bm.set_at(rho_Phys, slice(None), numerator / denominator)


        # 高斯积分点密度情况：形状为 (NC, NQ)
        from fealpy.mesh import QuadrangleMesh
        mesh = QuadrangleMesh.from_box(box=[0, 30, 0, 10], nx=30, ny=10)
        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()

        nx, ny = int(mesh.meshdata['nx']/3), int(mesh.meshdata['ny']/3)
        reshaped = rho_Phys.reshape(nx, ny, 3, 3)
        # 将列索引提前，实现按列分组
        transposed = reshaped.transpose(0, 2, 1, 3)
        rho_Phys = transposed.reshape(-1)

        

        return rho_Phys

    def filter_objective_sensitivities(self, rho_Phys: TensorLike, obj_grad: TensorLike) -> TensorLike:
        # weighted_dobj = self._cell_measure * obj_grad
        # numerator = self._H.matmul(weighted_dobj)

        # denominator = self._H.matmul(self._cell_measure)

        # obj_grad = bm.set_at(obj_grad, slice(None), numerator / denominator)

        from fealpy.mesh import QuadrangleMesh
        mesh = QuadrangleMesh.from_box(box=[0, 30, 0, 10], nx=30, ny=10)
        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NQ = ws.shape[0]
        weighted_dobj = bm.einsum('q, cq -> cq', ws, obj_grad)
        ws_all = bm.broadcast_to(ws[None, :], (NC, NQ))

        denominator = self._H.matmul(ws_all)

        nx, ny = 30, 10
        reshaped = weighted_dobj.reshape(nx, ny, 3, 3)
        # 将列索引提前，实现按列分组
        transposed = reshaped.transpose(0, 2, 1, 3)
        weighted_dobj = transposed.reshape(-1)

        numerator = self._H.matmul(weighted_dobj)

        return obj_grad 

    def filter_constraint_sensitivities(self, rho_Phys: TensorLike, con_grad: TensorLike) -> TensorLike:
        weighted_dcons = self._cell_measure * con_grad
        numerator = self._H.matmul(weighted_dcons)

        denominator = self._H.matmul(self._cell_measure)

        con_grad = bm.set_at(con_grad, slice(None), numerator / denominator)

        return con_grad


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
        self._rho_Tilde = None
        self._beta_iter = 0

    def get_initial_density(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        self._rho_Tilde = rho
        projected_values = (1 - bm.exp(-self.beta * self._rho_Tilde) + 
                    self._rho_Tilde * math.exp(-self.beta)) + 2

        rho_Phys = bm.set_at(rho_Phys, slice(None), projected_values)

        return rho_Phys

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