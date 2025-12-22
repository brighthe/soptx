import math
from abc import ABC, abstractmethod
from typing import Tuple, Union, Literal, Optional

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import Function
from fealpy.mesh import HomogeneousMesh
from fealpy.typing import TensorLike
from fealpy.sparse import CSRTensor

from soptx.utils.base_logged import BaseLogged
from soptx.analysis.utils import reshape_multiresolution_data

class _FilterStrategy(ABC):
    """过滤方法的抽象基类 (内部使用)"""
    @abstractmethod
    def get_initial_density(self, 
                        density:  Union[TensorLike, Function]
                    ) ->  Union[TensorLike, Function]:
        pass

    @abstractmethod
    def filter_design_variable(self,
                            design_variable: TensorLike, 
                            physical_density: Function
                        ) -> Function:
        pass

    @abstractmethod
    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad: TensorLike
                                    ) -> TensorLike:
        pass

    @abstractmethod
    def filter_constraint_sensitivities(self, 
                                        design_variable: TensorLike, 
                                        con_grad: TensorLike
                                    ) -> TensorLike:
        pass


class NoneStrategy(_FilterStrategy, BaseLogged):
    """ '无操作' 策略, 当不需要过滤时使用"""
    def __init__(self,
                mesh: HomogeneousMesh,
                density_location: Literal['element', 'node', 'element_multiresolution'],
                integration_order: int = 4,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._mesh = mesh
        self._density_location = density_location
        self._integration_order = integration_order

    def get_initial_density(self, 
                        density:  Union[TensorLike, Function]
                    ) ->  Union[TensorLike, Function]:
        
        if isinstance(density, Function):
            rho_phys = density.space.function(bm.copy(density[:]))
        else:
            rho_phys = bm.copy(density)

        return rho_phys
    
    def filter_design_variable(self,
                            design_variable: TensorLike, 
                            physical_density: Function
                        ) -> Function:

        if self._density_location in ['element', 'node']:
            physical_density[:] = bm.set_at(physical_density, slice(None), design_variable)

        elif self._density_location == 'element_multiresolution':
            reshaped_dv = bm.reshape(design_variable, physical_density.shape)
            physical_density[:] = bm.set_at(physical_density, slice(None), reshaped_dv)

        else:
            error_msg = f"Unsupported density_location: {self._density_location}"
            self._log_error(error_msg)

        return physical_density
    
    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        obj_grad_dv = bm.reshape(obj_grad_rho, design_variable.shape)

        return obj_grad_dv
    
    def filter_constraint_sensitivities(self, 
                                design_variable: TensorLike, 
                                con_grad_rho: TensorLike
                            ) -> TensorLike:
        con_grad_dv = bm.reshape(con_grad_rho, design_variable.shape)

        return con_grad_dv


class SensitivityStrategy(_FilterStrategy, BaseLogged):
    """灵敏度过滤策略"""
    def __init__(self, 
                H: CSRTensor, 
                # Hs: TensorLike, 
                mesh: HomogeneousMesh, 
                density_location: Literal['element', 'node'], 
            ) -> None:
        
        self._H = H
        # self._Hs = Hs
        self._mesh = mesh
        self._density_location = density_location

        # --- 预计算测度权重 ---
        if self._density_location == 'element':
            # 单元密度表征：权重即为单元体积/面积
            # shape: (NC, )
            self._measure_weight = self._mesh.entity_measure('cell')
            
        elif self._density_location == 'node':
            # 节点密度表征：权重为节点控制体积
            # shape: (NN, )
            cm = self._mesh.entity_measure('cell')
            NN = self._mesh.number_of_nodes()
            cell2node = self._mesh.cell_to_node()
            NNE = cell2node.shape[1]

            # 将单元测度均分给每个节点
            val = bm.repeat(cm / NNE, NNE)
            nm = bm.zeros(NN, dtype=bm.float64)
            # 累加得到节点测度
            self._measure_weight = bm.add_at(nm, cell2node.reshape(-1), val)
        
        else:
            raise ValueError(f"Unsupported density_location: {self._density_location}")

        # 预计算卷积归一化因子
        self._Hs = self._H.matmul(self._measure_weight)

    def get_initial_density(self, 
                        density:  Union[TensorLike, Function]
                    ) ->  Union[TensorLike, Function]:

        from soptx.interpolation.interpolation_scheme import DensityDistribution
        if isinstance(density, Function):
            rho_phys = density.space.function(bm.copy(density[:]))
        elif isinstance(density, DensityDistribution):
            rho_phys = density
        else:
            rho_phys = bm.copy(density)

        return rho_phys

    def filter_design_variable(self,
                            design_variable: TensorLike, 
                            physical_density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:
        if self._density_location in ['element', 'node']:
            physical_density[:] = bm.set_at(physical_density, slice(None), design_variable)

        return physical_density
    
    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        # 1. 准备源项
        weighted_source = design_variable * obj_grad_rho / self._measure_weight
        # 2. 卷积
        numerator_conv = self._H.matmul(weighted_source)
        # 3. 稳定性因子
        epsilon = 1e-3
        stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), design_variable)
        # 4. 分母        
        denominator = stability_factor * self._Hs
        # 5. 组合
        obj_grad_dv = (numerator_conv * self._measure_weight) / denominator

        return obj_grad_dv

        # if self._density_location in ['element']:
        #     # obj_grad_rho.shape = (NC, )
        #     cm = self._mesh.entity_measure('cell')

        #     weighted_obj_grad_rho = design_variable[:] * obj_grad_rho / cm
        #     numerator = self._H.matmul(weighted_obj_grad_rho)

        #     epsilon = 1e-3
        #     stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), design_variable)
        #     denominator = (stability_factor / cm) * self._Hs

        #     obj_grad_dv = numerator / denominator

        #     return obj_grad_dv
        
        # else:
        #     raise NotImplementedError("Sensitivity filtering only supports 'element' density location.")

    def filter_constraint_sensitivities(self, 
                                    design_variable: Union[TensorLike, Function],
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:
        
        # 对于简单的 OC 算法，体积约束不需要过滤
        con_grad_dv = bm.copy(con_grad_rho)

        return con_grad_dv
    
        # if self._density_location == 'element':
        #     # 对于通用的 MMA 算法，体积约束越需要过滤
        #     # cell_measure = self._integration_weights

        #     # weighted_con_grad = rho_Phys[:] * con_grad / cell_measure
        #     # numerator = self._H.matmul(weighted_con_grad)

        #     # epsilon = 1e-3
        #     # stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), rho_Phys)
        #     # denominator = (stability_factor / cell_measure) * self._Hs

        #     # con_grad = bm.set_at(con_grad, slice(None), numerator / denominator)

        #     # 对于简单的 OC 算法，体积约束不需要过滤
        #     con_grad_dv = bm.copy(con_grad_rho)

        #     return con_grad_dv

        # else:
        #     raise NotImplementedError("Sensitivity filtering only supports 'element' density location.")
 

class DensityStrategy(_FilterStrategy, BaseLogged):
    """密度过滤策略"""
    def __init__(self, 
                H: CSRTensor, 
                mesh: HomogeneousMesh, 
                density_location: Literal['element', 'node', 'element_multiresolution'],
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._H = H
        self._mesh = mesh                           
        self._density_location = density_location

        # --- 预计算测度权重 ---
        if self._density_location in ['element', 'element_multiresolution']:
            # 单元密度表征：权重即为设计变量网格单元体积/面积
            # shape: (NC, )
            self._measure_weight = self._mesh.entity_measure('cell')
            
        elif self._density_location == 'node':
            # 节点密度表征：权重为节点控制体积
            # shape: (NN, )
            cm = self._mesh.entity_measure('cell')
            NN = self._mesh.number_of_nodes()
            cell2node = self._mesh.cell_to_node()
            NNE = cell2node.shape[1]

            # 将单元测度均分给每个节点
            val = bm.repeat(cm / NNE, NNE)
            nm = bm.zeros(NN, dtype=bm.float64)
            # 累加得到节点测度
            self._measure_weight = bm.add_at(nm, cell2node.reshape(-1), val)

        else:
            error_msg = f"Unsupported density_location: {self._density_location}"
            self._log_error(error_msg)

        # 预计算卷积归一化因子
        self._Hs = self._H.matmul(self._measure_weight)
        
    def get_initial_density(self, 
                        density:  Union[TensorLike, Function]
                    ) ->  Union[TensorLike, Function]:

        from soptx.interpolation.interpolation_scheme import DensityDistribution
        if isinstance(density, Function):
            rho_phys = density.space.function(bm.copy(density[:]))
        elif isinstance(density, DensityDistribution):
            rho_phys = density
        else:
            rho_phys = bm.copy(density)

        return rho_phys
    
    def filter_design_variable(self,
                            design_variable: TensorLike, 
                            physical_density: Function
                        ) -> Function:
        
        # 1. 对设计变量进行测度加权
        weighted_dv = design_variable * self._measure_weight
        # 2. 卷积求和
        numerator = self._H.matmul(weighted_dv)
        # 3. 归一化并赋值
        if self._density_location in ['element', 'node']:
            physical_density[:] = bm.set_at(physical_density, slice(None), numerator / self._Hs)

        elif self._density_location == 'element_multiresolution':
            from soptx.analysis.utils import reshape_multiresolution_data_inverse
            n_sub = physical_density.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)
            sub_physical_density = reshape_multiresolution_data_inverse(nx=nx_displacement,
                                                                    ny=ny_displacement,
                                                                    data_flat=numerator / self._Hs,
                                                                    n_sub=n_sub) # (NC, n_sub)
            physical_density[:] = bm.set_at(physical_density, slice(None), sub_physical_density)
        
        else:
            error_msg = f"Unsupported density_location: {self._density_location}"
            self._log_error(error_msg)
        
        return physical_density

    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        if self._density_location == 'element_multiresolution':
            # 多分辨率：obj_grad_rho (NC, n_sub) ->  (NC * n_sub, )
            n_sub = obj_grad_rho.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)
            obj_grad_rho = reshape_multiresolution_data(nx=nx_displacement, ny=ny_displacement, data=obj_grad_rho)  # (NC * n_sub, )
            
        # 1. 缩放物理密度导数
        scaled_dobj = obj_grad_rho / self._Hs
        
        # 2. 卷积求和
        temp = self._H.matmul(scaled_dobj)

        # 3. 乘以测度权重
        obj_grad_dv = self._measure_weight * temp

        return obj_grad_dv

    def filter_constraint_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:
        if self._density_location == 'element_multiresolution':
            # 多分辨率：obj_grad_rho (NC, n_sub) ->  (NC * n_sub, )
            n_sub = con_grad_rho.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)
            con_grad_rho = reshape_multiresolution_data(nx=nx_displacement, ny=ny_displacement, data=con_grad_rho)  # (NC * n_sub, )
    
        # 1. 缩放物理密度导数
        scaled_dcon = con_grad_rho / self._Hs
        
        # 2. 卷积求和
        temp = self._H.matmul(scaled_dcon)

        # 3. 乘以测度权重
        con_grad_dv = self._measure_weight * temp

        return con_grad_dv
    

class ProjectionStrategy(DensityStrategy):
    """
    基于 Heaviside 投影的非线性映射策略
    该策略首先执行线性密度过滤, 然后应用 Heaviside 投影    
    """
    def __init__(self, 
                H: CSRTensor, 
                mesh: HomogeneousMesh, 
                density_location: Literal['element', 'node', 'element_multiresolution'],
                projection_type: Literal['tanh', 'exponential'] = 'exponential', 
                beta: float = 1.0,
                eta: float = 0.5,
                beta_max: float = 512.0,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(H, mesh, density_location, enable_logging, logger_name)
        
        # 投影参数
        self.beta = beta      # 控制投影陡峭程度
        self.eta = eta        # 投影阈值 (通常为 0.5)
        self.beta_max = beta_max
        
        # 用于存储线性过滤后的中间密度 (rho_tilde)，用于灵敏度分析的链式法则
        self._rho_tilde_cache: Optional[TensorLike] = None

        def get_initial_density(self, 
                            density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:
            rho_phys = super().get_initial_density(density)

            if isinstance(rho_phys, Function):
                val = rho_phys[:]
                self._rho_tilde_cache = bm.copy(val)
                
                # 3. 执行投影: rho_tilde -> rho_bar
                projected_val = self._apply_projection(self._rho_tilde_cache)
                
                # 4. 更新 Function 中的值
                rho_phys[:] = bm.set_at(rho_phys, slice(None), projected_val)
                
            else:
                # 处理 TensorLike 的情况
                self._rho_tilde_cache = bm.copy(rho_phys)
                rho_phys = self._apply_projection(self._rho_tilde_cache)

            return rho_phys

    def update_beta(self, current_step: int, update_interval: int = 50, double: bool = True):
        """
        更新投影参数 beta (Continuation Scheme)。
        通常在优化过程中逐步增大 beta 以避免局部最优。
        """
        if self.beta >= self.beta_max:
            return

        if current_step > 0 and current_step % update_interval == 0:
            if double:
                self.beta = min(self.beta * 2, self.beta_max)
            else:
                self.beta = min(self.beta + 1, self.beta_max)
            
            if self._enable_logging:
                print(f"Projection beta updated to: {self.beta}")

    def _tanh_projection(self, x: TensorLike) -> TensorLike:
        """
        计算 Tanh 形式的 Heaviside 投影。
        rho_bar = (tanh(beta*eta) + tanh(beta*(rho_tilde - eta))) / (tanh(beta*eta) + tanh(beta*(1 - eta)))
        """
        # 为了数值稳定性，避免重复计算常数项
        tanh_beta_eta = bm.tanh(self.beta * self.eta)
        tanh_beta_1_eta = bm.tanh(self.beta * (1.0 - self.eta))
        denominator = tanh_beta_eta + tanh_beta_1_eta
        
        numerator = tanh_beta_eta + bm.tanh(self.beta * (x - self.eta))
        return numerator / denominator

    def _tanh_projection_grad(self, x: TensorLike) -> TensorLike:
        """
        计算 Tanh 投影关于中间密度的导数 d(rho_bar) / d(rho_tilde)。
        """
        tanh_beta_eta = bm.tanh(self.beta * self.eta)
        tanh_beta_1_eta = bm.tanh(self.beta * (1.0 - self.eta))
        denominator = tanh_beta_eta + tanh_beta_1_eta
        
        # d/dx (tanh(u)) = sech^2(u) * u' = (1 - tanh^2(u)) * u'
        inner = self.beta * (x - self.eta)
        # 注意：这里假设 bm 支持 tanh，导数为 beta * (1 - tanh^2)
        derivative_numerator = self.beta * (1.0 - bm.square(bm.tanh(inner)))
        
        return derivative_numerator / denominator

    def filter_design_variable(self,
                            design_variable: TensorLike, 
                            physical_density: Function
                        ) -> Function:
        """
        正向映射: d -> rho_tilde -> rho_bar
        """
        # 1. 调用父类方法，执行线性密度过滤
        # 注意：父类方法会直接修改 physical_density 的值
        # 此时 physical_density 存储的是中间密度 (rho_tilde)
        super().filter_design_variable(design_variable, physical_density)
        
        # 2. 缓存中间密度 (rho_tilde)
        # 这一步非常关键，因为非线性映射的导数依赖于中间密度的值
        # 必须使用 copy，因为 physical_density 马上要被覆盖为 rho_bar
        self._rho_tilde_cache = bm.copy(physical_density[:]) 
        
        # 3. 执行 Heaviside 投影: rho_tilde -> rho_bar
        # 直接原地修改 physical_density
        projected_val = self._tanh_projection(self._rho_tilde_cache)
        physical_density[:] = bm.set_at(physical_density, slice(None), projected_val)
        
        return physical_density

    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        """
        灵敏度分析: Chain Rule
        dC/dd = (dC/d_rho_bar) * (d_rho_bar/d_rho_tilde) * (d_rho_tilde/dd)
        
        输入 obj_grad_rho 为 dC/d_rho_bar
        """
        if self._rho_tilde_cache is None:
            raise RuntimeError("filter_design_variable must be called before filter_sensitivities to cache intermediate density.")

        # 1. 处理多分辨率数据的形状 (如果是 element_multiresolution)
        # 确保 cached rho_tilde 和传入的梯度形状一致
        rho_tilde_for_grad = self._rho_tilde_cache
        
        # 如果是多分辨率，传入的梯度可能是 (NC, n_sub)，但也可能在外部已经被展平
        # 这里我们需要确保 rho_tilde 和 obj_grad_rho 形状对齐以进行逐元素相乘
        if self._density_location == 'element_multiresolution' and obj_grad_rho.ndim > 1:
             # 如果梯度是 (NC, n_sub)，我们的缓存通常也是 (NC, n_sub) 或者展平的
             # 根据父类逻辑，Function 内部通常存储为 (NC, n_sub)
             pass 

        # 2. 计算投影导数: d_rho_bar / d_rho_tilde
        # 这一步是非线性映射引入的额外缩放因子
        projection_derivative = self._tanh_projection_grad(rho_tilde_for_grad)
        
        # 3. 应用链式法则第一步: dC/d_rho_tilde = (dC/d_rho_bar) * projection_derivative
        # 逐元素相乘
        obj_grad_rho_tilde = obj_grad_rho * projection_derivative
        
        # 4. 应用链式法则第二步: dC/dd = (dC/d_rho_tilde) * (d_rho_tilde/dd)
        # 调用父类的灵敏度过滤方法，父类方法处理线性卷积部分
        return super().filter_objective_sensitivities(design_variable, obj_grad_rho_tilde)

    def filter_constraint_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:
        """
        约束灵敏度分析，逻辑同目标函数
        """
        if self._rho_tilde_cache is None:
             raise RuntimeError("filter_design_variable must be called before filter_sensitivities.")

        # 1. 计算投影导数
        projection_derivative = self._tanh_projection_grad(self._rho_tilde_cache)
        
        # 2. 链式法则第一步
        con_grad_rho_tilde = con_grad_rho * projection_derivative
        
        # 3. 链式法则第二步 (调用父类)
        return super().filter_constraint_sensitivities(design_variable, con_grad_rho_tilde)


class HeavisideDensityStrategy(_FilterStrategy):
    """密度过滤 + Heaviside 投影"""
    def __init__(self, 
                H: CSRTensor, 
                density_location: Literal['element', 'node'], 
                integration_weights: TensorLike, 
                mesh: HomogeneousMesh, 
                beta: float = 1.0, max_beta: float = 512, continuation_iter: int = 50
            ):
        
        self._H = H
        self._density_location = density_location
        self._integration_weights = integration_weights
        self._mesh = mesh
        
        self._beta = beta
        self._max_beta = max_beta
        self._continuation_iter = continuation_iter
        self._rho_Tilde = None
        self._beta_iter = 0

    def get_initial_density(self, density: Function, physical_density: Function) -> Function:

        self._rho_Tilde = density
        projected_values = 1 - bm.exp(-self._beta * self._rho_Tilde) + self._rho_Tilde * bm.exp(-self._beta)

        physical_density = bm.set_at(physical_density, slice(None), projected_values)

        return physical_density

    def filter_variables(self, 
                        design_variable: Union[TensorLike, Function], 
                        rho_Phys: Union[TensorLike, Function]
                    ) -> Union[TensorLike, Function]:
        
        if self._density_location in ['element', 'element_multiresolution']:

            dv = design_variable
            cell_measure = self._integration_weights
            weighted_rho = dv[:] * cell_measure
        
            numerator = self._H.matmul(weighted_rho)
            denominator = self._H.matmul(cell_measure)

            self._rho_Tilde[:] = bm.set_at(self._rho_Tilde, slice(None), numerator / denominator)

            projected_values = 1 - bm.exp(-self._beta * self._rho_Tilde) + self._rho_Tilde * bm.exp(-self._beta)

            rho_Phys[:] = bm.set_at(rho_Phys, slice(None), projected_values)

        return rho_Phys

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