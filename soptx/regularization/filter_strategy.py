import math
from abc import ABC, abstractmethod
from typing import Tuple, Union, Literal, Optional

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import Function
from fealpy.mesh import HomogeneousMesh
from fealpy.typing import TensorLike
from fealpy.sparse import CSRTensor

from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer
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

    def continuation_step(self, change: float) -> Tuple[float, bool]:

        return change, False


class NoneStrategy(_FilterStrategy, BaseLogged):
    """ '无操作' 策略, 当不需要过滤时使用"""
    def __init__(self,
                mesh: HomogeneousMesh,
                density_location: Literal['element', 'node', 'element_multiresolution'],
                integration_order: int = 4,
                enable_logging: bool = False,
                logger_name: Optional[str] = None,
                **kwargs
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
                                    obj_grad_rho: TensorLike,
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
                            physical_density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:
        if self._density_location in ['element', 'node']:
            physical_density_filter = bm.set_at(physical_density, slice(None), design_variable)
            # physical_density[:] = bm.set_at(physical_density, slice(None), design_variable)

        elif self._density_location == 'element_multiresolution':
            from soptx.analysis.utils import reshape_multiresolution_data_inverse
            n_sub = physical_density.shape[-1]
            n_sub_x = int(math.sqrt(n_sub))
            n_sub_y = int(math.sqrt(n_sub))
            nx_displacement = int(self._mesh.meshdata['nx'] / n_sub_x)
            ny_displacement = int(self._mesh.meshdata['ny'] / n_sub_y)
            sub_physical_density = reshape_multiresolution_data_inverse(
                                                    nx=nx_displacement,
                                                    ny=ny_displacement,
                                                    data_flat=design_variable, # 注意这里直接使用 dv，不做卷积
                                                    n_sub=n_sub
                                                ) 
            physical_density[:] = bm.set_at(physical_density, slice(None), sub_physical_density)
        
        else:
            error_msg = f"Unsupported density_location: {self._density_location}"
            self._log_error(error_msg)
        
        return physical_density_filter
    
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

        # 1. 准备源项
        weighted_source = self._measure_weight * design_variable * obj_grad_rho
        # 2. 卷积
        numerator_conv = self._H.matmul(weighted_source)
        # 3. 稳定性因子
        epsilon = 1e-3
        stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), design_variable)
        # 4. 分母        
        denominator = stability_factor * self._Hs
        # 5. 组合
        obj_grad_dv = numerator_conv  / denominator

        return obj_grad_dv

    def filter_constraint_sensitivities(self, 
                                    design_variable: Union[TensorLike, Function],
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:
        
        #* 对于简单的 OC 算法，体积约束不需要过滤
        if self._density_location == 'element_multiresolution':
            n_sub = con_grad_rho.shape[-1]
            n_sub_x = int(math.sqrt(n_sub))
            nx_displacement = int(self._mesh.meshdata['nx'] / n_sub_x)
            ny_displacement = int(self._mesh.meshdata['ny'] / n_sub_x)
            con_grad_dv = reshape_multiresolution_data(nx=nx_displacement, ny=ny_displacement, data=con_grad_rho) # (NC * n_sub, )

        else:
            con_grad_dv = bm.copy(con_grad_rho)

        return con_grad_dv


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
        #? matmul 函数下 self._H 必须是 COO 格式, 不能是 CSR 格式, 否则 GPU 下 device_put 函数会出错
        device = self._mesh.device
        self._H = self._H.device_put(device)
        self._Hs = self._H.matmul(self._measure_weight)
        # val = bm.tensor(data=1, dtype=bm.float32, device='cpu')
        # val = bm.device_put(val, device='cuda')
        # print("------------")
        
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
            physical_density_filter = bm.set_at(physical_density, slice(None), numerator / self._Hs)
            # physical_density[:] = bm.set_at(physical_density, slice(None), numerator / self._Hs)

        elif self._density_location == 'element_multiresolution':
            from soptx.analysis.utils import reshape_multiresolution_data_inverse
            n_sub = physical_density.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)
            sub_physical_density = reshape_multiresolution_data_inverse(nx=nx_displacement,
                                                                    ny=ny_displacement,
                                                                    data_flat=numerator / self._Hs,
                                                                    n_sub=n_sub) # (NC, n_sub)
            physical_density_filter = bm.set_at(physical_density, slice(None), sub_physical_density)
            # physical_density[:] = bm.set_at(physical_density, slice(None), sub_physical_density)
        
        else:
            error_msg = f"Unsupported density_location: {self._density_location}"
            self._log_error(error_msg)
        
        return physical_density_filter

    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike,
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
                continuation_iter: int = 50,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(H, mesh, density_location, enable_logging, logger_name)
        
        # 投影参数
        self.projection_type = projection_type
        self.beta = beta      # 控制投影陡峭程度
        self.eta = eta        # 投影阈值 (通常为 0.5)
        self.beta_max = beta_max
        self.continuation_iter = continuation_iter

        # 初始化计数器
        self._beta_iter = 0 
        
        # 用于存储线性过滤后的中间密度 (rho_tilde)，用于灵敏度分析的链式法则
        self._rho_tilde_cache: Optional[TensorLike] = None

    def _apply_projection(self, rho_tilde: TensorLike) -> TensorLike:
        if self.projection_type == 'tanh':
            tanh_beta_eta = bm.tanh(self.beta * self.eta)
            tanh_beta_1_eta = bm.tanh(self.beta * (1.0 - self.eta))
            numerator = tanh_beta_eta + bm.tanh(self.beta * (rho_tilde - self.eta))
            denominator = tanh_beta_eta + tanh_beta_1_eta
            return numerator / denominator
            
        elif self.projection_type == 'exponential':
            term1 = bm.exp(-self.beta * rho_tilde)
            term2 = rho_tilde * bm.exp(-self.beta)
            return 1.0 - term1 + term2
        
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")

    def get_initial_density(self, 
                            density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:
            rho_phys = super().get_initial_density(density)

            if isinstance(rho_phys, Function):
                val = rho_phys[:]
                self._rho_tilde_cache = bm.copy(val)
                
                # 执行投影: rho_tilde -> rho_bar
                projected_val = self._apply_projection(self._rho_tilde_cache)
                
                rho_phys[:] = bm.set_at(rho_phys, slice(None), projected_val)
                
            else:
                # 处理 TensorLike 的情况
                self._rho_tilde_cache = bm.copy(rho_phys)
                rho_phys = self._apply_projection(self._rho_tilde_cache)

            return rho_phys
    
    def filter_design_variable(self, design_variable: TensorLike, physical_density: Function) -> Function:
        super().filter_design_variable(design_variable, physical_density)
        
        self._rho_tilde_cache = bm.copy(physical_density[:])
        
        rho_val = self._apply_projection(self._rho_tilde_cache)
        physical_density[:] = bm.set_at(physical_density, slice(None), rho_val)
        
        return physical_density
    
    def _apply_projection_derivative(self, rho_tilde: TensorLike) -> TensorLike:
        if self.projection_type == 'tanh':
            tanh_beta_eta = bm.tanh(self.beta * self.eta)
            tanh_beta_1_eta = bm.tanh(self.beta * (1.0 - self.eta))
            denominator = tanh_beta_eta + tanh_beta_1_eta
            inner = self.beta * (rho_tilde - self.eta)
            numerator = self.beta * (1.0 - bm.square(bm.tanh(inner)))
            return numerator / denominator

        elif self.projection_type == 'exponential':
            return self.beta * bm.exp(-self.beta * rho_tilde) + bm.exp(-self.beta)
        
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")

    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        if self._rho_tilde_cache is None:
            raise RuntimeError("filter_design_variable must be called before filter_sensitivities to cache intermediate density.")
        
        d_rho_d_tilde = self._apply_projection_derivative(self._rho_tilde_cache)
        obj_grad_tilde = obj_grad_rho * d_rho_d_tilde

        obj_grad_dv = super().filter_objective_sensitivities(design_variable, obj_grad_tilde)

        return obj_grad_dv
    
    def filter_constraint_sensitivities(self, design_variable: TensorLike, con_grad_rho: TensorLike) -> TensorLike:
        if self._rho_tilde_cache is None:
            raise RuntimeError("filter_design_variable must be called before sensitivities.")

        d_rho_d_tilde = self._apply_projection_derivative(self._rho_tilde_cache)
        con_grad_tilde = con_grad_rho * d_rho_d_tilde

        con_grad_dv = super().filter_constraint_sensitivities(design_variable, con_grad_tilde)
        
        return con_grad_dv
    
    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """执行一步 beta continuation (更新 beta 值)"""
        self._beta_iter += 1
        
        # 判断条件：beta 未达上限 且 (达到迭代间隔 或 收敛)
        if (self.beta < self.beta_max and 
                (self._beta_iter >= self.continuation_iter or change <= 0.01)):
            
            # 记录旧值用于日志
            old_beta = self.beta
            
            # 1. 加倍 Beta
            self.beta = min(self.beta * 2, self.beta_max)
            
            # 2. 重置计数器
            self._beta_iter = 0
            
            if self._enable_logging:
                trigger = "Interval" if self._beta_iter >= self.continuation_iter else "Convergence"
                print(f"[{trigger}] Projection beta updated: {old_beta} -> {self.beta}")
            
            # 3. 关键：强制返回 1.0，防止外层循环提前退出
            return 1.0, True
        
        # 如果没有更新，保持原有的 change 值
        return change, False