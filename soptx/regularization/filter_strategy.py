import math
from abc import ABC, abstractmethod
from typing import Tuple, Union, Literal

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import Function
from fealpy.mesh import HomogeneousMesh
from fealpy.typing import TensorLike
from fealpy.sparse import CSRTensor

from soptx.utils.gauss_intergation_point_mapping import get_gauss_integration_point_mapping

class _FilterStrategy(ABC):
    """过滤方法的抽象基类 (内部使用)"""
    @abstractmethod
    def get_initial_density(self, 
                        density:  Union[TensorLike, Function]
                    ) ->  Union[TensorLike, Function]:
        pass

    @abstractmethod
    def filter_design_variable(self,
                            design_variable: Union[TensorLike, Function], 
                            physical_density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:
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


class NoneStrategy(_FilterStrategy):
    """ '无操作' 策略, 当不需要过滤时使用"""
    def __init__(self,
                mesh: HomogeneousMesh,
                density_location: Literal['element', 'node'],
                integration_order: int = 4,
            ) -> None:
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
                            physical_density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:

        if self._density_location in ['element', 'node']:

            physical_density[:] = bm.set_at(physical_density, slice(None), design_variable)

        return physical_density
    
    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        obj_grad_dv = bm.copy(obj_grad_rho)

        return obj_grad_dv
    
    def filter_constraint_sensitivities(self, 
                                design_variable: TensorLike, 
                                con_grad_rho: TensorLike
                            ) -> TensorLike:
        con_grad_dv = bm.copy(con_grad_rho)

        return con_grad_dv


class SensitivityStrategy(_FilterStrategy):
    """灵敏度过滤策略"""
    def __init__(self, 
                H: CSRTensor, 
                Hs: TensorLike, 
                density_location: Literal['element'], 
                integration_weights: TensorLike,
                mesh: HomogeneousMesh, 
                integration_order: int
            ) -> None:
        
        self._H = H
        self._Hs = Hs
        self._density_location = density_location
        self._integration_weights = integration_weights

    def get_initial_density(self, rho: Function, rho_Phys: Function) -> Function:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)
        
        return rho_Phys

    def filter_variables(self, rho: Function, rho_Phys: Function) -> Function:
        rho_Phys = bm.set_at(rho_Phys, slice(None), rho)

        return rho_Phys

    def filter_objective_sensitivities(self, 
                                    rho_Phys: Union[TensorLike, Function], 
                                    obj_grad: TensorLike
                                ) -> TensorLike:

        if self._density_location == 'element':

            cell_measure = self._integration_weights

            weighted_obj_grad = rho_Phys[:] * obj_grad / cell_measure
            numerator = self._H.matmul(weighted_obj_grad)

            epsilon = 1e-3
            stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), rho_Phys)
            denominator = (stability_factor / cell_measure) * self._Hs

            obj_grad = bm.set_at(obj_grad, slice(None), numerator / denominator)

            return obj_grad
        
        else:
            raise NotImplementedError("Sensitivity filtering only supports 'element' density location.")

    def filter_constraint_sensitivities(self, 
                                    rho_Phys: Union[TensorLike, Function], 
                                    con_grad: TensorLike
                                ) -> TensorLike:
        
        if self._density_location == 'element':

            # 对于通用的 MMA 算法，体积约束越需要过滤
            # cell_measure = self._integration_weights

            # weighted_con_grad = rho_Phys[:] * con_grad / cell_measure
            # numerator = self._H.matmul(weighted_con_grad)

            # epsilon = 1e-3
            # stability_factor = bm.maximum(bm.tensor(epsilon, dtype=bm.float64), rho_Phys)
            # denominator = (stability_factor / cell_measure) * self._Hs

            # con_grad = bm.set_at(con_grad, slice(None), numerator / denominator)

            # 对于简单的 OC 算法，体积约束不需要过滤
            return con_grad
        
        else:
            raise NotImplementedError("Sensitivity filtering only supports 'element' density location.")
 

class DensityStrategy(_FilterStrategy):
    """密度过滤策略"""
    def __init__(self, 
                H: CSRTensor, 
                mesh: HomogeneousMesh, 
                density_location: Literal['element', 'node'], 
            ) -> None:
        
        self._H = H
        self._mesh = mesh
        self._density_location = density_location
        
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
                            design_variable: Union[TensorLike, Function], 
                            physical_density: Union[TensorLike, Function]
                        ) -> Union[TensorLike, Function]:

        if self._density_location in ['element']:
            # design_variable.shape = (NC, )
            cm = self._mesh.entity_measure('cell')
            
            weighted_dv = design_variable * cm

            numerator = self._H.matmul(weighted_dv)
            denominator = self._H.matmul(cm)

            physical_density[:] = bm.set_at(physical_density, slice(None), numerator / denominator) # (NC, )
            
            return physical_density
        
        elif self._density_location in ['node']:
            # design_variable.shape = (NN, )
            cm = self._mesh.entity_measure('cell')
            NN = self._mesh.number_of_nodes()
            cell2node = self._mesh.cell_to_node()
            NNE = cell2node.shape[1]

            # 计算节点测度
            val = bm.repeat(cm / NNE, NNE)
            nm = bm.zeros(NN, dtype=bm.float64)
            nm = bm.add_at(nm, cell2node.reshape(-1), val)

            # 过滤设计变量
            weighted_dv = design_variable * nm

            numerator = self._H.matmul(weighted_dv)
            denominator = self._H.matmul(nm)

            physical_density[:] = bm.set_at(physical_density, slice(None), numerator / denominator)  # (NN, )
            
            return physical_density
        
        elif self._density_location in ['element_multiresolution']:
            # design_variable.shape = (NC*n_sub, )
            cm = self._mesh.entity_measure('cell')
            weighted_dv = design_variable * cm

            numerator = self._H.matmul(weighted_dv)
            denominator = self._H.matmul(cm)

            data_flat = numerator / denominator

            from soptx.analysis.utils import reshape_multiresolution_data_inverse
            n_sub = physical_density.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)
            physical_density_sub = reshape_multiresolution_data_inverse(nx=nx_displacement,
                                                                ny=ny_displacement,
                                                                data_flat=data_flat,
                                                                n_sub=n_sub) # (NC, n_sub)

            physical_density[:] = bm.set_at(physical_density, slice(None), physical_density_sub) # (NC, n_sub)
            
            return physical_density


        # elif self._density_location == 'gauss_integration_point' or self._density_location == 'density_subelement_gauss_point':

        #     weighted_rho_local = bm.einsum('cq, cq -> cq', rho, self._integration_weights) # (NC, NQ)

        #     nx, ny = self._mesh.meshdata['nx'], self._mesh.meshdata['ny']
        #     local_to_global, global_to_local = get_gauss_integration_point_mapping(nx=nx, ny=ny,
        #                                                             nq_per_dim=self._integration_order)
            
        #     weighted_rho = weighted_rho_local[local_to_global] # (NC*NQ, )

        #     integration_weights = self._integration_weights[local_to_global] # (NC*NQ, )

        #     numerator_global = self._H.matmul(weighted_rho) # (NC*NQ, )
        #     numerator = numerator_global[global_to_local] # (NC, NQ)
            
        #     denominator_global = self._H.matmul(integration_weights) # (NC*NQ, )
        #     denominator = denominator_global[global_to_local] # (NC, NQ)

        #     rho_Phys[:] = bm.set_at(rho_Phys, slice(None), numerator / denominator)

        #     return rho_Phys

    def filter_objective_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:
        
        if self._density_location in ['element']:
            # obj_grad_rho.shape = (NC, )
            cm = self._mesh.entity_measure('cell')
            
            Hs = self._H.matmul(cm)
            
            scaled_dobj = obj_grad_rho / Hs
            temp = self._H.matmul(scaled_dobj)

            obj_grad_dv = cm * temp # (NC, )

            return obj_grad_dv
        
        elif self._density_location in ['node']:
            # obj_grad_rho.shape = (NN, )
            cm = self._mesh.entity_measure('cell')
            NN = self._mesh.number_of_nodes()
            cell2node = self._mesh.cell_to_node()
            NNE = cell2node.shape[1]

            # 计算节点测度
            val = bm.repeat(cm / NNE, NNE)
            nm = bm.zeros(NN, dtype=bm.float64)
            nm = bm.add_at(nm, cell2node.reshape(-1), val)

            # 灵敏度过滤
            Hs = self._H.matmul(nm)
            
            scaled_dobj = obj_grad_rho / Hs
            temp = self._H.matmul(scaled_dobj)

            obj_grad_dv = nm * temp # (NN, )

            return obj_grad_dv

        elif self._density_location in ['element_multiresolution']:
            # obj_grad_rho.shape = (NC, n_sub)
            n_sub = obj_grad_rho.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)

            from soptx.analysis.utils import reshape_multiresolution_data
            obj_grad_rho_reshaped = reshape_multiresolution_data(nx=nx_displacement, 
                                                                ny=ny_displacement, 
                                                                data=obj_grad_rho) # (NC*n_sub, )

            cm = self._mesh.entity_measure('cell')
            
            Hs = self._H.matmul(cm)

            scaled_dobj = obj_grad_rho_reshaped / Hs
            temp = self._H.matmul(scaled_dobj)

            obj_grad_dv = cm * temp # (NC*n_sub, )

            return obj_grad_dv

        elif self._density_location in ['node_multiresolution']:
            pass 

        # elif self._density_location == 'gauss_integration_point' or self._density_location == 'density_subelement_gauss_point':
            
        #     from soptx.utils.gauss_intergation_point_mapping import get_gauss_integration_point_mapping

        #     weighted_dobj_local = bm.einsum('cq, cq -> cq', obj_grad, self._integration_weights) # (NC, NQ)

        #     nx, ny = self._mesh.meshdata['nx'], self._mesh.meshdata['ny']
        #     local_to_global, global_to_local = get_gauss_integration_point_mapping(nx=nx, ny=ny,
        #                                                             nq_per_dim=self._integration_order)
            
        #     weighted_dobj = weighted_dobj_local[local_to_global] # (NC*NQ, )

        #     integration_weights = self._integration_weights[local_to_global] # (NC*NQ, )

        #     numerator_global = self._H.matmul(weighted_dobj) # (NC*NQ, )
        #     numerator = numerator_global[global_to_local] # (NC, NQ)
            
        #     denominator_global = self._H.matmul(integration_weights) # (NC*NQ, )
        #     denominator = denominator_global[global_to_local] # (NC, NQ)

        #     obj_grad = bm.set_at(obj_grad, slice(None), numerator / denominator)

            # return obj_grad 

    def filter_constraint_sensitivities(self, 
                                    design_variable: TensorLike, 
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:

        if self._density_location in ['element']:
            # con_grad_rho.shape = (NC, )
            cm = self._mesh.entity_measure('cell')
            Hs = self._H.matmul(cm)
            
            scaled_dcon = con_grad_rho / Hs
            temp = self._H.matmul(scaled_dcon)

            con_grad_dv = cm * temp # (NC, )

            return con_grad_dv
        
        elif self._density_location in ['node']:
            # con_grad_rho.shape = (NN, )
            cm = self._mesh.entity_measure('cell')
            NN = self._mesh.number_of_nodes()
            cell2node = self._mesh.cell_to_node()
            NNE = cell2node.shape[1]

            # 计算节点测度
            val = bm.repeat(cm / NNE, NNE)
            nm = bm.zeros(NN, dtype=bm.float64)
            nm = bm.add_at(nm, cell2node.reshape(-1), val)

            # 灵敏度过滤
            Hs = self._H.matmul(nm)
            
            scaled_dcon = con_grad_rho / Hs
            temp = self._H.matmul(scaled_dcon)

            con_grad_dv = nm * temp  # (NN, )

            return con_grad_dv

        elif self._density_location in ['element_multiresolution']:
            # con_grad_rho.shape = (NC, n_sub)
            n_sub = con_grad_rho.shape[-1]
            n_sub_x, n_sub_y = int(math.sqrt(n_sub)), int(math.sqrt(n_sub))
            nx_displacement, ny_displacement = int(self._mesh.meshdata['nx'] / n_sub_x), int(self._mesh.meshdata['ny'] / n_sub_y)

            from soptx.analysis.utils import reshape_multiresolution_data
            con_grad_rho_reshaped = reshape_multiresolution_data(nx=nx_displacement, 
                                                                ny=ny_displacement, 
                                                                data=con_grad_rho) # (NC*n_sub, )

            cm = self._mesh.entity_measure('cell')
            Hs = self._H.matmul(cm)

            scaled_dobj = con_grad_rho_reshaped / Hs
            temp = self._H.matmul(scaled_dobj)

            con_grad_dv = cm * temp # (NC*n_sub, )

            return con_grad_dv

        elif self._density_location in ['node_multiresolution']:
            pass 

        # elif self._density_location == 'gauss_integration_point' or self._density_location == 'density_subelement_gauss_point':
        #     from soptx.utils.gauss_intergation_point_mapping import get_gauss_integration_point_mapping

        #     weighted_dobj_local = bm.einsum('cq, cq -> cq', con_grad, self._integration_weights) # (NC, NQ)

        #     nx, ny = self._mesh.meshdata['nx'], self._mesh.meshdata['ny']
        #     local_to_global, global_to_local = get_gauss_integration_point_mapping(nx=nx, ny=ny,
        #                                                             nq_pcer_dim=self._integration_order)
            
        #     weighted_dobj = weighted_dobj_local[local_to_global] # (NC*NQ, )

        #     integration_weights = self._integration_weights[local_to_global] # (NC*NQ, )

        #     numerator_global = self._H.matmul(weighted_dobj) # (NC*NQ, )
        #     numerator = numerator_global[global_to_local] # (NC, NQ)
            
        #     denominator_global = self._H.matmul(integration_weights) # (NC*NQ, )
        #     denominator = denominator_global[global_to_local] # (NC, NQ)

        # con_grad = bm.set_at(con_grad, slice(None), numerator / denominator)

        # return con_grad


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