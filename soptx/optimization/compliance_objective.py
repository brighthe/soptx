from typing import Optional, Literal, Union

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from ..analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from ..utils.base_logged import BaseLogged

class ComplianceObjective(BaseLogged):
    def __init__(self,
                analyzer: LagrangeFEMAnalyzer,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._tensor_space = analyzer._tensor_space
        self._interpolation_scheme = analyzer._interpolation_scheme

    def fun(self, 
            density: Union[Function, TensorLike], 
            displacement: Optional[Function] = None,
        ) -> float:
        """计算柔顺度目标函数值"""

        if displacement is None:
            uh = self._analyzer.solve_displacement(rho_val=density)
        else:
            uh = displacement

        F = self._analyzer.force_vector
        c = bm.einsum('i, i ->', uh[:], F)

        # NOTE uKu 更低效
        # K = self._analyzer.stiffness_matrix
        # Ku = K.matmul(uh[:])
        # c = bm.einsum('i, i ->', uh[:], Ku)

        return c
    
    def jac(self, 
            density: Union[Function, TensorLike], 
            displacement: Optional[Function] = None,
            diff_mode: Literal["auto", "manual"] = "manual"
        ) -> TensorLike:
        """计算柔顺度目标函数相对于物理密度的灵敏度"""

        if diff_mode == "manual":

            return self._manual_differentiation(density=density, displacement=displacement)

        elif diff_mode == "auto": 

            return self._auto_differentiation(density=density, displacement=displacement)

        else:
        
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)
        
    
    #####################################################################################################
    # 内部方法
    #####################################################################################################
        
    def _manual_differentiation(self, 
                                density: Union[Function, TensorLike], 
                                displacement: Optional[Function] = None
                            ) -> TensorLike:
        """手动计算柔顺度目标函数相对于物理密度的灵敏度"""

        if displacement is None:
            uh = self._analyzer.solve_displacement(rho_val=density)
        else:
            uh = displacement
        
        cell2dof = self._tensor_space.cell_to_dof()
        uhe = uh[cell2dof]

        diff_ke = self._analyzer.get_stiffness_matrix_derivative(rho_val=density)

        density_location = self._interpolation_scheme.density_location

        if density_location in ['element']:
            
            dc = -bm.einsum('ci, cij, cj -> c', uhe, diff_ke, uhe) # (NC, )

            # if bm.any(dc > 1e-12):
            #     self._log_error(f"目标函数关于物理密度的灵敏度中存在正值, 可能导致目标函数上升")

            return dc[:]
        
        elif density_location in ['element_multiresolution']:

            dc = -bm.einsum('ci, cnij, cj -> cn', uhe, diff_ke, uhe) # (NC, n_sub)

            # if bm.any(dc > 1e-12):
            #     self._log_error(f"目标函数关于物理密度的灵敏度中存在正值, 可能导致目标函数上升")

            return dc[:]
        
        elif density_location in ['node']:

            dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_ke, uhe) # (NC, NCN)

            mesh = self._tensor_space.mesh
            cell2node = mesh.cell_to_node() # (NC, NCN)
            NN = mesh.number_of_nodes()

            dc = bm.zeros((NN, ), dtype=uhe.dtype, device=uhe.device) # (NN, )
            dc = bm.add_at(dc, cell2node.reshape(-1), dc_e.reshape(-1))

            if bm.any(dc > 1e-12):
                self._log_error(f"目标函数关于物理密度的灵敏度中存在正值, 可能导致目标函数上升")

            return dc[:]
        
        elif density_location in ['node_multiresolution']:

            dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_ke, uhe) # (NC, NCN)

            density_mesh = density.space.mesh
            cell2node = density_mesh.cell_to_node() # (NC, NCN)
            NN = density_mesh.number_of_nodes()

            dc = bm.zeros((NN, ), dtype=uhe.dtype, device=uhe.device) # (NN, )
            dc = bm.add_at(dc, cell2node.reshape(-1), dc_e.reshape(-1))

            if bm.any(dc > 1e-12):
                self._log_error(f"目标函数关于物理密度的灵敏度中存在正值, 可能导致目标函数上升")

            return dc[:]
        
        else:
            
            error_msg = f"Unknown density location: {density_location}"
            self._log_error(error_msg)
    
    def _auto_differentiation(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> Function:
        # TODO 待实现
        pass
        
    