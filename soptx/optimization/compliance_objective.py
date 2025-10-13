from typing import Optional, Literal, Union

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged

class ComplianceObjective(BaseLogged):
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                state_variable: Literal['u', 'sigma'] = 'u',
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._state_variable = state_variable
        # self._tensor_space = analyzer._tensor_space
        self._interpolation_scheme = analyzer._interpolation_scheme

    def fun(self, 
            density: Union[Function, TensorLike], 
            displacement: Optional[Function] = None,
            stress: Optional[Function] = None
        ) -> float:
        """计算柔顺度目标函数值"""

        if self._analyzer.__class__ in [LagrangeFEMAnalyzer]:
            # 拉格朗日位移有限元
            if displacement is None:
                uh = self._analyzer.solve_displacement(rho_val=density)
            else:
                uh = displacement
        
            F = self._analyzer.force_vector
            c = bm.einsum('i, i ->', uh[:], F)

            # K = self._analyzer.stiffness_matrix
            # Ku = K.matmul(uh[:])
            # c = bm.einsum('i, i ->', uh[:], Ku)

        elif self._analyzer.__class__ in [HuZhangMFEMAnalyzer]:
            # 胡张应力位移混合有限元
            if displacement is None and stress is None:
                sigmah, uh = self._analyzer.solve_displacement(rho_val=density)
            else:
                sigmah, uh = displacement, stress
                
            if self._state_variable == 'u':
                B_u_sigma = self._analyzer.get_B_u_sigma()
                A_inv = self._analyzer.get_A_sigma_sigma_inverse(rho_val=density)
                f_sigma = self._analyzer.get_f_sigma()
                f_u = self._analyzer.get_f_u()

                F = B_u_sigma @ A_inv @ f_sigma - f_u
                c = bm.einsum('i, i ->', uh[:], F)

                # B_sigma_u = self._analyzer.get_B_sigma_u()
                # K = B_u_sigma @ A_inv @ B_sigma_u
                # Ku = K.matmul(uh[:])
                # c = bm.einsum('i, i ->', uh[:], Ku)

            elif self._state_variable == 'sigma':
                A = self._analyzer.get_stress_matrix(rho_val=density)
                Asigmah = A.matmul(sigmah[:])
                c = bm.einsum('i, i ->', sigmah[:], Asigmah)

            else:
                error_msg = f"Unknown state_variable: {self._state_variable}"
                self._log_error(error_msg)

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

        density_location = self._interpolation_scheme.density_location

        if self._analyzer.__class__ in [LagrangeFEMAnalyzer]:
            # 拉格朗日位移有限元
            if displacement is None:
                uh = self._analyzer.solve_displacement(rho_val=density)
            else:
                uh = displacement
            
            space_uh = self._analyzer._tensor_space
            cell2dof = space_uh.cell_to_dof()
            uhe = uh[cell2dof]

            diff_KE = self._analyzer.get_stiffness_matrix_derivative(rho_val=density)

            if density_location in ['element']:
                dc = -bm.einsum('ci, cij, cj -> c', uhe, diff_KE, uhe) # (NC, )

                return dc[:]
            
            elif density_location in ['element_multiresolution']:
                dc = -bm.einsum('ci, cnij, cj -> cn', uhe, diff_KE, uhe) # (NC, n_sub)

                return dc[:]
            
            elif density_location in ['node']:
                dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_KE, uhe) # (NC, NCN)

                mesh = space_uh.mesh
                cell2node = mesh.cell_to_node() # (NC, NCN)
                NN = mesh.number_of_nodes()

                dc = bm.zeros((NN, ), dtype=uhe.dtype, device=uhe.device) # (NN, )
                dc = bm.add_at(dc, cell2node.reshape(-1), dc_e.reshape(-1))

                return dc[:]
            
            elif density_location in ['node_multiresolution']:
                dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_KE, uhe) # (NC, NCN)

                density_mesh = density.space.mesh
                cell2node = density_mesh.cell_to_node() # (NC, NCN)
                NN = density_mesh.number_of_nodes()

                dc = bm.zeros((NN, ), dtype=uhe.dtype, device=uhe.device) # (NN, )
                dc = bm.add_at(dc, cell2node.reshape(-1), dc_e.reshape(-1))

                return dc[:]
            
            else:
                error_msg = f"Unknown density location: {density_location}"
                self._log_error(error_msg)

        elif self._analyzer.__class__ in [HuZhangMFEMAnalyzer]:
            # 胡张应力位移混合有限元
            if displacement is None:
                sigmah, uh = self._analyzer.solve_displacement(rho_val=density)
            else:
                sigmah, uh = displacement, None
            
            space_sigmah = self._analyzer._huzhang_space
            cell2dof = space_sigmah.cell_to_dof()
            sigmahe = sigmah[cell2dof] # (NC, TLDOF_sigma)

            diff_AE = self._analyzer.get_local_stress_matrix_derivative(rho_val=density) # (NC, TLDOF_sigma, TLDOF_sigma)

            if density_location in ['element']:
                dc = -bm.einsum('ci, cij, cj -> c', sigmahe, diff_AE, sigmahe) # (NC, )

                return dc[:]
            
            elif density_location in ['node']:
                raise NotImplementedError("节点密度尚未实现")
            
            elif density_location in ['element_multiresolution']:
                raise NotImplementedError("多分辨率单元密度尚未实现")
            
            elif density_location in ['node_multiresolution']:
                raise NotImplementedError("多分辨率节点密度尚未实现")

    
    def _auto_differentiation(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> Function:
        # TODO 待实现
        pass
        
    