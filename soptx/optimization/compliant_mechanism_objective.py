from typing import Optional, Union, Literal
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function
from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from ..utils.base_logged import BaseLogged

class CompliantMechanismObjective(BaseLogged):
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._pde = analyzer.pde
        self._interpolation_scheme = analyzer._interpolation_scheme

    def fun(self, 
            density: Union[Function, TensorLike], 
            displacement: Optional[TensorLike] = None,
           ) -> float:
        """
        计算柔顺机械的目标函数值 (即输出点的位移 u_out).
        """
        if self._analyzer.__class__ in [LagrangeFEMAnalyzer]:
            #* 拉格朗日位移有限元 *#
            if displacement is None:
                U = self._analyzer.solve_displacement(rho_val=density, adjoint=True) 
            else:
                U = displacement
            
            space_uh = self._analyzer.tensor_space

            threshold_dout = self._pde.is_dout_boundary()
            isBdTDof = space_uh.is_boundary_dof(threshold=threshold_dout, method='interp')

            uh_real = U[:, 0]
            u_out = uh_real[isBdTDof]

        elif self._analyzer.__class__ in [HuZhangMFEMAnalyzer]:
            #* 胡张应力位移混合有限元 *#
            pass
        
        return u_out.item()
    
    def jac(self, 
            density: Union[Function, TensorLike], 
            displacement: Optional[Function] = None,
            diff_mode: Literal["auto", "manual"] = "manual"
        ) -> TensorLike:
        """计算柔顺机械相对于物理密度的灵敏度"""

        if diff_mode == "manual":
            return self._manual_differentiation(density=density, displacement=displacement)

        elif diff_mode == "auto": 
            return self._auto_differentiation(density=density, displacement=displacement)

        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)

    def _manual_differentiation(self, 
                                density: Union[Function, TensorLike], 
                                displacement: Optional[Function] = None
                            ) -> TensorLike:
        """手动计算柔顺机械相对于物理密度的灵敏度"""

        density_location = self._interpolation_scheme.density_location

        if self._analyzer.__class__ in [LagrangeFEMAnalyzer]:
            #* 拉格朗日位移有限元 *#
            if displacement is None:
                uh = self._analyzer.solve_displacement(rho_val=density, adjoint=True)
            else:
                uh = displacement
            
            space_uh = self._analyzer._tensor_space
            cell2dof = space_uh.cell_to_dof()
            uhe = uh[cell2dof, 0]
            lambdahe = uh[cell2dof, 1]
            
            diff_KE = self._analyzer.get_stiffness_matrix_derivative(rho_val=density)

            if density_location in ['element']:
                dc = bm.einsum('ci, cij, cj -> c', lambdahe, diff_KE, uhe) # (NC, )

                return dc[:]
            
            elif density_location in ['element_multiresolution']:
                pass
            
            elif density_location in ['node']:
                pass
            
            elif density_location in ['node_multiresolution']:
                pass
            
            else:
                error_msg = f"Unknown density location: {density_location}"
                self._log_error(error_msg)