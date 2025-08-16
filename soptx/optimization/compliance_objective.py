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
            density_distribution: Function, 
            displacement: Optional[Function] = None,
        ) -> float:
        """计算柔顺度目标函数值"""

        if displacement is None:
            uh = self._analyzer.solve_displacement(density_distribution=density_distribution)
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
            density_distribution: Function, 
            displacement: Optional[Function] = None,
            diff_mode: Literal["auto", "manual"] = "manual"
        ) -> TensorLike:
        """计算柔顺度目标函数的梯度 (灵敏度)"""

        if diff_mode == "manual":
            return self._manual_differentiation(density_distribution, displacement)
        elif diff_mode == "auto":  
            return self._auto_differentiation(density_distribution, displacement)
        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
    
    #####################################################################################################
    # 内部方法
    #####################################################################################################
        
    def _manual_differentiation(self, 
                                density_distribution: Union[Function, TensorLike], 
                                displacement: Optional[Function] = None
                            ) -> TensorLike:
        """手动计算目标函数梯度"""

        if displacement is None:
            uh = self._analyzer.solve_displacement(density_distribution=density_distribution)
        else:
            uh = displacement
        
        cell2dof = self._tensor_space.cell_to_dof()
        uhe = uh[cell2dof]

        diff_ke = self._analyzer.get_stiffness_matrix_derivative(density_distribution=density_distribution)

        density_location = self._interpolation_scheme.density_location

        if density_location == 'element':
            dc = -bm.einsum('ci, cij, cj -> c', uhe, diff_ke, uhe)

            self._log_info(f"ComplianceObjective derivative: dc shape is (NC, ) = {dc.shape}")

            return dc[:]
        
        elif density_location == 'lagrange_interpolation_point':
            dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_ke, uhe)

            density_space = density_distribution.space
            cell2dof = density_space.cell_to_dof()   # (NC, LDOF_rho)
            gdof_rho = density_space.number_of_global_dofs()

            dc = bm.zeros((gdof_rho,), dtype=uhe.dtype, device=uhe.device)
            dc = bm.add_at(dc, cell2dof.reshape(-1), dc_e.reshape(-1))

            self._log_info(f"ComplianceObjective derivative: dc shape is (GDOF_rho, ) = {dc.shape}")

            return dc[:]

        elif density_location == 'gauss_integration_point' or density_location == 'density_subelement_gauss_point':
            dc = -bm.einsum('ci, cqij, cj -> cq', uhe, diff_ke, uhe)

            self._log_info(f"ComplianceObjective derivative: dc shape is (NC, NQ) = {dc.shape}")
        
            return dc[:]
    
    def _auto_differentiation(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> Function:
        # TODO 待实现
        pass
        
    