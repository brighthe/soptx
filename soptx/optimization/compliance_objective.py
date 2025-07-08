from typing import Optional, Literal

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
        self._tensor_space = self._analyzer._tensor_space
        self._top_material = self._analyzer._material
        self._interpolation_scheme = self._top_material._interpolation_scheme

    def fun(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None,
        ) -> float:
        """计算柔顺度目标函数值"""

        self._analyzer.density_distribution = density_distribution

        if displacement is None:
            uh = self._analyzer.solve()
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
        
    def _manual_differentiation(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> Function:
        """手动计算目标函数梯度"""

        # 设置新的密度分布
        self._analyzer.density_distribution = density_distribution

        if displacement is None:
            uh = self._analyzer.solve()
        else:
            uh = displacement
        
        cell2dof = self._tensor_space.cell_to_dof()
        uhe = uh[cell2dof]

        if density_distribution.shape == (NC, ):
            interpolate_derivative = self._interpolation_scheme.interpolate_derivative(
                                            base_material=self._top_material._base_material, 
                                            density_distribution=density_distribution
                                        )
            ke0 = self._analyzer.get_base_local_stiffness_matrix()
            dc = bm.einsum('c, ci, cik, ck -> c', interpolate_derivative, uhe, ke0, uhe)

        elif density_distribution.shape == (NC, NQ):
            pass

        return dc
        
    