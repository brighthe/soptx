from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function, TensorFunctionSpace

from ..analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from ..interpolation.topology_optimization_material import TopologyOptimizationMaterial
from ..utils.base_logged import BaseLogged

class ComplianceObjective(BaseLogged):
    def __init__(self,
                analyzer: LagrangeFEMAnalyzer,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        if not isinstance(analyzer.material, TopologyOptimizationMaterial):
            error_msg = (
                f"ComplianceObjective requires TopologyOptimizationMaterial, "
                f"got {type(analyzer.material).__name__}. "
            )
            self._log_error(error_msg) 
            raise TypeError(error_msg)  

        self._analyzer = analyzer
        self._tensor_space: TensorFunctionSpace = self._analyzer.tensor_space
        self._top_material: TopologyOptimizationMaterial = self._analyzer.material
        self._base_material = self._top_material.base_material
        self._interpolation_scheme = self._top_material.interpolation_scheme

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
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> Function:
        """手动计算目标函数梯度"""

        self._top_material.density_distribution = density_distribution

        if displacement is None:
            uh = self._analyzer.solve()
        else:
            uh = displacement
        
        cell2dof = self._tensor_space.cell_to_dof()
        uhe = uh[cell2dof]

        density_location = self._top_material.density_location
        diff_ke = self._analyzer.get_stiffness_matrix__derivative()

        if density_location == 'element':
            dc = -bm.einsum('ci, cij, cj -> c', uhe, diff_ke, uhe)

            self._log_info(f"ComplianceObjective derivative: dc shape is (NC, ) = {dc.shape}")


        elif density_location == 'element_gauss_integrate_point':
            dc = -bm.einsum('ci, cqij, cj -> cq', uhe, diff_ke, uhe)

            self._log_info(f"ComplianceObjective derivative: dc shape is (NC, NQ) = {dc.shape}")

        return dc
    
    def _auto_differentiation(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> Function:
        # TODO 待实现
        pass
        
    