from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from ..analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from ..utils.base_logged import BaseLogged


class VolumeConstraint(BaseLogged):
    def __init__(self,
                analyzer: LagrangeFEMAnalyzer,
                volume_fraction: float,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._volume_fraction = volume_fraction
        self._mesh = self._analyzer._mesh
        self._interpolation_scheme = self._analyzer._interpolation_scheme
        self._integrator_order = self._analyzer._integrator_order


    #####################################################################################################
    # 核心方法
    #####################################################################################################

    def fun(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None,
        ) -> float:
        """计算体积约束函数值"""

        density_location = self._interpolation_scheme.density_location

        cell_measure = self._mesh.entity_measure('cell')

        if density_location == 'element':

            g = bm.einsum('c, c -> ', cell_measure, density_distribution[:])
            g0 = self._volume_fraction * bm.sum(cell_measure)
            gneq = g - g0
            # gneq = bm.einsum('c, c -> ', cell_measure, rho) / \
            #         (self.volume_fraction * bm.sum(cell_measure)) - 1 # float

        elif density_location == 'element_gauss_integrate_point':

            qf = self._mesh.quadrature_formula(q=self._integrator_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            g = bm.einsum('c, q, cq -> ', cell_measure, ws, density_distribution[:])
            g0 = self._volume_fraction * bm.sum(cell_measure)
            gneq = g - g0

        return gneq
    
    def jac(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None,
            diff_mode: Literal["auto", "manual"] = "manual"
        ) -> TensorLike:
        """计算体积约束函数的梯度 (灵敏度)"""

        if diff_mode == "manual":
            return self._manual_differentiation(density_distribution, displacement)
        elif diff_mode == "auto":  
            return self._auto_differentiation(density_distribution, displacement)
        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
    def get_volume_fraction(self, density_distribution: Function) -> float:
        """计算当前设计的体积分数"""
        cell_measure = self._mesh.entity_measure('cell')
        current_volume = bm.einsum('c, c -> ', cell_measure, density_distribution[:])
        total_volume = bm.sum(cell_measure)
        volume_fraction = current_volume / total_volume
        
        return volume_fraction


    #####################################################################################################
    # 内部方法
    #####################################################################################################
        
    def _manual_differentiation(self, 
            density_distribution: Function, 
            displacement: Optional[Function] = None
        ) -> TensorLike:
        """手动计算目标函数梯度"""

        density_location = self._interpolation_scheme.density_location

        cell_measure = self._mesh.entity_measure('cell')

        if density_location == 'element':
        
            dg = bm.copy(cell_measure)

        elif density_location == 'element_gauss_integrate_point':

            qf = self._mesh.quadrature_formula(q=self._integrator_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            dg = bm.einsum('c, q -> cq', cell_measure, ws)

        return dg