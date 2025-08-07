from typing import Optional, Literal, Union

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import SimplexMesh, TensorMesh
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
        self._scalar_space = self._analyzer._scalar_space
        self._interpolation_scheme = self._analyzer._interpolation_scheme
        self._density_location = self._interpolation_scheme._density_location
        self._integration_order = self._analyzer._integration_order


    #####################################################################################################
    # 核心方法
    #####################################################################################################

    def fun(self, 
            physical_density: Function, 
            displacement: Optional[Function] = None,
        ) -> float:
        """计算体积约束函数值"""

        g = self._compute_volume(physical_density=physical_density)
        g0 = self._volume_fraction * self._compute_volume(physical_density=None)
        gneq = g - g0

        return gneq
    
    def jac(self, 
            physical_density: Function, 
            displacement: Optional[Function] = None,
            diff_mode: Literal["auto", "manual"] = "manual"
        ) -> TensorLike:
        """计算体积约束函数的梯度 (灵敏度)"""

        if diff_mode == "manual":
            return self._manual_differentiation(physical_density, displacement)
        elif diff_mode == "auto":  
            return self._auto_differentiation(physical_density, displacement)
        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        
    #####################################################################################################
    # 外部调用方法
    #####################################################################################################
        
    def get_volume_fraction(self, physical_density: Function) -> float:
        """计算当前设计的体积分数"""

        current_volume = self._compute_volume(physical_density=physical_density)            
        total_volume = self._compute_volume(physical_density=None)

        volume_fraction = current_volume / total_volume

        return volume_fraction


    #####################################################################################################
    # 内部方法
    #####################################################################################################

    def _compute_volume(self, physical_density: Union[Function, TensorLike] = None) -> float:
        """计算当前设计的体积"""

        if physical_density is None:

            cell_measure = self._mesh.entity_measure('cell')
            current_volume = bm.sum(cell_measure)

        else:

            NC = self._mesh.number_of_cells()
            gdof = physical_density.shape[0]

            if self._density_location == 'element':

                cell_measure = self._mesh.entity_measure('cell')
                current_volume = bm.einsum('c, c -> ', cell_measure, physical_density[:])

                return current_volume

            elif self._density_location == 'gauss_integration_point':

                qf = self._mesh.quadrature_formula(self._integration_order)
                bcs, ws = qf.get_quadrature_points_and_weights()

                if isinstance(self._mesh, SimplexMesh):
                    cm = self._mesh.entity_measure('cell')
                    current_volume = bm.einsum('q, cq, c -> ', ws, physical_density, cm)
                
                elif isinstance(self._mesh, TensorMesh):
                    J = self._mesh.jacobi_matrix(bcs)
                    detJ = bm.linalg.det(J)
                    current_volume = bm.einsum('q, cq, cq -> ', ws, physical_density, detJ)

            elif self._density_location == 'lagrange_interpolation_point':

                qf = self._mesh.quadrature_formula(self._integration_order)
                bcs, ws = qf.get_quadrature_points_and_weights()

                physical_density_gauss = physical_density(bcs) 

                if isinstance(self._mesh, SimplexMesh):
                    cm = self._mesh.entity_measure('cell')
                    current_volume = bm.einsum('q, cq, c -> ', ws, physical_density_gauss, cm)
                
                elif isinstance(self._mesh, TensorMesh):
                    J = self._mesh.jacobi_matrix(bcs)
                    detJ = bm.linalg.det(J)
                    current_volume = bm.einsum('q, cq, cq -> ', ws, physical_density_gauss, detJ)

            else:
                raise ValueError(f"Unexpected physical_density shape/type: {type(physical_density)}, shape={physical_density.shape}")
            
        return current_volume


    def _manual_differentiation(self, 
            physical_density: Function, 
            displacement: Optional[Function] = None
        ) -> TensorLike:
        """手动计算目标函数梯度"""

        cell_measure = self._mesh.entity_measure('cell')

        if self._density_location == 'element':
        
            dg = bm.copy(cell_measure)

        elif self._density_location == 'gauss_integration_point':

            qf = self._mesh.quadrature_formula(q=self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            if isinstance(self._mesh, SimplexMesh):
                dg = bm.einsum('q, c -> cq', ws, cell_measure)
            elif isinstance(self._mesh, TensorMesh):
                J = self._mesh.jacobi_matrix(bcs)
                detJ = bm.linalg.det(J)
                dg = bm.einsum('q, cq -> cq', ws, detJ)

        else:
            raise ValueError(f"Unsupported density_location: {self._density_location}")

        return dg