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
    # 属性访问器
    #####################################################################################################

    @property
    def volume_fraction(self) -> float:
        """获取体积分数"""
        return self._volume_fraction


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

            return current_volume

        else:

            if self._density_location == 'element':

                cell_measure = self._mesh.entity_measure('cell')
                current_volume = bm.einsum('c, c -> ', cell_measure, physical_density[:])

                return current_volume
            
            elif self._density_location == 'lagrange_interpolation_point':

                space = physical_density.space

                qf = self._mesh.quadrature_formula(self._integration_order)
                bcs, ws = qf.get_quadrature_points_and_weights()

                rho_q = physical_density(bcs) 

                if isinstance(self._mesh, SimplexMesh):
                    cm = self._mesh.entity_measure('cell')
                    current_volume = bm.einsum('q, cq, c -> ', ws, rho_q, cm)
                
                elif isinstance(self._mesh, TensorMesh):
                    J = self._mesh.jacobi_matrix(bcs)
                    detJ = bm.linalg.det(J)
                    current_volume = bm.einsum('q, cq, cq -> ', ws, rho_q, detJ)

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

                return current_volume

            elif self._density_location == 'density_subelement_gauss_point':

                NC = self._mesh.number_of_cells()
                NQ = physical_density.shape[1]  # 子单元数量

                cell_measure = self._mesh.entity_measure('cell')
                subcell_measure = cell_measure[:, None] / NQ
                subcell_measure = bm.broadcast_to(subcell_measure, (NC, NQ))
                current_volume = bm.einsum('cq, cq -> ', subcell_measure, physical_density[:])
                
                return current_volume

            else:
                raise ValueError(f"Unexpected physical_density shape/type: {type(physical_density)}, shape={physical_density.shape}")

    def _manual_differentiation(self, 
            physical_density: Function, 
            displacement: Optional[Function] = None
        ) -> TensorLike:
        """手动计算目标函数梯度"""

        cell_measure = self._mesh.entity_measure('cell')

        if self._density_location == 'element':
        
            dg = bm.copy(cell_measure)

            return dg
        
        elif self._density_location == 'lagrange_interpolation_point':

            qf = self._mesh.quadrature_formula(self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()      

            N = self._mesh.shape_function(bcs)                    # (NQ, LDOF)

            if isinstance(self._mesh, SimplexMesh):
                dg_e = bm.einsum('q, c, ql -> cl', ws, cell_measure, N) # (NC, LDOF)

            elif isinstance(self._mesh, TensorMesh):

                J = self._mesh.jacobi_matrix(bcs)                     # (NC, NQ, GD, GD)
                detJ = bm.linalg.det(J)                               # (NC, NQ)
                dg_e = bm.einsum('q, cq, ql -> cl', ws, detJ, N)      # (NC, LDOF)
            
            lagrange_space = physical_density.space
            gdof = lagrange_space.number_of_global_dofs()  
            cell2dof = lagrange_space.cell_to_dof() # (NC, LDOF)

            dg = bm.zeros((gdof,), dtype=bm.float64, device=self._mesh.device)
            dg = bm.add_at(dg, cell2dof.reshape(-1), dg_e.reshape(-1)) # (GDOF,)

            return dg

        elif self._density_location == 'gauss_integration_point':

            qf = self._mesh.quadrature_formula(q=self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            if isinstance(self._mesh, SimplexMesh):
                dg = bm.einsum('q, c -> cq', ws, cell_measure)

            elif isinstance(self._mesh, TensorMesh):
                J = self._mesh.jacobi_matrix(bcs)
                detJ = bm.linalg.det(J)
                dg = bm.einsum('q, cq -> cq', ws, detJ)

            return dg

        elif self._density_location == 'density_subelement_gauss_point':
            
            NC = self._mesh.number_of_cells()
            NQ = physical_density.shape[1]  # 子单元数量
            
            subcell_measure = cell_measure[:, None] / NQ
            dg = bm.broadcast_to(subcell_measure, (NC, NQ))

            return dg

        else:
            raise ValueError(f"Unsupported density_location: {self._density_location}")

