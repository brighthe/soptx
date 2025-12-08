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
        self._integration_order = self._analyzer._integration_order
        self._density_location = self._interpolation_scheme._density_location        

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
            density: Function, 
            displacement: Optional[Function] = None,
        ) -> float:
        """计算体积分数约束函数值"""

        g = self._compute_volume(density=density)
        g0 = self._volume_fraction * self._compute_volume(density=None)
        gneq = g - g0

        return gneq
    
    def jac(self, 
            density: Function, 
            displacement: Optional[Function] = None,
            diff_mode: Literal["auto", "manual"] = "manual"
        ) -> TensorLike:
        """计算体积约束函数相对于物理密度的灵敏度"""

        if diff_mode == "manual":
            return self._manual_differentiation(density, displacement)
        
        elif diff_mode == "auto":  
            return self._auto_differentiation(density, displacement)
        
        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)

        
    #####################################################################################################
    # 外部调用方法
    #####################################################################################################
        
    def get_volume_fraction(self, density: Function) -> float:
        """计算当前设计的体积分数"""

        current_volume = self._compute_volume(density=density)            
        total_volume = self._compute_volume(density=None)

        volume_fraction = current_volume / total_volume

        return volume_fraction


    #####################################################################################################
    # 内部方法
    #####################################################################################################

    def _compute_volume(self, density: Union[Function, TensorLike] = None) -> float:
        """计算当前设计的体积"""

        if density is None:
            cell_measure = self._mesh.entity_measure('cell')
            current_volume = bm.sum(cell_measure)

            return current_volume

        else:
            if self._density_location in ['element']:
                rho_element = density  # (NC, )

                cell_measure = self._mesh.entity_measure('cell')
                current_volume = bm.einsum('c, c -> ', cell_measure, rho_element[:])

                return current_volume
            
            elif self._density_location in ['element_multiresolution']:

                rho_sub_element = density # (NC, n_sub)

                NC, n_sub = rho_sub_element.shape
                cell_measure = self._mesh.entity_measure('cell')
                sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)

                current_volume = bm.einsum('cn, cn -> ', sub_cm, rho_sub_element[:])

                return current_volume

            elif self._density_location in ['node']:
                #* 标准节点密度表征下的体积计算
                # 计算单元积分点处的重心坐标
                qf = self._mesh.quadrature_formula(q=self._integration_order)
                # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
                bcs, ws = qf.get_quadrature_points_and_weights()

                rho_q = density(bcs) # (NC, NQ)

                if isinstance(self._mesh, SimplexMesh):
                    cm = self._mesh.entity_measure('cell')
                    current_volume = bm.einsum('q, cq, c -> ', ws, rho_q, cm)
                
                elif isinstance(self._mesh, TensorMesh):
                    J = self._mesh.jacobi_matrix(bcs)
                    detJ = bm.abs(bm.linalg.det(J))
                    current_volume = bm.einsum('q, cq, cq -> ', ws, rho_q, detJ)

                #* 简化节点密度表征下的体积计算
                # cell_measure = self._mesh.entity_measure('cell')
                # total_volume = bm.sum(cell_measure)

                # rho_node = density[:] # (NN, )
                # avg_rho = bm.sum(rho_node) / rho_node.shape[0]
                # current_volume = total_volume * avg_rho

                return current_volume
            
            elif self._density_location in ['node_multiresolution']:

                rho_sub_q = density # (NC, n_sub, NQ)
                NC, n_sub, NQ = rho_sub_q.shape

                if isinstance(self._mesh, SimplexMesh):
                    cell_measure = self._mesh.entity_measure('cell')
                    sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)
                    current_volume = bm.einsum('q, cnq, cn -> ', ws, rho_sub_q, sub_cm)
                
                elif isinstance(self._mesh, TensorMesh):
                    # 计算位移单元积分点处的重心坐标
                    qf_e = self._mesh.quadrature_formula(q=self._integration_order)
                    # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
                    bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()

                    # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
                    from soptx.analysis.utils import map_bcs_to_sub_elements
                    # bcs_eg.shape = ( (n_sub, NQ, GD), (n_sub, NQ, GD) ), ws_e.shape = (NQ, )
                    bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
                    bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

                    detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
                    for s_idx in range(n_sub):
                        sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ, GD), (NQ, GD))

                        J_sub = self._mesh.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                        detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                        detJ_eg[:, s_idx, :] = detJ_sub

                    current_volume = bm.einsum('q, cnq, cnq -> ', ws_e, rho_sub_q, detJ_eg)

            else:

                error_msg = f"Unknown density_location: {self._density_location}"
                self._log_error(error_msg)

    def _manual_differentiation(self, 
                                density: Union[Function, TensorLike], 
                                displacement: Optional[Function] = None
                            ) -> TensorLike:
        """手动计算体积约束函数相对于物理密度的灵敏度"""

        if self._density_location in ['element']:
            cell_measure = self._mesh.entity_measure('cell')
            dg = bm.copy(cell_measure) # (NC, )

            return dg
        
        elif self._density_location in ['element_multiresolution']:
            
            NC, n_sub = density.shape
            cell_measure = self._mesh.entity_measure('cell')
            sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / n_sub
            
            dg = bm.copy(sub_cm) # (NC, n_sub)

            return dg
        
        elif self._density_location in ['node']:
            #* 标准节点密度表征下的体积分数梯度计算
            mesh = self._mesh
            density_space = density.space

            qf = mesh.quadrature_formula(self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            phi = density_space.basis(bcs)[0] # (NQ, NCN)

            if isinstance(mesh, SimplexMesh):
                cell_measure = self._mesh.entity_measure('cell')
                dg_e = bm.einsum('q, c, ql -> cl', ws, cell_measure, phi) # (NC, NCN)
            elif isinstance(mesh, TensorMesh):
                J = mesh.jacobi_matrix(bcs)                           # (NC, NQ, GD, GD)
                detJ = bm.abs(bm.linalg.det(J))                       # (NC, NQ)
                dg_e = bm.einsum('q, cq, ql -> cl', ws, detJ, phi)    # (NC, NCN)

            NN = mesh.number_of_nodes()
            cell2node = mesh.cell_to_node() # (NC, NCN)

            dg = bm.zeros((NN, ), dtype=bm.float64, device=self._mesh.device) # (NN, )
            dg = bm.add_at(dg, cell2node.reshape(-1), dg_e.reshape(-1)) # (NN, )
            
            #* 简化节点密度表征下体积分数梯度计算
            # mesh = self._mesh
            # NN = mesh.number_of_nodes()

            # dg = bm.full((NN, ), 1.0 / NN, dtype=bm.float64, device=mesh.device)

            return dg

        elif self._density_location in ['node_multiresolution']:
            pass
        
        elif self._density_location in ['lagrange_interpolation_point', 
                                        'berstein_interpolation_point', 
                                        'shepard_interpolation_point']:

            qf = self._mesh.quadrature_formula(self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            density_space = physical_density.space

            # space_degree = density_space.p
            # N = self._mesh.shape_function(bcs=bcs, p=space_degree)          # (NQ, LDOF)
            
            phi_rho = density_space.basis(bcs)[0] # (NQ, LDOF_rho)

            if isinstance(self._mesh, SimplexMesh):
                dg_e = bm.einsum('q, c, ql -> cl', ws, cell_measure, phi_rho) # (NC, LDOF_rho)

            elif isinstance(self._mesh, TensorMesh):

                J = self._mesh.jacobi_matrix(bcs)                           # (NC, NQ, GD, GD)
                detJ = bm.abs(bm.linalg.det(J))                             # (NC, NQ)
                dg_e = bm.einsum('q, cq, ql -> cl', ws, detJ, phi_rho)      # (NC, LDOF_rho)
            
            gdof = density_space.number_of_global_dofs()  
            cell2dof = density_space.cell_to_dof() # (NC, LDOF)

            dg = bm.zeros((gdof, ), dtype=bm.float64, device=self._mesh.device)
            dg = bm.add_at(dg, cell2dof.reshape(-1), dg_e.reshape(-1)) # (GDOF,)

            return dg

        elif self._density_location in ['gauss_integration_point']:

            qf = self._mesh.quadrature_formula(q=self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            if isinstance(self._mesh, SimplexMesh):
                dg = bm.einsum('q, c -> cq', ws, cell_measure)

            elif isinstance(self._mesh, TensorMesh):
                J = self._mesh.jacobi_matrix(bcs)
                detJ = bm.abs(bm.linalg.det(J)) 
                dg = bm.einsum('q, cq -> cq', ws, detJ)  

            return dg # (NC, NQ)

        elif self._density_location == 'density_subelement_gauss_point':
            
            NC = self._mesh.number_of_cells()
            NQ = physical_density.shape[1]  # 子单元数量
            
            subcell_measure = cell_measure[:, None] / NQ
            dg = bm.broadcast_to(subcell_measure, (NC, NQ))

            return dg

        else:
            raise ValueError(f"Unsupported density_location: {self._density_location}")

