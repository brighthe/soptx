from typing import Optional, Literal, Union, Tuple, Dict
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from ..utils.base_logged import BaseLogged

class StressConstraint(BaseLogged):
    """
    应力约束类

    实现基于聚类 P-norm 的全局应力聚合约束
    """
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                stress_limit: float,
                p_norm_factor: float = 8.0,
                n_clusters: int = 1,
                recluster_freq: int = 5,          
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit

        self._p_norm_factor = p_norm_factor
        self._n_clusters = n_clusters
        self._recluster_freq = recluster_freq
        
        # 缓存一些常用对象
        self._mesh = self._analyzer.disp_mesh
        self._space_uh = self._analyzer.tensor_space
        self._material = self._analyzer.material
        self._interpolation_scheme = self._analyzer.interpolation_scheme

        self._density_location = self._interpolation_scheme.density_location

        if isinstance(self._analyzer, LagrangeFEMAnalyzer):
            self._B = self._compute_strain_displacement_matrix()
        else:
            self._B = None

    def _compute_strain_displacement_matrix(self) -> TensorLike:
        """构建并缓存应变-位移矩阵 B"""
        density_location = self._density_location
        q = 1

        if density_location in ['element']:
            qf = self._mesh.quadrature_formula(q)
            bcs, _ = qf.get_quadrature_points_and_weights()
            gphi = self._analyzer.scalar_space.grad_basis(bcs, variable='x') # (NC, NQ, LDOF, GD)
            B = self._material.strain_displacement_matrix(
                                            dof_priority=self._space_uh.dof_priority, 
                                            gphi=gphi
                                        ) # (NC, NQ, NS, TLDOF)
            
        elif density_location in ['element_multiresolution']:
            nx_u, ny_u = self._mesh.meshdata['nx'], self._mesh.meshdata['ny']
            n_sub = 4
            from soptx.interpolation.utils import calculate_multiresolution_gphi_eg, reshape_multiresolution_data_inverse
            gphi_eg_reshaped = calculate_multiresolution_gphi_eg(
                                            s_space_u=self._analyzer.scalar_space,
                                            q=q,
                                            n_sub=n_sub) # (NC*n_sub, NQ, LDOF, GD)
            B_reshaped = self._material.strain_displacement_matrix(
                                                dof_priority=self._space_uh.dof_priority, 
                                                gphi=gphi_eg_reshaped
                                            ) # (NC*n_sub, NQ, NS, TLDOF)
            B = reshape_multiresolution_data_inverse(nx=nx_u, ny=ny_u, 
                                                    data_flat=B_reshaped, 
                                                    n_sub=n_sub) # (NC, n_sub, NQ, NS, TLDOF)
            
        else:
            self._log_error(f"Unsupported density location: {self._density_location}")
        
        return B

    def _compute_stress_state(self, density: TensorLike, state: dict) -> Dict[str, TensorLike]:
        """根据当前状态变量计算应力信息"""
        if isinstance(self._analyzer, LagrangeFEMAnalyzer):
            if state is None:
                state = self._analyzer.solve_state(rho_val=density)
        
            uh = state['displacement']
            cell2dof = self._space_uh.cell_to_dof()
            uh_e = uh[cell2dof]
            
            # 计算实体应力 (与密度无关)
            stress_solid = self._material.calculate_stress_vector(self._B, uh_e)
            
            # 计算惩罚后的应力
            stress_penalized = self._interpolation_scheme.interpolate_stress(
                                                                stress_solid=stress_solid,
                                                                rho_val=density
                                                            )
            
            # 计算 von Mises 应力
            von_mises = self._material.calculate_von_mises_stress(stress_vector=stress_penalized)

            return {
                'stress_solid': stress_solid,
                'stress_penalized': stress_penalized,
                'von_mises': von_mises
            }
        
        elif isinstance(self._analyzer, HuZhangMFEMAnalyzer):
            if state is None:
                state = self._analyzer.solve_state(rho_val=density)
            
            stress = state['stress'] # (GDOF, )

            # 转换为积分点应力
            stress_at_quad = self._analyzer.extract_stress_at_quadrature_points(stress_dof=stress)  # (NC, NQ, NS)
            
            # 计算 von Mises 应力
            von_mises = self._material.calculate_von_mises_stress(stress_vector=stress_at_quad)
            
            return {
                'von_mises': von_mises
                }
        
        else:
            self._log_error("State dictionary must contain either 'stress' or 'displacement'.")
        
    
    def _compute_clustered_pnorm(self, sigma_vm: TensorLike) -> float:
        """计算聚类 P-norm 应力约束值

        Parameters
        ----------
        sigma_vm: von Mises 应力场
            - STOP: (NC, NQ)
            - MTOP: (NC, n_sub, NQ)
        """
        vals = sigma_vm.flatten()
        
        normalized_stress = vals / self._stress_limit
        term = (bm.maximum(normalized_stress, 0.0) + 1e-12) ** self._p_norm_factor
        
        # 聚类求和
        aggregated_sum = bm.zeros((self._n_clusters, ), dtype=vals.dtype, device=vals.device)
        bm.scatter_add(aggregated_sum, self._clustering_map, term)
        
        # Holmberg 平均化修正: sum / N_m
        mean_term = aggregated_sum / self._cluster_counts
        
        # 开 P 次方
        pnorm_values = mean_term ** (1.0 / self._p_norm_factor)
        
        # 约束值 g = PN - 1.0 <= 0
        return pnorm_values - 1.0

        
    def fun(self, 
            density: Union[Function, TensorLike], 
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """计算应力约束函数值"""
        stress_state = self._compute_stress_state(density, state)
        sigma_vm = stress_state['von_mises']

        val = self._compute_clustered_pnorm(sigma_vm)

        return val

    def jac(self, 
            density: Function, 
            state: dict,
            **kwargs
        ) -> TensorLike:
        """计算应力约束的灵敏度"""
        pass