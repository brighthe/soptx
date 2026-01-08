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
    实现基于 P-norm 的全局应力聚合约束
    """
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                stress_limit: float,
                p_norm_factor: float = 8.0,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit
        self._p_norm_factor = p_norm_factor
        
        # 缓存一些常用对象
        self._mesh = self._analyzer.mesh
        self._space_uh = self._analyzer.tensor_space
        self._material = self._analyzer.material
        self._interpolation_scheme = self._analyzer.interpolation_scheme

        self._density_location = self._interpolation_scheme.density_location

        q = 1


    def _compute_strain_displacement_matrix(self) -> TensorLike:
        """构建并缓存应变-位移矩阵 B"""

        if self._density_location in ['element']:
            qf = self._mesh.quadrature_formula(q)
            bcs, _ = qf.get_quadrature_points_and_weights()
            gphi = self._analyzer.scalar_space.grad_basis(bcs, variable='x') # (NC, NQ, LDOF, GD)
            B = self._material.strain_displacement_matrix(
                                            dof_priority=self._space_uh.dof_priority, 
                                            gphi=gphi
                                        ) # (NC, NQ, NS, TLDOF)
            
        elif self._density_location in ['multiresolution_element']:
            from soptx.interpolation.utils import calculate_multiresolution_gphi_eg, reshape_multiresolution_data_inverse
            gphi_eg_reshaped = calculate_multiresolution_gphi_eg(
                                            s_space_u=self._analyzer.scalar_space,
                                            q=q,
                                            n_sub=n_sub) # (NC*n_sub, NQ, LDOF, GD)
            B_reshaped = self._material.strain_displacement_matrix(
                                                dof_priority=self._tensor_space.dof_priority, 
                                                gphi=gphi_eg_reshaped
                                            ) # (NC*n_sub, NQ, NS, TLDOF)
            B = reshape_multiresolution_data_inverse(nx=nx_u, ny=ny_u, 
                                                    data_flat=B_reshaped, 
                                                    n_sub=n_sub) # (NC, n_sub, NQ, NS, TLDOF)
            
        else:
            self._log_error(f"Unsupported density location: {self._density_location}")
        
        return B

    def _compute_stress_state(self, density: TensorLike, state: dict) -> Dict[str, TensorLike]:
        """根据当前位移和密度计算完整的应力状态"""
        if isinstance(self._analyzer, LagrangeFEMAnalyzer):
            if state is None:
                state = self._analyzer.solve_state(rho_val=density)
        
            uh = state['displacement']
            cell2dof = self._space_uh.cell_to_dof()
            # 提取单元位移
            uh_e = uh[cell2dof]
            
            # 计算实体应力 (与密度无关)
            stress_solid = self._material.calculate_stress_vector(self._B, uh_e)
            
            # 计算惩罚后的应力
            stress_penalized = self._interpolation_scheme.interpolate_stress(
                                                                stress_solid=stress_solid,
                                                                rho_val=density
                                                            )
        
        elif isinstance(self._analyzer, HuZhangMFEMAnalyzer):
            raise NotImplementedError(f"暂未实现")
        
        else:
            self._log_error("State dictionary must contain either 'stress' or 'displacement'.")
        
        # 计算 von Mises 应力
        von_mises = self._material.calculate_von_mises_stress(stress_vector=stress_penalized)
        
        return {
                'stress_solid': stress_solid,
                'stress_penalized': stress_penalized,
                'von_mises': von_mises
            }
        
    def fun(self, 
            density: Union[Function, TensorLike], 
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """计算应力约束函数值"""
        stress_state = self._compute_stress_state(density, state)
        sigma_vm = stress_state['von_mises']
        
        # 2. 计算 P-norm 聚合应力 [cite: 206]
        # sigma_PN = (1/N * sum(sigma_vm^p))^(1/p)
        # 或者 Holmberg 论文中的公式 (6)
        
        # 注意数值稳定性：通常会先除以 limit 再做 power，或者使用 log-sum-exp (KS)
        # 这里演示标准的 P-norm
        p = self._p_norm_factor
        N = sigma_vm.shape[0] # 评估点数量
        
        # 为了数值稳定性，建议先归一化
        # term = (sigma_vm / self._stress_limit) ** p
        # sum_term = bm.sum(term)
        # sigma_pn_norm = (sum_term / N) ** (1/p)
        # g = sigma_pn_norm - 1.0
        
        # 直接使用论文公式 (6) [cite: 206]
        sum_pow = bm.sum(sigma_vm ** p)
        sigma_pn = (sum_pow / N) ** (1/p)
        
        g = sigma_pn / self._stress_limit - 1.0
        
        return g

    def jac(self, 
            density: Function, 
            state: dict,
            **kwargs
        ) -> TensorLike:
        """
        计算应力约束的灵敏度 (使用伴随法)
        参考文献: Holmberg et al. (2013) Section 7.4 [cite: 332]
        """
        # 1. 准备数据
        sigma_vm = state['von_mises']     # (Ne, )
        stress_vec = state['stress_penalized'] # (Ne, 3)
        uh = state['displacement']
        p = self._p_norm_factor
        N = sigma_vm.shape[0]
        
        # 2. 计算 P-norm 对局部 von Mises 应力的导数 d(sigma_PN)/d(sigma_vm) [cite: 311]
        # 这是一个向量，每个评估点一个值
        # 这里的实现需要根据具体的聚合公式推导
        # partial_PN_vm = ... 
        
        # 3. 计算 von Mises 对应力分量的导数 d(sigma_vm)/d(sigma) [cite: 317]
        # partial_vm_sigma = ...
        
        # 4. 组装伴随方程的右端项 (Pseudo Load)
        # RHS = sum( partial_PN_vm * partial_vm_sigma * E * B )
        # 这通常需要调用 material 类的方法来辅助计算
        
        # 5. 求解伴随方程 K * lambda = RHS [cite: 348]
        # lambda_adj = self._analyzer.solve(RHS) 
        # 或者 analyzer 提供一个专门的 solve_adjoint 接口
        
        # 6. 计算最终灵敏度 [cite: 350]
        # d(sigma_PN)/dx = ... - lambda^T * (dK/dx) * u
        # 这部分通常可以通过 filter.filter_constraint_sensitivities 结合 dK/dx 的计算逻辑完成
        
        # 这里返回的是关于物理密度 rho 的灵敏度
        return sensitivity_rho