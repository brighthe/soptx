from typing import Optional, Dict
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.utils.base_logged import BaseLogged

class StressConstraint(BaseLogged):
    """
    局部应力约束计算器 (Aggregation-free)
    
    虽然它是一个 "Constraint", 但在 ALM 框架中, 它主要被 AugmentedLagrangianObjective 调用
    来计算罚函数项及其梯度
    """
    def __init__(self,
                analyzer: LagrangeFEMAnalyzer,
                stress_limit: float,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit
        
    @property
    def analyzer(self) -> LagrangeFEMAnalyzer:
        """获取当前的分析器"""
        return self._analyzer

    def fun(self, 
            density: TensorLike, 
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """计算多项式消失约束值 g = E * (s^3 + s)"""
        if state is None:
            state = {}

        # 计算或获取实体应力
        if 'stress_solid' not in state:
            solid_stress_dict = self._analyzer.compute_stress_state(state)
            state.update(solid_stress_dict)

        # 计算或获取材料刚度插值系数 (相对刚度)
        if 'stiffness_ratio' not in state:
            state['stiffness_ratio'] = self._analyzer._cached_stiffness_relative
            
        E = state['stiffness_ratio'] # (NC, )

        # 计算或获取 von Mises 应力和归一化偏差
        if 'von_mises' not in state:
            state['von_mises'] = self._analyzer.material.calculate_von_mises_stress(state['stress_solid'])
            
        vm = state['von_mises'] # (NC, NQ)

        s = vm / self._stress_limit - 1.0 # (NC, NQ)
        state['stress_deviation'] = s

        # 计算约束值 g = E * (s^3 + s)
        g = E[:, None] * (s**3 + s) # (NC, NQ)

        return g

    def jac(self, 
            density: TensorLike, 
            state: Optional[Dict] = None, 
            **kwargs
        ) -> TensorLike:
        pass

    def compute_partial_gradient_wrt_stiffness(self, state: Dict) -> TensorLike:
        """计算多项式消除约束相对于相对刚度的倒数 dg / dE = s^3 + s"""
        s = state['stress_deviation']

        return s**3 + s

    def compute_adjoint_load(self, dPenaldVM: TensorLike, state: Dict) -> TensorLike:
        """计算伴随方程的右端项"""
        material = self._analyzer.material
        disp_space = self._analyzer.tensor_space
        
        # --- 获取应变位移矩阵 B, 刚度矩阵 D, 和 von Mises 投影矩阵 M ---
        B = self._analyzer.compute_strain_displacement_matrix(integration_order=1) # (NC, NQ, NS, LDOF)
        D = material.elastic_matrix()[0, 0]    # (NS, NS)
        M = material.von_mises_matrix()        # (NS, NS)

        # --- 计算 dVM / dSigma (von Mises 应力相对于 Cauchy 应力张量的导数)---  
        stress_vector = state['stress_solid'] # (NC, NQ, NS)
        vm_val = state['von_mises']           # (NC, NQ)
        vm_safe = bm.where(vm_val < 1e-12, 1.0, vm_val)

        M_sigma = bm.einsum('ij, cqj -> cqi', M, stress_vector) # (NC, NQ, NS)

        dVM_dSigma = M_sigma / vm_safe[..., None] # (NC, NQ, NS)

        # --- 应力 -> 应变 ( D^T * dVM/dSigma ) ---
        term1_eps = bm.einsum('kl, cqk -> cql', D, dVM_dSigma) # (NC, NQ, NS)

        # --- 应变 -> 位移 ( B^T * term1_eps ) ---
        element_sens = bm.einsum('cqkl, cqk -> cql', B, term1_eps) # (NC, NQ, LDOF)

        # --- 应用罚函数权重 dPenaldVM ---
        weights = dPenaldVM[..., None] # (NC, NQ)
        
        # 单元级伴随载荷 (来自伴随方程定义 F_adj = - dP/dU)
        element_loads = -1.0 * element_sens * weights # (NC, NQ, LDOF)

        # 如果 NQ > 1 (多个积分点), 通常取平均或积分,
        # 但对应力约束而言，通常是每个点一个约束.
        element_loads = bm.sum(element_loads, axis=1) # (NC, LDOF)

        # --- 全局组装 ---
        cell2dof = disp_space.cell_to_dof() # (NC, LDOF)
        indices = cell2dof.flatten()
        values = element_loads.flatten()
        
        # 初始化全局载荷向量
        gdofs = disp_space.number_of_global_dofs()
        adjoint_load = bm.zeros((gdofs, ), dtype=bm.float64)
        
        # 累加组装
        bm.add_at(adjoint_load, indices, values)
        
        return adjoint_load
    
    def compute_implicit_sensitivity_term(self, 
                                          adjoint_vector: TensorLike, 
                                          state: Dict
                                        ) -> TensorLike:
        """计算伴随法隐式灵敏度项: psi^T * (dF_int / dE)"""
        # 获取单元基础刚度矩阵 K0
        if self._analyzer._cached_ke0 is None:
            K0 = self._analyzer.compute_solid_stiffness_matrix()
        else:
            K0 = self._analyzer._cached_ke0
            
        # 提取全局位移 U 和伴随向量 psi
        uh = state['displacement']
        
        # 确保拉平为 1D 以便切片
        uh_flat = uh.reshape(-1)
        psi_flat = adjoint_vector.reshape(-1)
        
        # 提取单元级的局部向量 (NC, TLDOF)
        cell2dof = self._analyzer.tensor_space.cell_to_dof()
        uh_e = uh_flat[cell2dof]
        psi_e = psi_flat[cell2dof]
        
        # 张量收缩: 一次性计算所有单元的 psi_e^T * K_{0,e} * u_e
        implicit_term = bm.einsum('ci, cij, cj -> c', psi_e, K0, uh_e)
        
        return implicit_term