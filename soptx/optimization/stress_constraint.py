from typing import Optional, Dict, Union
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged

class StressConstraint(BaseLogged):
    """
    局部应力约束计算器 (Aggregation-free)

    多项式消失约束: g_j = m_E(ρ) · ε_j · (ε_j² + 1),  ε_j = σ^v_j / σ_lim - 1
    
    在增广拉格朗日 (ALM) 框架中, 该类主要被 AugmentedLagrangianObjective 调用,
    用于计算罚函数项 P(k) 及其梯度中的应力约束相关量.

    Parameters
    ----------
    analyzer : LagrangeFEMAnalyzer
        有限元分析器, 提供位移求解、应力计算等功能.
    stress_limit : float
        材料的应力极限 σ_lim (许用应力).
    """
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                stress_limit: float,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit

        self._interpolation_scheme = self._analyzer.interpolation_scheme
        self._n_sub = self._interpolation_scheme.n_sub if self._interpolation_scheme.n_sub is not None else 1
        self._is_multiresolution = (self._n_sub > 1)
        
    @property
    def analyzer(self) -> LagrangeFEMAnalyzer:
        """获取当前的分析器"""
        return self._analyzer

    def fun(self, 
            density: TensorLike, 
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """计算多项式消失约束值 
            g_j = E_j · (ε_j³ + ε_j),  其中 ε_j = σ^v_j / σ_lim - 1"""
        if state is None:
            state = {}

        # 计算或获取实体应力
        if 'stress_solid' not in state:
            solid_stress_dict = self._analyzer.compute_stress_state(state)
            state.update(solid_stress_dict)

        # 计算或获取材料刚度插值系数 (相对刚度)
        if 'stiffness_ratio' not in state:
            cached = self._analyzer._cached_stiffness_relative
            if cached is None:
                raise RuntimeError(
                    "stiffness_ratio 未缓存: 请确保在调用 fun() 前已完成有限元分析, "
                    "使得 analyzer._cached_stiffness_relative 已被计算."
                    )
            state['stiffness_ratio'] = cached
            
        E = state['stiffness_ratio'] # 单分辨率: (NC,) | 多分辨率: (NC, n_sub)

        # 计算或获取 von Mises 应力和归一化偏差
        if 'von_mises' not in state:
            state['von_mises'] = self._analyzer.material.calculate_von_mises_stress(state['stress_solid'])
            
        vm = state['von_mises'] # 单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        s = vm / self._stress_limit - 1.0 # 单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)
        state['stress_deviation'] = s

        # 计算约束值 g = E * (s^3 + s)
        is_multiresolution = (E.ndim == 2)  # (NC, n_sub) vs (NC,)
        if is_multiresolution:
            g = E[:, :, None] * (s**3 + s)  # (NC, n_sub, NQ)
        else:
            g = E[:, None] * (s**3 + s)     # (NC, NQ)

        return g

    def jac(self, 
            density: TensorLike, 
            state: Optional[Dict] = None, 
            **kwargs
        ) -> TensorLike:
        """预留接口: 约束函数关于密度的完整梯度 (未实现)"""
        pass

    def compute_partial_gradient_wrt_stiffness(self, state: Dict) -> TensorLike:
        """计算约束关于相对刚度的偏导数 ∂g/∂E.

        由 g_j = E_j · (ε_j³ + ε_j), 对 E_j 求偏导 (ε_j 不显式依赖于 E_j):
            ∂g_j / ∂E_j = ε_j³ + ε_j"""
        s = state['stress_deviation']

        return s**3 + s

    def compute_adjoint_load(self, dPenaldVM: TensorLike, state: Dict) -> TensorLike:
        """计算伴随方程的右端项.
            K_T ξ = -Σ_j [λ_j + μ h_j] · ∂h_j/∂U
        
          其中 ∂h_j/∂U 通过链式法则展开:
               ∂h_j/∂U = (∂g_j/∂σ^v_j) · (∂σ^v_j/∂σ) · (∂σ/∂U)

        各项分别为:
        - ∂g_j/∂σ^v_j: 约束关于 von Mises 应力的导数 (由 dPenaldVM 加权传入)
        - ∂σ^v_j/∂σ = V₀σ / σ^v
        - ∂σ/∂U = D·B

        Parameters
        ----------
        dPenaldVM : TensorLike, shape (NC, NQ)
            罚函数关于 von Mises 应力的加权导数:
            dPenaldVM = (λ + μ·h) · ∂g/∂σ^v
            由 AugmentedLagrangianObjective 计算后传入.
        state : dict
            必须包含:
            - 'stress_solid': 实体 Cauchy 应力向量, shape (NC, NQ, NS)
            - 'von_mises': von Mises 应力标量, shape (NC, NQ)

        Returns
        -------
        adjoint_load : TensorLike, shape (gdofs,)
            组装后的全局伴随载荷向量.
        """
        material = self._analyzer.material
        disp_space = self._analyzer.tensor_space
        
        # --- 获取应变位移矩阵 B, 刚度矩阵 D, 和 von Mises 投影矩阵 M ---
        # 单分辨率: (NC, NQ, NS, LDOF) | 多分辨率: (NC, n_sub, NQ, NS, LDOF)
        B = self._analyzer.compute_strain_displacement_matrix(integration_order=1) 
        D = material.elastic_matrix()[0, 0]    # (NS, NS)
        M = material.von_mises_matrix()        # (NS, NS)

        # --- 计算 dVM / dSigma (von Mises 应力相对于 Cauchy 应力张量的导数)---  
        stress_vector = state['stress_solid'] # 单分辨率: (NC, NQ, NS) | 多分辨率: (NC, n_sub, NQ, NS)
        vm_val = state['von_mises']           # 单分辨率: (NC, NQ)     | 多分辨率: (NC, n_sub, NQ)
        vm_safe = bm.where(vm_val < 1e-12, 1.0, vm_val)

        if self._is_multiresolution:
            # (NC, n_sub, NQ, NS)
            M_sigma = bm.einsum('ij, csnj -> csni', M, stress_vector) 
            dVM_dSigma = M_sigma / vm_safe[..., None]

            term1_eps = bm.einsum('kl, csnk -> csnl', D, dVM_dSigma) # # (NC, n_sub, NQ, NS)

            element_sens = bm.einsum('csnkl, csnk -> csnl', B, term1_eps) # (NC, n_sub, NQ, LDOF)

            weights = dPenaldVM[..., None]                      # (NC, n_sub, NQ, 1)
            element_loads = -1.0 * element_sens * weights       # (NC, n_sub, NQ, LDOF)
            element_loads = bm.sum(element_loads, axis=(1, 2))  # (NC, LDOF)

        else:
            # (NC, NQ, NS)
            M_sigma = bm.einsum('ij, cqj -> cqi', M, stress_vector) 
            dVM_dSigma = M_sigma / vm_safe[..., None]

            # --- 应力 -> 应变 ( D^T * dVM/dSigma ) ---
            term1_eps = bm.einsum('kl, cqk -> cql', D, dVM_dSigma) # (NC, NQ, NS)

            # --- 应变 -> 位移 ( B^T * term1_eps ) ---
            element_sens = bm.einsum('cqkl, cqk -> cql', B, term1_eps) # (NC, NQ, LDOF)

            # --- 应用罚函数权重 dPenaldVM ---
            weights = dPenaldVM[..., None] # (NC, NQ)
            
            # 单元级伴随载荷 (来自伴随方程定义 F_adj = - dP/dU)
            element_loads = -1.0 * element_sens * weights # (NC, NQ, LDOF)

            # TODO 如果 NQ > 1 (多个积分点), 通常取平均或积分,
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
        """计算伴随法隐式灵敏度项
            ξ^T · (∂F_int/∂E_ℓ)"""
        # 获取单元基础刚度矩阵 K0
        if self._analyzer._cached_ke0 is None:
            K0 = self._analyzer.compute_solid_stiffness_matrix()
        else:
            K0 = self._analyzer._cached_ke0
            
        # 提取位移向量 U 和伴随向量 ψ
        uh = state['displacement']
        
        # 确保拉平为 1D 以便切片
        uh_flat = uh.reshape(-1)
        psi_flat = adjoint_vector.reshape(-1)
        
        # 提取单元级的局部向量 (NC, TLDOF)
        cell2dof = self._analyzer.tensor_space.cell_to_dof()
        uh_e = uh_flat[cell2dof]
        psi_e = psi_flat[cell2dof]
        
        # 逐单元张量收缩: ψ_e^T · K₀^e · u_e, shape (NC,)
        implicit_term = bm.einsum('ci, cij, cj -> c', psi_e, K0, uh_e)
        
        return implicit_term