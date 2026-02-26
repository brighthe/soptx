from typing import Optional, Dict
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from soptx.utils.base_logged import BaseLogged
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer

class ApparentStressConstraint(BaseLogged):
    """
    基于表观应力的无奇异局部应力约束计算器 (Apparent Stress Constraint)
    
    采用 ε-松弛格式: g_e = σ^v_e / σ_lim - η(ρ_e) ≤ 0
    阈值函数: η(ρ_e) = m_E(ρ_e) + ε * (1 - m_E(ρ_e))
    
    该类专为 Hu-Zhang 混合有限元分析器设计。通过直接约束系统原生求解出的宏观表观应力，
    彻底避免了传统位移法中因还原“实体应力”而引入的设计变量分母，从而在根源上消除了
    应力奇异性问题 。
    """
    def __init__(self,
                analyzer: HuZhangMFEMAnalyzer,
                stress_limit: float,
                epsilon: float = 1e-4,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit
        self._epsilon = epsilon
        
    @property
    def analyzer(self) -> HuZhangMFEMAnalyzer:
        return self._analyzer

    def fun(self, 
            density: TensorLike, 
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """
        计算 ε-松弛约束值: g_e = σ^v_e / σ_lim - η(ρ_e)
        """
        if state is None:
            state = {}

        # 1. 获取表观应力场 Σ 
        if 'stress_apparent' not in state:
            state.update(self._analyzer.compute_stress_state(state=state, rho_val=density))

        # 2. 获取材料刚度插值系数 m_E(ρ)
        if 'stiffness_ratio' not in state:
            E_rho_cached = self._analyzer._E_rho
            if E_rho_cached is None:
                raise RuntimeError("stiffness_ratio 未缓存，请确保已执行分析器正向计算。")
            E0 = self._analyzer.material.youngs_modulus
            state['stiffness_ratio'] = E_rho_cached / E0  
            
        m_E = state['stiffness_ratio'] # (NC,)

        # 3. 计算 von Mises 应力
        if 'von_mises' not in state:
            state['von_mises'] = self._analyzer.material.calculate_von_mises_stress(state['stress_apparent'])
            
        vm = state['von_mises'] # (NC, NQ) 

        # 4. 计算松弛阈值 η(ρ_e) = m_E + ε * (1 - m_E)
        eta = m_E + self._epsilon * (1.0 - m_E) # (NC, )
        state['eta_threshold'] = eta

        # 5. 执行约束评估
        g = (vm / self._stress_limit) - eta[..., None]

        return g
    
    def compute_partial_gradient_wrt_mE(self, state: Dict) -> TensorLike:
        """
        计算显式偏导数 ∂g_e / ∂m_E.
        
        根据公式 g = σ^v/σ_lim - (m_E + ε(1-m_E)):
        ∂g_e / ∂m_E = -(1 - ε)
        """
        return -(1.0 - self._epsilon)  # 标量

    def compute_adjoint_load(self, dPenaldVM: TensorLike, state: Dict) -> TensorLike:
        """
        计算伴随载荷 (右端项).
        
        对于混合元，伴随载荷直接作用在应力自由度 Σ 上:
            F_adj = -(∂σ^v/∂Σ)^T * dPenaldVM
        """
        material = self._analyzer.material
        stress_vector = state['stress_apparent']  # (NC, NQ, NS)
        vm_val = state['von_mises']               # (NC, NQ)
        
        # 避免零应力导致的除零
        vm_safe = bm.where(vm_val < 1e-12, 1.0, vm_val) # (NC, NQ)
        
        # von Mises 投影矩阵 M
        M = material.von_mises_matrix() # (NS, NS)

        # 1. 计算 ∂σ^v / ∂Σ = (M * Σ) / σ^v
        M_sigma = bm.einsum('ij, ...j -> ...i', M, stress_vector) # (NC, NQ, NS)
        dVM_dSigma = M_sigma / vm_safe[..., None]                 # (NC, NQ, NS)

        # 2. 获取应力基函数矩阵 Ψ
        stress_space = self._analyzer.huzhang_space
        disp_mesh = self._analyzer.disp_mesh
        integration_order = 1
        qf = disp_mesh.quadrature_formula(integration_order, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        # TODO
        phi = stress_space.basis(bcs) # (NC, NQ, LDOF, NS) HuZhang 原生 [xx, xy, yy]
        phi = phi[..., [0, 2, 1]]     # 标准 Voigt [xx, yy, xy]

        # 3. 通过基函数将 ∂σ^v/∂σ_vec 映射到自由度空间
        element_sens = bm.einsum('cqls, cqs -> cql', phi, dVM_dSigma)  # (NC, NQ, LDOF)

        # 4. 应用罚函数权重并对积分点求和
        weights = dPenaldVM[..., None] * ws[None, :, None]      # (NC, NQ, 1)
        # TODO
        element_loads = 1.0 * element_sens * weights            # (NC, NQ, LDOF)
        # element_loads = -1.0 * element_sens * weights           # (NC, NQ, LDOF)
        element_loads = bm.sum(element_loads, axis=1)           # (NC, LDOF)

        # 5. 全局组装至应力自由度
        gdofs = stress_space.number_of_global_dofs()
        adjoint_load = bm.zeros((gdofs,), dtype=bm.float64)
        
        cell2dof = stress_space.cell_to_dof()  # (NC, LDOF)
        indices = cell2dof.flatten()
        values = element_loads.flatten()
        
        bm.add_at(adjoint_load, indices, values)

        print(f"adjoint_load max: {float(bm.max(bm.abs(adjoint_load)))}")
        print(f"adjoint_load nonzero: {int(bm.sum(bm.abs(adjoint_load) > 1e-12))}")
        
        return adjoint_load

    def compute_implicit_sensitivity_term(self, 
                                          adjoint_vector: TensorLike, 
                                          state: Dict
                                        ) -> TensorLike:
        """计算伴随法隐式项: λ_σ^T * (∂A_σσ / ∂ρ) * Σ"""
        # TODO 
        A0 = self._analyzer._cached_Ae0  # (NC, LDOF, LDOF)

        huzhang_space = self._analyzer.huzhang_space
        cell2dof = huzhang_space.cell_to_dof()  # (NC, LDOF)

        # 提取伴随向量中的应力部分 λ_σ
        gdof_sigma = huzhang_space.number_of_global_dofs()
        lambda_sigma_e = adjoint_vector[:gdof_sigma][cell2dof]  # (NC, LDOF)

        # 提取应力自由度 Σ
        sigma_e = state['stress'][cell2dof]  # (NC, LDOF)

        # 逐单元计算: λ_σ,e^T * A0_e * Σ_e, shape (NC,)
        term = bm.einsum('ci, cij, cj -> c', lambda_sigma_e, A0, sigma_e)
        
        return term
    
    def compute_stress_measure(self, rho: TensorLike, state: Dict) -> TensorLike:
        """归一化应力测度：σ^v_apparent / (η * σ_lim)，>1 表示违反"""
        vm  = state['von_mises']         # (NC, NQ) 从系统直接获取独立表观应力
        eta = state['eta_threshold']     # (NC, ) 获取当前密度的松弛阈值

        # 将动态阈值 eta 移至分母，为外层 MMA 提供统一的 <= 1.0 判定标尺
        return vm / (eta[..., None] * self._stress_limit)
    
    def compute_visual_stress_measure(self, state: Dict) -> TensorLike:
        """可视化用归一化应力: σ^vM / σ_lim，不含 η 阈值"""
        vm = state['von_mises']  # (NC, NQ)
        return vm / self._stress_limit