from typing import Optional, Literal, Union, Dict, TYPE_CHECKING
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.volume_objective import VolumeObjective
from soptx.optimization.vanish_stress_constraint import VanishingStressConstraint
from soptx.optimization.apparent_stress_constaint import ApparentStressConstraint 
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

# 使用 TYPE_CHECKING 避免循环导入，仅用于类型提示
if TYPE_CHECKING:
    from soptx.optimization.al_mma_optimizer import ALMMMAOptions

class AugmentedLagrangianObjective(BaseLogged):
    def __init__(self,
                volume_objective: VolumeObjective, 
                stress_constraint: Union[VanishingStressConstraint, ApparentStressConstraint],
                options: 'ALMMMAOptions',
                initial_lambda: Optional[TensorLike] = None,
                diff_mode: Literal["auto", "manual"] = "manual",
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """增广拉格朗日目标函数 - 体积最小化 + 应力约束

        归一化增广拉格朗日子问题:
            J^(k)(z, U) = f(z) + (1/N) · P^(k)(z, U)

        其中:
        - f(z) 为体积目标函数 (归一化), 由 VolumeObjective 计算
        - P^(k) 为罚函数项:
            P^(k) = Σ_j [λ_j · h_j + (μ/2) · h_j²]
        - h_j = max(g_j, -λ_j/μ), Eq. (40)
        - g_j 为应力约束
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._volume_objective = volume_objective
        self._stress_constraint = stress_constraint

        self._is_apparent = isinstance(stress_constraint, ApparentStressConstraint)

        self._options = options

        self._diff_mode = diff_mode

        # 缓存: 用于在 fun() 和 jac() 之间共享中间结果
        self._cache_g = None  # 约束值 g, shape (NC, NQ)
        self._cache_h = None  # 辅助等式约束 h, shape (NC, NQ)

        self._analyzer = stress_constraint.analyzer
        self._disp_mesh = self._analyzer.disp_mesh
        self._NC = self._disp_mesh.number_of_cells()
        
        self._interpolation_scheme = self._analyzer.interpolation_scheme
        self._material = self._analyzer.material

        self._n_sub = self._interpolation_scheme.n_sub if self._interpolation_scheme.n_sub is not None else 1
        self._is_multiresolution = (self._n_sub > 1)

        # --- ALM 参数初始化 ---
        
        # 罚因子 μ^(k)
        self.mu = float(options.mu_0)
        # 最大罚因子 μ_max
        self.mu_max = float(options.mu_max)

        # 拉格朗日乘子 lambda 的初始化
        if initial_lambda is not None:
            # Case A: 热启动
            if self._is_multiresolution:
                expected_shape = (self._NC, self._n_sub, 1)
            else:
                expected_shape = (self._NC, 1)

            if initial_lambda.shape != expected_shape:
                self._log_error(
                    f"Shape mismatch: 期望 {expected_shape}, 实际得到 {initial_lambda.shape}"
                )
            self.lamb = bm.copy(initial_lambda)
        else:
            # Case B: 冷启动
            init_val = options.lambda_0_init_val
            if self._is_multiresolution:
                self.lamb = bm.full((self._NC, self._n_sub, 1), init_val, dtype=bm.float64)
            else:
                self.lamb = bm.full((self._NC, 1), init_val, dtype=bm.float64)

    def fun(self, 
            density: Union[Function, TensorLike],
            state: Optional[Dict] = None, 
            **kwargs
        ) -> float:
        # 1. 计算体积部分 f
        f = self._volume_objective.fun(density, state)

        # 2. 计算应力约束 g
        g = self._stress_constraint.fun(density, state) # 单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        # 设计约定断言：每单元恰好1个应力评估点
        assert g.shape[-1] == 1, (
                        f"要求 NQ=1，但实际 NQ={g.shape[-1]}，请检查积分阶数设置。"
                    )

        # 3. 计算 ALM 的 h 和 Penal
        #    h_j = max(g_j, -lambda_j / mu)
        h = bm.maximum(g, -self.lamb / self.mu) # 单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        penal = bm.sum(self.lamb * h + 0.5 * self.mu * h**2)

        self._cache_g = g
        self._cache_h = h

        # 4. 组装归一化后的增广拉格朗日目标函数 J
        n_constraints = g.numel() if hasattr(g, 'numel') else g.size
        J = f + penal / n_constraints

        return J
    
    def jac(self, 
            density: Union[Function, TensorLike], 
            state: Optional[dict] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None,
            **kwargs
        ) -> TensorLike:
        mode = diff_mode if diff_mode is not None else self._diff_mode

        if mode == "manual":
            return self._manual_differentiation(density=density, state=state, **kwargs)
        
        elif mode == "auto":  
            return self._auto_differentiation(density=density, state=state, **kwargs)
        
        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)

    def _manual_differentiation(self, 
                    density: Union[Function, TensorLike],
                    state: Optional[dict] = None, 
                    enable_timing: bool = None, 
                    **kwargs
                ) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"目标函数灵敏度分析")
            next(t)

        if state is None:
            state = {}
        
        # --- 缓存检查与状态同步 ---
        if self._is_apparent:
            if self._cache_g is None or 'stiffness_ratio' not in state:
                self.fun(density, state)
        else:
            if self._cache_g is None or 'stiffness_ratio' not in state or 'stress_deviation' not in state:
                self.fun(density, state)

        # --- 获取缓存的物理量 ---
        slim = self._stress_constraint._stress_limit
        m_E = state['stiffness_ratio']  # 单分辨率: (NC,) | 多分辨率: (NC, n_sub)

        g = self._cache_g             # 单分辨率/混合元: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)
        h = self._cache_h             # 单分辨率/混合元: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        # --- 确定激活集 (Mask) ---
        # 当 g > -lambda/mu 时，约束激活（或违反），h = g
        limit_term = -self.lamb / self.mu
        mask = g > limit_term         # 单分辨率/混合元: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        # --- 计算 dPenaldVM (罚函数对 von Mises 应力的偏导数) ---
        # 两种约束对 ∂g/∂σ^vM 的形式不同:
        # - 位移元 (VanishingStressConstraint): ∂g/∂σ^vM = m_E(ρ) * (3ε² + 1) / σ_lim
        # - 混合元 (ApparentStressConstraint):  ∂g/∂σ^vM = 1 / σ_lim (常数)
        if self._is_apparent:
            dhdVM_val = bm.ones_like(g) / slim    # (NC, NQ)
        else:
            s = state['stress_deviation']
            if self._is_multiresolution:
                dhdVM_val = m_E[:, :, None] * (3 * s**2 + 1) / slim  # (NC, n_sub, NQ)
            else:
                dhdVM_val = m_E[:, None] * (3 * s**2 + 1) / slim     # (NC, NQ)

        dhdVM = bm.where(mask, dhdVM_val, 0.0)
        dPenaldVM = (self.lamb + self.mu * h) * dhdVM

        # --- 计算 dPenal/dm_E 的显式部分 ---
        # 两种约束对 ∂g/∂m_E 的形式不同:
        # - 位移元: ∂g/∂m_E = ε³ + ε    (单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ))
        # - 混合元: ∂g/∂m_E = -(1 - ε)  (常数)
        dgdm_E = self._stress_constraint.compute_partial_gradient_wrt_mE(state=state) 
        dPenaldm_E_explicit = bm.where(mask, (self.lamb + self.mu * h) * dgdm_E, 0.0)

        if enable_timing:
            t.send('罚函数偏导数')

        # --- 伴随法 ---
        # 计算伴随载荷:
        # - 位移元: F_adj 作用在位移自由度上, F_adj = -(∂σ^v/∂U)^T * dPenaldVM
        # - 混合元: F_adj 作用在应力自由度上, F_adj = -(∂σ^v/∂Σ)^T * dPenaldVM
        adjoint_load = self._stress_constraint.compute_adjoint_load(
                                                    dPenaldVM=dPenaldVM, state=state)  # (gdofs,)
        if enable_timing:
            t.send('组装伴随向量')

        # 解伴随方程
        # - 位移元: K * ψ = F_adj
        # - 混合元: [A  B^T; B  0] * [λ_σ; λ_u] = [F_adj; 0] (复用正向分解)
        adjoint_vector = self._analyzer.solve_adjoint(
                                            rhs=adjoint_load, rho_val=density)  # (gdofs,)
        if enable_timing:
            t.send('解伴随方程')

        # --- 计算隐式灵敏度项 ---
        # - 位移元: ψ^T * (∂K/∂m_E) * U
        # - 混合元: λ_σ^T * (∂A_σσ/∂ρ) * Σ  (内积，不含 ρ 的幂次系数)
        dPenaldm_E_implicit = self._stress_constraint.compute_implicit_sensitivity_term(
                                                        adjoint_vector, state)  # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

        # --- 显式项归约: 对 NQ 维度求和 ---
        dPenaldm_E_explicit_reduced = bm.sum(dPenaldm_E_explicit, axis=-1)
        # 单分辨率/混合元: (NC,) | 多分辨率: (NC, n_sub)

        # --- 链式法则: dPenal/dρ ---
        # 最终灵敏度一般公式（适用于任意插值模型）:
        #   dJ/dρ_e = ∂f/∂ρ_e + (1/N_e) * ∂P/∂ρ_e + m_E'(ρ)/m_E²(ρ) * (λ_σ^T A0 Σ)
        #
        # 两种框架的链式法则系数存在根本差异:
        # - 位移元: A_σσ 不存在, 刚度 E = m_E * E0
        #   显式项与隐式项均通过同一系数 dm_E/dρ 串联:
        #   dP/dρ = (dP/dm_E) * dm_E/dρ
        # - 混合元: A_σσ,e = (1/m_E) * A0,  ∂A/∂ρ = -m_E'(ρ)/m_E²(ρ) * A0
        #   显式项通过 dm_E/dρ 串联
        #   隐式项通过通用系数 m_E'(ρ)/m_E²(ρ) 串联（适用于任意插值模型）
        dE_drho_absolute = self._interpolation_scheme.interpolate_material_derivative(
                                                            material=self._material, rho_val=density)  # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)
        E0 = self._material.youngs_modulus
        dm_E_drho = dE_drho_absolute / E0  # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

        if self._is_apparent:
            # 混合元: 显式项与隐式项系数分开处理
            # 显式项: (∂P/∂m_E) * m_E'(ρ)
            dP_drho_explicit = dPenaldm_E_explicit_reduced * dm_E_drho  # (NC,)

            # 隐式项系数: m_E'(ρ) / m_E²(ρ)  —— 适用于任意插值模型
            coeff_implicit = dm_E_drho / m_E**2                         # (NC,)
            dP_drho_implicit = coeff_implicit * dPenaldm_E_implicit     # (NC,)

            dP_drho = dP_drho_explicit + dP_drho_implicit               # (NC,)

        else:
            # 位移元: 显式项与隐式项共享同一个 dE/dρ
            dPenaldm_E_total = dPenaldm_E_explicit_reduced + dPenaldm_E_implicit # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

            dP_drho = dPenaldm_E_total * dm_E_drho  # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

        # --- 体积目标函数梯度 ---
        dVol_drho = self._volume_objective.jac(
                        density=density, state=state)  # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

        # --- 归一化并组装总梯度 ---
        n_constraints = g.numel() if hasattr(g, 'numel') else g.size # (单分辨率: NC*NQ | 多分辨率: NC*n_sub*NQ)
        dP_drho_normalized = dP_drho / n_constraints  # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

        dJ_drho = dVol_drho + dP_drho_normalized      # (单分辨率: (NC, ) | 多分辨率: (NC, n_sub)

        # 解伴随方程后
        print(f"adjoint_vector max: {float(bm.max(bm.abs(adjoint_vector)))}")

        # 各灵敏度分量
        print(f"dVol_drho max:              {float(bm.max(bm.abs(dVol_drho)))}")
        print(f"dP_drho_explicit max:       {float(bm.max(bm.abs(dP_drho_explicit)))}")  
        print(f"dPenaldm_E_implicit max:    {float(bm.max(bm.abs(dPenaldm_E_implicit)))}")
        print(f"coeff_implicit max:         {float(bm.max(bm.abs(coeff_implicit)))}")
        print(f"dP_drho_implicit max:       {float(bm.max(bm.abs(dP_drho_implicit)))}")
        print(f"dP_drho_normalized max:     {float(bm.max(bm.abs(dP_drho_normalized)))}")
        print(f"dJ_drho max:                {float(bm.max(bm.abs(dJ_drho)))}")

        print(f"dP_drho_explicit min: {float(bm.min(dP_drho_explicit))}")  # 确认符号
        print(f"dP_drho_implicit min: {float(bm.min(dP_drho_implicit))}")  # 确认符号
        print(f"dP_drho max (before norm): {float(bm.max(dP_drho))}")
        print(f"dP_drho min (before norm): {float(bm.min(dP_drho))}")

        if enable_timing:
            t.send('其他')
            t.send(None)

        return dJ_drho

    def _manual_differentiation_backup(self, 
                        density: Union[Function, TensorLike],
                        state: Optional[dict] = None, 
                        enable_timing: bool = False, 
                        **kwargs
                    ) -> TensorLike:
        # --- 缓存检查与状态同步 ---
        if (self._cache_g is None) or ('stiffness_ratio' not in state) or ('stress_deviation' not in state):
            self.fun(density, state)
        
        # --- 获取缓存的物理量 ---
        slim = self._stress_constraint._stress_limit
        E = state['stiffness_ratio']  # 单分辨率: (NC,)         | 多分辨率: (NC, n_sub)
        s = state['stress_deviation'] # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)

        g = self._cache_g             # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)
        h = self._cache_h             # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)

        # --- 确定激活集 a1 (Mask) ---
        #  逻辑: 当 g > -lambda/mu 时，h = g, 此时约束激活（或违反）
        limit_term = -self.lamb / self.mu
        mask = g > limit_term         # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)

        # --- 计算显式灵敏度 ---
        #  计算 dPenaldVM (罚函数对 Von Mises 应力的偏导数)
        #  单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)
        if self._is_multiresolution:
            dhdVM_val = E[:, :, None] * (3 * s**2 + 1) / slim  # (NC, n_sub, NQ)
        else:
            dhdVM_val = E[:, None] * (3 * s**2 + 1) / slim     # (NC, NQ)
        # dhdVM_val = E[:, None] * (3 * s**2 + 1) / slim
        dhdVM = bm.where(mask, dhdVM_val, 0.0)
        dPenaldVM = (self.lamb + self.mu * h) * dhdVM  

        #  计算 dPenal/dE 的显式部分
        #  单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)
        dgdE = self._stress_constraint.compute_partial_gradient_wrt_stiffness(state=state)
        dPenaldE_explicit = bm.where(mask, (self.lamb + self.mu * h) * dgdE, 0.0) 

        # --- 伴随法 ---        
        #  计算伴随载荷 F_adj = - (dVM/dU)^T * dPenaldVM
        adjoint_load = self._stress_constraint.compute_adjoint_load(dPenaldVM=dPenaldVM, state=state) # (gdofs, )
        
        # 解伴随方程: K * psi = F_adj
        adjoint_vector = self._analyzer.solve_adjoint(rhs=adjoint_load, rho_val=density) # (gdofs, )
        
        # --- 计算隐式灵敏度 ---
        # dPenal/dE_implicit = psi^T * (dF_int / dE)
        dPenaldE_implicit = self._stress_constraint.compute_implicit_sensitivity_term(adjoint_vector, state)  # (NC, )

        # --- 显式项归约: 对 NQ 维度求和 (NQ 始终在最后一维) ---
        dPenaldE_explicit_reduced = bm.sum(dPenaldE_explicit, axis=-1) # 单分辨率: (NC,) | 多分辨率: (NC, n_sub)

        # --- 总灵敏度 (关于刚度变量 E) ---
        if self._is_multiresolution:
            dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit[:, None]  # (NC, n_sub)
        else:
            dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit  # (NC,)
        # dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit # 单分辨率: (NC,) | 多分辨率: (NC, n_sub)

        # --- 链式法则: dPenal/drho = dPenal/dE * dE/drho ---
        # 获取 dE/drho (取决于具体的插值模型)
        dE_drho_absolute = self._interpolation_scheme.interpolate_material_derivative(
                                                    material=self._material, 
                                                    rho_val=density
                                                )  # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )
        E0 = self._material.youngs_modulus
        dE_drho = dE_drho_absolute / E0    # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

        dP_drho = dPenaldE_total * dE_drho # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

        # 计算主目标函数 (体积) 的梯度
        dVol_drho = self._volume_objective.jac(density=density, state=state) # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

        # 归一化处理
        dP_drho_normalized = dP_drho / self._NC  # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

        # 组装总拉格朗日函数的梯度 (关于物理密度 rho)
        dJ_drho = dVol_drho + dP_drho_normalized # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

        return dJ_drho
    
    def update_multipliers(self) -> None:
        """更新拉格朗日乘子 λ 和 罚因子 μ.
        
        此方法应在每一轮 ALM 外层迭代结束时调用.
        """
        if self._cache_h is None:
            raise RuntimeError(
                "update_multipliers() 必须在 fun() 之后调用, "
                "以确保 h 已被计算并缓存."
            )
    
        # 1. 更新拉格朗日乘子 λ
        # λ^(k+1) = λ^(k) + μ^(k) · h
        self.lamb = self.lamb + self.mu * self._cache_h
        
        # 2. 更新罚因子 μ
        # μ^(k+1) = min(α · μ^(k), μ_max) [cite: 303]
        self.mu = min(self._options.alpha * self.mu, self.mu_max)
    