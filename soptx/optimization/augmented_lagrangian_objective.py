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
        
        # --- 确保正向计算已完成，缓存 g、h 及 state 中间量 ---
        if self._is_apparent:
            if self._cache_g is None or 'stiffness_ratio' not in state:
                self.fun(density, state)
        else:
            if self._cache_g is None or 'stiffness_ratio' not in state \
                    or 'stress_deviation' not in state:
                self.fun(density, state)

        # ------------------------------------------------------------------ #
        # 第一步：准备公共中间量
        # ------------------------------------------------------------------ #
        slim = self._stress_constraint._stress_limit
        g    = self._cache_g   # (NC, NQ) 或 (NC, n_sub, NQ)
        h    = self._cache_h   # (NC, NQ) 或 (NC, n_sub, NQ)

        # 激活集：当 g > -λ/μ 时约束激活，h = g；否则 h = -λ/μ，梯度为零
        mask = g > (-self.lamb / self.mu)  # 与 g、h 同形

        # ------------------------------------------------------------------ #
        # 第二步：计算 dP/d(σ^vM)，用于构造伴随载荷
        #   位移元: ∂g/∂σ^vM = m_E · (3Λ² + 1) / σ_lim
        #   混合元: ∂g/∂σ^vM = 1 / σ_lim
        # ------------------------------------------------------------------ #
        if self._is_apparent:
            dhdVM_val = bm.ones_like(g) / slim
        else:
            s = state['stress_deviation']
            m_E = state['stiffness_ratio']
            if self._is_multiresolution:
                dhdVM_val = m_E[:, :, None] * (3 * s**2 + 1) / slim
            else:
                dhdVM_val = m_E[:, None] * (3 * s**2 + 1) / slim

        dPenaldVM = (self.lamb + self.mu * h) * bm.where(mask, dhdVM_val, 0.0)

        if enable_timing:
            t.send('罚函数偏导数')

        # ------------------------------------------------------------------ #
        # 第三步：计算显式偏导数 ∂P/∂m_E|_explicit
        #   位移元: ∂g/∂m_E = Λ³ + Λ
        #   混合元: ∂g/∂m_E = -(1 - ε)
        # ------------------------------------------------------------------ #
        dgdm_E = self._stress_constraint.compute_partial_gradient_wrt_mE(state=state)
        dPenaldm_E_explicit = bm.where(mask, (self.lamb + self.mu * h) * dgdm_E, 0.0)  # (NC, NQ) 或 (NC, n_sub, NQ)

        # ------------------------------------------------------------------ #
        # 第四步：伴随法求解隐式偏导数 ∂P/∂m_E|_implicit
        #
        #   组装伴随载荷: F_adj = ∂P/∂(状态变量)
        #     位移元作用在位移自由度: F_adj = (∂σ^v/∂U)^T · dP/dσ^v
        #     混合元作用在应力自由度: F_adj = (∂σ^v/∂Σ)^T · dP/dσ^v
        #
        #   求解伴随方程:
        #     位移元: K · ψ = F_adj
        #     混合元: [A  B^T; B  0] · [λ_σ; λ_u] = [F_adj; 0]  (复用正向 LU 分解)
        #
        #   隐式项:
        #     位移元: ψ^T · (∂K_e/∂m_E) · U_e
        #     混合元: (1/m_E²) · λ_σ,e^T · A⁰_e · Σ_e
        # ------------------------------------------------------------------ #
        adjoint_load = self._stress_constraint.compute_adjoint_load(dPenaldVM=dPenaldVM, state=state)  # (gdofs, )

        if enable_timing:
            t.send('组装伴随向量')

        adjoint_vector = self._analyzer.solve_adjoint(rhs=adjoint_load, rho_val=density)  # (gdofs, )

        if enable_timing:
            t.send('解伴随方程')

        dPenaldm_E_implicit = self._stress_constraint.compute_implicit_sensitivity_term(
                                                        adjoint_vector, state)  # (NC,) 或 (NC, n_sub)
        
        # ------------------------------------------------------------------ #
        # 第五步：链式法则组装 dP/dρ
        # ------------------------------------------------------------------ #
        # 显式项对 NQ 维度求和，还原为单元级标量
        dPenaldm_E_explicit_reduced = bm.sum(dPenaldm_E_explicit, axis=-1)  # (NC,) 或 (NC, n_sub)

        # 计算 dm_E/dρ = (dE/dρ) / E₀
        dm_E_drho = self._interpolation_scheme.interpolate_material_derivative(
                                                            material=self._material, rho_val=density
                                                        ) / self._material.youngs_modulus  # (NC,) 或 (NC, n_sub)

        if self._is_apparent:
            # 混合元:
            #   显式项: (∂P/∂m_E|_explicit) · dm_E/dρ
            #   隐式项: (1/m_E²) · λ_σ^T A⁰ Σ · m_E' = dPenaldm_E_implicit · dm_E_drho
            #           (compute_implicit_sensitivity_term 已预除 m_E²，外部只需乘 m_E')
            dP_drho = dPenaldm_E_explicit_reduced * dm_E_drho + dPenaldm_E_implicit * dm_E_drho           # (NC,)
        else:
            # 位移元: 显式与隐式通过同一 dm_E/dρ 串联
            dP_drho = (dPenaldm_E_explicit_reduced + dPenaldm_E_implicit) * dm_E_drho  # (NC,) 或 (NC, n_sub)
            
        # ------------------------------------------------------------------ #
        # 第六步：归一化并组装总梯度
        #   dJ/dρ = ∂f/∂ρ + (1/N) · dP/dρ
        # ------------------------------------------------------------------ #
        dVol_drho     = self._volume_objective.jac(density=density, state=state)
        n_constraints = g.numel() if hasattr(g, 'numel') else g.size

        dJ_drho = dVol_drho + dP_drho / n_constraints  # (NC,) 或 (NC, n_sub)

        print(f"[SIGN] dPenaldm_E_explicit_reduced: mean={float(bm.mean(dPenaldm_E_explicit_reduced)):.4e}")
        print(f"[SIGN] dPenaldm_E_implicit:         mean={float(bm.mean(dPenaldm_E_implicit)):.4e}")
        print(f"[SIGN] dm_E_drho:                   mean={float(bm.mean(dm_E_drho)):.4e}")
        print(f"[SIGN] dP_drho:                     mean={float(bm.mean(dP_drho)):.4e}")
        print(f"[SIGN] dJ_drho:                     mean={float(bm.mean(dJ_drho)):.4e}")
        # 同时打印 mask 覆盖率（确认激活约束比例）
        print(f"[SIGN] mask 激活比例: {float(bm.mean(mask.astype(bm.float64))):.4f}")
        
        if enable_timing:
            t.send('其他')
            t.send(None)

        return dJ_drho

    # def _manual_differentiation_backup(self, 
    #                     density: Union[Function, TensorLike],
    #                     state: Optional[dict] = None, 
    #                     enable_timing: bool = False, 
    #                     **kwargs
    #                 ) -> TensorLike:
    #     # --- 缓存检查与状态同步 ---
    #     if (self._cache_g is None) or ('stiffness_ratio' not in state) or ('stress_deviation' not in state):
    #         self.fun(density, state)
        
    #     # --- 获取缓存的物理量 ---
    #     slim = self._stress_constraint._stress_limit
    #     E = state['stiffness_ratio']  # 单分辨率: (NC,)         | 多分辨率: (NC, n_sub)
    #     s = state['stress_deviation'] # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)

    #     g = self._cache_g             # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)
    #     h = self._cache_h             # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)

    #     # --- 确定激活集 a1 (Mask) ---
    #     #  逻辑: 当 g > -lambda/mu 时，h = g, 此时约束激活（或违反）
    #     limit_term = -self.lamb / self.mu
    #     mask = g > limit_term         # 单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)

    #     # --- 计算显式灵敏度 ---
    #     #  计算 dPenaldVM (罚函数对 Von Mises 应力的偏导数)
    #     #  单分辨率: (NC, NQ)      | 多分辨率: (NC, n_sub, NQ)
    #     if self._is_multiresolution:
    #         dhdVM_val = E[:, :, None] * (3 * s**2 + 1) / slim  # (NC, n_sub, NQ)
    #     else:
    #         dhdVM_val = E[:, None] * (3 * s**2 + 1) / slim     # (NC, NQ)
    #     # dhdVM_val = E[:, None] * (3 * s**2 + 1) / slim
    #     dhdVM = bm.where(mask, dhdVM_val, 0.0)
    #     dPenaldVM = (self.lamb + self.mu * h) * dhdVM  

    #     #  计算 dPenal/dE 的显式部分
    #     #  单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)
    #     dgdE = self._stress_constraint.compute_partial_gradient_wrt_mE(state=state)
    #     dPenaldE_explicit = bm.where(mask, (self.lamb + self.mu * h) * dgdE, 0.0) 

    #     # --- 伴随法 ---        
    #     #  计算伴随载荷 F_adj = - (dVM/dU)^T * dPenaldVM
    #     adjoint_load = self._stress_constraint.compute_adjoint_load(dPenaldVM=dPenaldVM, state=state) # (gdofs, )
        
    #     # 解伴随方程: K * psi = F_adj
    #     adjoint_vector = self._analyzer.solve_adjoint(rhs=adjoint_load, rho_val=density) # (gdofs, )
        
    #     # --- 计算隐式灵敏度 ---
    #     # dPenal/dE_implicit = psi^T * (dF_int / dE)
    #     dPenaldE_implicit = self._stress_constraint.compute_implicit_sensitivity_term(adjoint_vector, state)  # (NC, )

    #     # --- 显式项归约: 对 NQ 维度求和 (NQ 始终在最后一维) ---
    #     dPenaldE_explicit_reduced = bm.sum(dPenaldE_explicit, axis=-1) # 单分辨率: (NC,) | 多分辨率: (NC, n_sub)

    #     # --- 总灵敏度 (关于刚度变量 E) ---
    #     if self._is_multiresolution:
    #         dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit[:, None]  # (NC, n_sub)
    #     else:
    #         dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit  # (NC,)
    #     # dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit # 单分辨率: (NC,) | 多分辨率: (NC, n_sub)

    #     # --- 链式法则: dPenal/drho = dPenal/dE * dE/drho ---
    #     # 获取 dE/drho (取决于具体的插值模型)
    #     dE_drho_absolute = self._interpolation_scheme.interpolate_material_derivative(
    #                                                 material=self._material, 
    #                                                 rho_val=density
    #                                             )  # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )
    #     E0 = self._material.youngs_modulus
    #     dE_drho = dE_drho_absolute / E0    # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

    #     dP_drho = dPenaldE_total * dE_drho # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

    #     # 计算主目标函数 (体积) 的梯度
    #     dVol_drho = self._volume_objective.jac(density=density, state=state) # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

    #     # 归一化处理
    #     dP_drho_normalized = dP_drho / self._NC  # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

    #     # 组装总拉格朗日函数的梯度 (关于物理密度 rho)
    #     dJ_drho = dVol_drho + dP_drho_normalized # 单分辨率: (NC, ) | 多分辨率: (NC*n_sub, )

    #     return dJ_drho
    
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
    