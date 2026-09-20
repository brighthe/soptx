import math
from typing import Optional, Literal, Union, Dict, TYPE_CHECKING
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.core import BaseLogged, timer
from soptx.topology.constraints.stress_formulation import StressConstraintProtocol
from soptx.topology.objectives.volume import VolumeObjective

# 使用 TYPE_CHECKING 避免循环导入，仅用于类型提示
if TYPE_CHECKING:
    from soptx.topology.optimizers.al_mma import ALMMMAOptions

class AugmentedLagrangianObjective(BaseLogged):
    def __init__(self,
                volume_objective: VolumeObjective, 
                stress_constraint: StressConstraintProtocol,
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
        # 乘子安全阈 λ_max (None 表示不设上限, 复现无阈更新); 用 getattr 读取,
        # 使单测可用 SimpleNamespace 注入不含该字段的选项对象。
        lambda_max = getattr(options, 'lambda_max', None)
        self.lambda_max = None if lambda_max is None else float(lambda_max)
        if self.lambda_max is not None and not (
                math.isfinite(self.lambda_max) and self.lambda_max > 0):
            raise ValueError("lambda_max 必须是有限正数或 None.")
        # 最近一次 update_multipliers() 中被 λ_max 截断的乘子个数 (诊断量)
        self.last_capped_count = 0

        # 罚因子条件放大所需的违反度缓存 (由优化器通过 set_current_violation 推入)
        self._prev_violation = None
        self._pending_violation = None

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
        
        # 每次刷新当前状态下的约束值与 AL 权重, 不检查具体模型的状态键.
        # 有限元状态应由调用方提供; 约束对象负责准备应力中间量.
        self.fun(density=density, state=state)

        # ------------------------------------------------------------------ #
        # 第一步：准备公共中间量
        # ------------------------------------------------------------------ #
        g    = self._cache_g   # (NC, NQ) 或 (NC, n_sub, NQ)
        h    = self._cache_h   # (NC, NQ) 或 (NC, n_sub, NQ)

        # 激活集：当 g > -λ/μ 时约束激活，h = g；否则 h = -λ/μ，梯度为零
        mask = g > (-self.lamb / self.mu)  # 与 g、h 同形

        # ------------------------------------------------------------------ #
        # 第二步：计算 dP/d(σ^vM)，用于构造伴随载荷. 具体约束模型
        # 的 ∂g/∂σ^vM 由约束对象给出, 避免按有限元类型猜测约束形式.
        # ------------------------------------------------------------------ #
        dhdVM_val = self._stress_constraint.compute_gradient_wrt_von_mises(state)

        dPenaldVM = (self.lamb + self.mu * h) * bm.where(mask, dhdVM_val, 0.0)

        if enable_timing:
            t.send('罚函数偏导数')

        # ------------------------------------------------------------------ #
        # 第三步: 由约束接口计算固定状态时的显式偏导数 dP/dm_E.
        # 松弛形式及原生应力表示的差异由约束对象处理.
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

        dPenaldm_E_implicit = self._stress_constraint.compute_implicit_sensitivity_term(adjoint_vector, state)  # (NC,) 或 (NC, n_sub)
        
        # ------------------------------------------------------------------ #
        # 第五步：链式法则组装 dP/dρ
        # ------------------------------------------------------------------ #
        # 显式项对 NQ 维度求和，还原为单元级标量
        dPenaldm_E_explicit_reduced = bm.sum(dPenaldm_E_explicit, axis=-1)  # (NC,) 或 (NC, n_sub)

        # 计算 dm_E/dρ = (dE/dρ) / E₀
        dm_E_drho = self._interpolation_scheme.interpolate_material_derivative(
                                                            material=self._material, rho_val=density
                                                        ) / self._material.youngs_modulus  # (NC,) 或 (NC, n_sub)

        # 隐式项已由有限元适配器统一为对 m_E 的贡献, 包括混合元的 m_E^-2 因子.
        dP_drho = (dPenaldm_E_explicit_reduced + dPenaldm_E_implicit) * dm_E_drho

        # ------------------------------------------------------------------ #
        # 第六步：归一化并组装总梯度
        #   dJ/dρ = ∂f/∂ρ + (1/N) · dP/dρ
        # ------------------------------------------------------------------ #
        dVol_drho     = self._volume_objective.jac(density=density, state=state)
        n_constraints = g.numel() if hasattr(g, 'numel') else g.size
        dP_drho_norm = dP_drho / n_constraints

        # current_step = kwargs.get('iteration', 0) 
        # if current_step <= 1: 
        #     self._check_gradient_magnitude_balance(dVol_drho, dP_drho_norm, current_step)

        dJ_drho = dVol_drho + dP_drho_norm  # (NC,) 或 (NC, n_sub)

        if enable_timing:
            t.send('其他')
            t.send(None)

        return dJ_drho

    def lagrangian_jac(
        self,
        density: Union[Function, TensorLike],
        state: Optional[dict] = None,
    ) -> TensorLike:
        """计算原约束问题 Lagrangian 对物理密度的梯度.

        使用

        ``L = f_V + (1 / N_c) * sum(lambda_j * g_j)``

        中的当前 AL 乘子 ``lambda``. 该梯度不含增广罚项 ``mu`` 和截断函数
        ``h``，用于原问题的一阶最优性诊断，不能以 AL 子问题梯度替代.

        Parameters
        ----------
        density : Function or TensorLike
            当前物理密度.
        state : dict, optional
            与当前物理密度一致的状态解.

        Returns
        -------
        TensorLike
            原 Lagrangian 关于物理密度的梯度.
        """
        if state is None:
            state = {}
        self.fun(density=density, state=state)

        g = self._cache_g
        weights = bm.ones_like(g) * self.lamb
        dL_dVM = weights * self._stress_constraint.compute_gradient_wrt_von_mises(state)
        dL_dm_explicit = weights * self._stress_constraint.compute_partial_gradient_wrt_mE(
            state=state
        )

        adjoint_load = self._stress_constraint.compute_adjoint_load(
            dPenaldVM=dL_dVM,
            state=state,
        )
        adjoint_vector = self._analyzer.solve_adjoint(rhs=adjoint_load, rho_val=density)
        dL_dm_implicit = self._stress_constraint.compute_implicit_sensitivity_term(
            adjoint_vector,
            state,
        )

        dL_dm_explicit = bm.sum(dL_dm_explicit, axis=-1)
        dm_E_drho = self._interpolation_scheme.interpolate_material_derivative(
            material=self._material,
            rho_val=density,
        ) / self._material.youngs_modulus
        dconstraint_drho = (dL_dm_explicit + dL_dm_implicit) * dm_E_drho

        dvolume_drho = self._volume_objective.jac(density=density, state=state)
        n_constraints = g.numel() if hasattr(g, 'numel') else g.size
        return dvolume_drho + dconstraint_drho / n_constraints

    def _check_gradient_magnitude_balance(self, 
                                          dVol_drho: TensorLike, 
                                          dP_drho_norm: TensorLike, 
                                          step_k: int = 0) -> None:
        """
        验证体积梯度与惩罚项梯度是否在同一数量级，辅助校准 mu_0
        """
        # 计算最大绝对值 (无穷大范数)
        max_vol_grad = bm.max(bm.abs(dVol_drho))
        max_pen_grad = bm.max(bm.abs(dP_drho_norm))
        
        # 计算比值 (加入极小数避免除零报错)
        ratio = max_pen_grad / (max_vol_grad + 1e-12)
        
        discretization = getattr(self._stress_constraint, "discretization_name", "应力约束")
        print(f"\n[{discretization}] --- ALM 迭代步 {step_k} 梯度量级诊断 ---")
        print(f"最大体积梯度 ||dVol_drho||_inf   : {max_vol_grad:.4e}")
        print(f"最大惩罚梯度 ||dP_drho_norm||_inf: {max_pen_grad:.4e}")
        print(f"梯度量级比值 (Penal / Vol)       : {ratio:.4f}")
        
        # 给出学术建议
        if ratio < 0.1:
            print("👉 诊断结论：惩罚力【过弱】。优化器可能会无视应力约束疯狂挖洞。")
            print("💡 调整建议：请成倍【增大】初始惩罚因子 mu_0。")
        elif ratio > 10.0:
            print("👉 诊断结论：惩罚力【过强】。应力惩罚项将主导优化，可能导致拓扑演化停滞或全灰。")
            print("💡 调整建议：请成倍或按数量级【减小】初始惩罚因子 mu_0。")
        else:
            print("👉 诊断结论：量级【完美平衡】！(理想范围 0.1 ~ 10.0)")
            print("💡 调整建议：保持当前 mu_0 不变。")
        print("-" * 50 + "\n")

    def update_multipliers(self) -> None:
        """更新拉格朗日乘子 λ 和 罚因子 μ.

        此方法应在每一轮 ALM 外层迭代结束时调用.

        乘子更新采用带安全阈的投影形式 (safeguarded augmented Lagrangian,
        Andreani, Birgin, Martinez & Schuverdt 2007; Birgin & Martinez 2014):

            λ^(k+1) = P_[0, λ_max](λ^(k) + μ^(k) h),

        其中 h = max(g, -λ/μ) 保证 λ^(k) + μ^(k) h >= 0, 下界投影自动成立, 只需
        对上界截断. ``lambda_max`` 为 None 时退化为无阈更新 (今日行为). 有阈时,
        被约束长期轻微违反且设计无法改动的单元 (滤波尾部灰度单元) 上 λ 不再随
        外层步线性爬升: 若约束可满足, 极限点仍是 KKT 点; 若不可满足, 迭代收敛到
        不可行度的驻点而非发散.
        """
        if self._cache_h is None:
            raise RuntimeError(
                "update_multipliers() 必须在 fun() 之后调用, "
                "以确保 h 已被计算并缓存."
            )
    
        # 1. 更新拉格朗日乘子 λ
        # λ^(k+1) = λ^(k) + μ^(k) · h
        self.lamb = self.lamb + self.mu * self._cache_h
        lambda_max = getattr(self, 'lambda_max', None)
        if lambda_max is not None:
            capped = self.lamb > lambda_max
            self.last_capped_count = int(bm.sum(capped))
            self.lamb = bm.minimum(self.lamb, lambda_max)
        else:
            self.last_capped_count = 0

        # 2. 更新罚因子 μ
        # μ^(k+1) = min(α · μ^(k), μ_max) [cite: 303]
        if self._should_grow_penalty():
            self.mu = min(self._options.alpha * self.mu, self.mu_max)

    def set_current_violation(self, value) -> None:
        """由优化器在 update_multipliers() 之前推入本外层步的最大相对超限量.

        目标函数自身没有 rho / state, 无法调用 compute_relative_violation,
        故违反度必须从优化器侧推入, 而不是在此重建. 取值在 0 处截断:
        真正可行后 v = 0, 条件规则下 mu 冻结.
        """
        self._pending_violation = None if value is None else max(float(value), 0.0)

    def _should_grow_penalty(self) -> bool:
        """判断本外层步是否放大罚因子 mu.

        'unconditional' (默认) 复现今日行为: 每个外层步无条件放大.
        'conditional': 仅当不可行度未取得足够下降 (v > tau * v_prev) 时放大.
        """
        rule = getattr(self._options, 'mu_update_rule', 'unconditional')
        if rule != 'conditional':
            return True

        viol = self._pending_violation
        prev = self._prev_violation
        self._prev_violation = viol if viol is not None else prev
        self._pending_violation = None

        if viol is None or prev is None:
            return True

        tau = getattr(self._options, 'mu_violation_ratio', 0.5)

        return viol > tau * prev
