from typing import Optional, Literal, Union, Dict, TYPE_CHECKING
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.volume_objective import VolumeObjective
from soptx.optimization.stress_constraint import StressConstraint 
from soptx.utils.base_logged import BaseLogged

# 使用 TYPE_CHECKING 避免循环导入，仅用于类型提示
if TYPE_CHECKING:
    from soptx.optimization.al_mma_optimizer import ALMMMAOptions

class AugmentedLagrangianObjective(BaseLogged):
    def __init__(self,
                volume_objective: VolumeObjective, 
                stress_constraint: StressConstraint,
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
        - g_j 为多项式消失约束, 由 StressConstraint 计算
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

        # 2. 计算消失约束 g
        g = self._stress_constraint.fun(density, state) # 单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        # 3. 计算 ALM 的 h 和 Penal
        #    h_j = max(g_j, -lambda_j / mu)
        h = bm.maximum(g, -self.lamb / self.mu) # 单分辨率: (NC, NQ) | 多分辨率: (NC, n_sub, NQ)

        penal = bm.sum(self.lamb * h + 0.5 * self.mu * h**2)

        self._cache_g = g
        self._cache_h = h

        # 4. 组装归一化后的增广拉格朗日目标函数 J
        J = f + penal / self._NC

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
                        enable_timing: bool = False, 
                        **kwargs
                    ) -> TensorLike:
        # --- 缓存检查与状态同步 ---
        if (self._cache_g is None) or ('stiffness_ratio' not in state) or ('stress_deviation' not in state):
            self.fun(density, state)
        
        # --- 获取缓存的物理量 ---
        slim = self._stress_constraint._stress_limit
        E = state['stiffness_ratio']  # (NC, )
        s = state['stress_deviation'] # (NC, NQ)

        g = self._cache_g
        h = self._cache_h

        # --- 确定激活集 a1 (Mask) ---
        #  逻辑: 当 g > -lambda/mu 时，h = g, 此时约束激活（或违反）
        limit_term = -self.lamb / self.mu
        mask = g > limit_term  # (NC, NQ) bool

        # --- 计算显式灵敏度 ---
        #  计算 dPenaldVM (罚函数对 Von Mises 应力的偏导数)
        dhdVM_val = E[:, None] * (3 * s**2 + 1) / slim
        dhdVM = bm.where(mask, dhdVM_val, 0.0)
        dPenaldVM = (self.lamb + self.mu * h) * dhdVM # (NC, NQ)

        #  计算 dPenal/dE 的显式部分
        dgdE = self._stress_constraint.compute_partial_gradient_wrt_stiffness(state=state)
        dPenaldE_explicit = bm.where(mask, (self.lamb + self.mu * h) * dgdE, 0.0) # (NC, NQ)

        # --- 伴随法 ---        
        #  计算伴随载荷 F_adj = - (dVM/dU)^T * dPenaldVM
        adjoint_load = self._stress_constraint.compute_adjoint_load(dPenaldVM=dPenaldVM, state=state) # (gdofs, )
        
        # 解伴随方程: K * psi = F_adj
        adjoint_vector = self._analyzer.solve_adjoint(rhs=adjoint_load, rho_val=density) # (gdofs, )
        
        # --- 计算隐式灵敏度 ---
        # dPenal/dE_implicit = psi^T * (dF_int / dE)
        dPenaldE_implicit = self._stress_constraint.compute_implicit_sensitivity_term(adjoint_vector, state)  # (NC, )

        # --- 总灵敏度 (关于刚度变量 E) ---
        # 归约显式项 (NC, NQ) -> (NC, )
        dPenaldE_explicit_reduced = bm.sum(dPenaldE_explicit, axis=1)
        dPenaldE_total = dPenaldE_explicit_reduced + dPenaldE_implicit # (NC, )

        # --- 链式法则: dPenal/drho = dPenal/dE * dE/drho ---
        # 获取 dE/drho (取决于具体的插值模型)
        dE_drho_absolute = self._interpolation_scheme.interpolate_material_derivative(
                                                    material=self._material, 
                                                    rho_val=density
                                                )  # (NC, )
        E0 = self._material.youngs_modulus
        dE_drho = dE_drho_absolute / E0

        dP_drho = dPenaldE_total * dE_drho

        # 计算主目标函数 (体积) 的梯度
        dVol_drho = self._volume_objective.jac(density=density, state=state) # (NC, )

        # 归一化处理
        dP_drho_normalized = dP_drho / self._NC

        # 组装总拉格朗日函数的梯度 (关于物理密度 rho)
        dJ_drho = dVol_drho + dP_drho_normalized # (NC, )

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
    