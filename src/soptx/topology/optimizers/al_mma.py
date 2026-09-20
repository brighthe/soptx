from dataclasses import dataclass
from typing import Union, Tuple, Optional
from time import time
import math
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.core import timer
from soptx.topology.filters import Filter
from soptx.topology.objectives import AugmentedLagrangianObjective
from soptx.topology.optimizers.history import OptimizationHistory
from soptx.topology.optimizers.mma import MMAOptions, MMAOptimizer

@dataclass
class ALMMMAOptions(MMAOptions):
    """专门针对 ALM-MMA 双层优化算法的配置选项"""
    # =========================================================================
    # 1. 覆盖父类默认值 (应力问题更保守)
    # =========================================================================
    change_tolerance: float = 0.002         # C1 阈值 (Tol), 度量对象由 change_measure 决定
    move_limit: float = 0.15                # 更保守的移动限制
    asymp_init: float = 0.2                 # 更紧的初始渐近线距离
    asymptote_min_distance: float = 1.0e-4  # 渐近线距当前点的最小距离, 相对全局设计区间;
                                            # 仅防 U-L 下溢, 不得高到压过振荡收缩 (见 _solve_unconstrained_subproblem)
    use_penalty_continuation: bool = False  # ALM 框架下不使用 SIMP 连续化

    # =========================================================================
    # 2. ALM 外层循环控制
    # =========================================================================
    max_al_iterations: int = 150           # ALM 外层步数
    mma_iters_per_al: int = 5              # 每个 ALM 步中 MMA 内层迭代次数
    stress_tolerance: float = 0.003        # 应力约束容差 (TolS)
    # C2 验收子集: 非 None 时只在 rho_phys >= 阈值的 (未豁免) 单元上取最大相对违
    # 反度; None 复现全域口径。2026-09-18 起用于排除滤波尾部不承载的灰度单元
    # (g 长期停在容差附近略偏正, 设计无法改动), 全域值仍作诊断记录。
    acceptance_solid_threshold: Optional[float] = None
    hold_steps: int = 3                    # 停止准则 C3: 需连续满足 C1/C2 的 ALM 外层步数
    inner_stop_rule: str = 'legacy'        # 'legacy' | 'projected_gradient'
    inner_relative_tolerance: float = 0.1  # 相对外层初始残差的验收比例
    inner_absolute_tolerance: float = 1e-6 # 显式内层绝对容差

    # =========================================================================
    # 3. 渐近线振荡控制 (与父类 asymp_incr/asymp_decr 配合使用)
    # =========================================================================
    osc: float = 0.2                       # 振荡控制参数

    # =========================================================================
    # 4. 增广拉格朗日罚参数
    # =========================================================================
    mu_0: float = 10.0                     # 初始罚因子 μ^(0)
    mu_max: float = 10000.0                # 最大罚因子 μ_max
    # 乘子安全阈 lambda_max (safeguarded AL): lambda <- P_[0, lambda_max](lambda + mu h);
    # None = 无上限 (今日行为)。见 AugmentedLagrangianObjective.update_multipliers。
    lambda_max: Optional[float] = None
    alpha: float = 1.1                     # 罚因子更新参数 α > 1
    lambda_0_init_val: float = 0.0         # 初始拉格朗日乘子标量值 λ^(0)

    # =========================================================================
    # 5. 阈值投影连续化参数 (对应 opt.contB = [BFreq, B0, Binc, Bmax])
    # =========================================================================
    beta_freq: int = 5                     # 每隔多少 AL 步更新一次 β
    beta_init: float = 1.0                 # β 初始值
    beta_incr: float = 1.0                 # β 每次增量
    beta_max: float = 10.0                 # β 最大值

    # --- 末期移动限制衰减 (默认关闭, 复现今日行为) ---
    move_limit_decay: float = 1.0            # 1.0 = 关闭
    move_limit_min: float = 0.005            # 衰减下限, 必须严格高于 change_tolerance
    move_limit_progress_window: int = 10     # 判别窗口 W (外层步)
    move_limit_progress_ratio: float = 0.3   # 相干行程份额阈值 p_min
    move_limit_progress_cell: float = 0.7    # 单元被判为相干的 P_cell 阈值

    # --- 收敛判据 C1 的度量对象 ---
    # 'mean': 外层步首末设计变量的平均绝对变化 (PolyStress 口径, 2026-09-11 起默认由
    #         cases.toml 指定); 'design' | 'physical': 逐 MMA 步的最大绝对变化 (对照)。
    change_measure: str = 'design'           # 'mean' | 'design' | 'physical'

    # --- 罚参数 mu 的放大规则 ---
    mu_update_rule: str = 'unconditional'    # 'unconditional'(今日) | 'conditional'
    mu_violation_ratio: float = 0.5          # tau

    # --- 原问题 KKT 诊断与验收 ---
    # 默认关闭, 避免每个候选收敛外层步额外求解一次伴随方程. 只开 diagnostics
    # 时终态报告残差但不参与停机; acceptance 还要求显式给出三个正容差.
    kkt_diagnostics_enabled: bool = False
    kkt_acceptance_enabled: bool = False
    kkt_stationarity_tolerance: float = 0.0
    kkt_complementarity_tolerance: float = 0.0
    kkt_dual_tolerance: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.asymptote_min_distance) or not 0 < self.asymptote_min_distance < 10:
            raise ValueError("asymptote_min_distance 必须是位于 (0, 10) 的有限数.")
        if self.inner_stop_rule not in ('legacy', 'projected_gradient'):
            raise ValueError("inner_stop_rule 必须是 'legacy' 或 'projected_gradient'.")
        if not math.isfinite(self.inner_relative_tolerance) or not 0 < self.inner_relative_tolerance < 1:
            raise ValueError("inner_relative_tolerance 必须是位于 (0, 1) 的有限数.")
        if not math.isfinite(self.inner_absolute_tolerance) or self.inner_absolute_tolerance <= 0:
            raise ValueError("inner_absolute_tolerance 必须是有限正数.")
        if self.inner_stop_rule == 'projected_gradient' and self.use_penalty_continuation:
            raise ValueError("projected_gradient 内层停止规则要求关闭 use_penalty_continuation.")
        if (self.inner_stop_rule == 'projected_gradient'
                and (isinstance(self.mma_iters_per_al, bool)
                     or not isinstance(self.mma_iters_per_al, int)
                     or self.mma_iters_per_al < 1)):
            raise ValueError("projected_gradient 内层停止规则要求 mma_iters_per_al 为正整数.")
        if not 0 < self.move_limit_decay <= 1:
            raise ValueError("move_limit_decay 必须位于 (0, 1].")
        if self.move_limit_decay < 1.0 and not (self.change_tolerance < self.move_limit_min <= self.move_limit):
            raise ValueError("启用移动限制衰减时要求 change_tolerance < move_limit_min <= move_limit.")
        if isinstance(self.move_limit_progress_window, bool) or not isinstance(self.move_limit_progress_window, int) or self.move_limit_progress_window < 2:
            raise ValueError("move_limit_progress_window 必须是不小于 2 的整数.")
        if not 0 <= self.move_limit_progress_ratio <= 1:
            raise ValueError("move_limit_progress_ratio 必须位于 [0, 1].")
        if not 0 < self.move_limit_progress_cell <= 1:
            raise ValueError("move_limit_progress_cell 必须位于 (0, 1].")
        if self.change_measure not in ('mean', 'design', 'physical'):
            raise ValueError("change_measure 必须是 'mean', 'design' 或 'physical'.")
        if self.mu_update_rule not in ('unconditional', 'conditional'):
            raise ValueError("mu_update_rule 必须是 'unconditional' 或 'conditional'.")
        if not 0 < self.mu_violation_ratio < 1:
            raise ValueError("mu_violation_ratio 必须位于 (0, 1).")
        if self.lambda_max is not None and not (
                math.isfinite(self.lambda_max) and self.lambda_max > 0):
            raise ValueError("lambda_max 必须是有限正数或 None.")
        if self.acceptance_solid_threshold is not None and not (
                math.isfinite(self.acceptance_solid_threshold)
                and 0 < self.acceptance_solid_threshold < 1):
            raise ValueError("acceptance_solid_threshold 必须位于 (0, 1) 或为 None.")
        if self.kkt_acceptance_enabled and not self.kkt_diagnostics_enabled:
            raise ValueError("启用 KKT 验收前必须启用 KKT 诊断.")
        kkt_tolerances = (
            self.kkt_stationarity_tolerance,
            self.kkt_complementarity_tolerance,
            self.kkt_dual_tolerance,
        )
        if self.kkt_acceptance_enabled and any(value <= 0 for value in kkt_tolerances):
            raise ValueError("启用 KKT 验收时必须显式给出三个正的 KKT 容差.")
        if any(not math.isfinite(value) or value < 0 for value in kkt_tolerances):
            raise ValueError("KKT 容差必须是有限的非负数.")

def coherent_travel_share(window, cell_ratio: float = 0.7) -> float:
    """窗口内由"相干运动"单元承担的行程份额, 取值 [0, 1].

    window 是长度 W+1 的物理密度快照序列 (按外层步递增). 对每个单元计算
    净位移与累计行程之比 P_cell; P_cell > cell_ratio 视为相干 (单调重组),
    返回这些单元的行程占窗口总行程的比例.

    不用全局的 net/travel: 全局 L1 比值等于按行程加权的 P_cell 均值, 少数
    单元的单调重组会被大量单元的非相干抖动淹没 (已在盘上两条轨迹上验证不可分).
    period-2 极限环返回接近 0; 单调重组返回接近 1; 窗口不足或完全静止返回 1.0
    (即不触发衰减, 取保守侧).
    """
    if window is None or len(window) < 2:
        return 1.0

    travel = bm.abs(window[1] - window[0])
    for j in range(1, len(window) - 1):
        travel = travel + bm.abs(window[j + 1] - window[j])

    total = float(bm.sum(travel))
    if total <= 1e-30:
        return 1.0

    net = bm.abs(window[-1] - window[0])
    p_cell = net / bm.maximum(travel, bm.full_like(travel, 1e-30))
    coherent = bm.where(p_cell > cell_ratio, travel, bm.zeros_like(travel))

    return float(bm.sum(coherent)) / total


def projected_gradient_residual(
    design: TensorLike,
    gradient: TensorLike,
    passive_mask: Optional[TensorLike] = None,
) -> float:
    """计算全局设计盒 ``[0, 1]`` 上的 projected-gradient 无穷范数.

    被动单元不属于优化变量, 因而从残差中剔除. 该量用于固定
    ``lambda``、``mu`` 与投影参数的 AL 内层子问题, 不代表原问题 KKT 残差.
    """
    projected = design - bm.clip(design - gradient, 0.0, 1.0)
    if passive_mask is not None:
        projected = projected[~passive_mask]
    if projected.shape[0] == 0:
        return 0.0
    return float(bm.max(bm.abs(projected)))


def _stable_mma_candidate(
    lower: TensorLike,
    upper: TensorLike,
    p: TensorLike,
    q: TensorLike,
) -> TensorLike:
    """用无消减形式计算无约束 MMA 子问题的驻点候选.

    原式的分子、分母在 ``p`` 接近 ``q`` 时同时趋近于零. 令
    ``a=sqrt(p)``, ``b=sqrt(q)``, 消去公共因子 ``a-b`` 后得到
    ``(lower*a + upper*b)/(a+b)``. ``a+b=0`` 时取其中点作为对称回退值.
    """
    sqrt_p = bm.sqrt(p)
    sqrt_q = bm.sqrt(q)
    denominator = sqrt_p + sqrt_q
    positive = denominator > 0.0
    safe_denominator = bm.where(
        positive,
        denominator,
        bm.ones_like(denominator),
    )
    weighted = (lower * sqrt_p + upper * sqrt_q) / safe_denominator
    midpoint = 0.5 * (lower + upper)
    return bm.where(positive, weighted, midpoint)


def _bound_mma_asymptotes(
    current: TensorLike,
    lower: TensorLike,
    upper: TensorLike,
    *,
    span: float,
    min_distance: float,
) -> Tuple[TensorLike, TensorLike]:
    """限制 MMA 渐近线相对当前点的距离, 不修改设计变量的盒约束或移动限制."""
    lower = bm.minimum(
        bm.maximum(lower, current - 10.0 * span),
        current - min_distance * span,
    )
    upper = bm.maximum(
        bm.minimum(upper, current + 10.0 * span),
        current + min_distance * span,
    )
    return lower, upper


class ALMMMAOptimizer(MMAOptimizer):
    def __init__(self,
                al_objective: AugmentedLagrangianObjective,
                filter: Filter,
                options: ALMMMAOptions = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        基于增广拉格朗日法的 MMA 双层优化器 (复刻 PolyStress 逻辑)

        双层结构:
          外层 (ALM): 更新 Lagrange 乘子 lambda 和惩罚因子 mu
          内层 (MMA): 对给定 lambda, mu 求解无约束子问题 min J(z)
        """
        if options is None:
            options = ALMMMAOptions()
        elif not isinstance(options, ALMMMAOptions):
            raise TypeError(
                "options 必须是 ALMMMAOptions 实例."
            )
            
        # 调用父类初始化，传入空约束列表 []，使得 MMA 子问题退化为无约束问题 (m=0)
        super().__init__(objective=al_objective, 
                        constraint=[], 
                        filter=filter, 
                        options=options, 
                        enable_logging=enable_logging, 
                        logger_name=logger_name)
        
        self._al_objective = al_objective

        # 动态渐近线参数
        self._asym_inc_dynamic = self.options.asymp_incr
        self._asym_decr_dynamic = self.options.asymp_decr

    def _effective_move_limit(self) -> float:
        """当前生效的移动限制. 未启用衰减时恒等于 options.move_limit."""
        base = getattr(self.options, 'move_limit', ALMMMAOptions.move_limit)
        current = getattr(self, '_move_limit_now', None)

        return float(base) if current is None else float(current)

    def _update_penalty(self, iter_idx: int) -> None:
        """重写父类的连续化技术：自动捕获插值方案中的目标惩罚因子作为上限"""
        if not self.options.use_penalty_continuation:
            return
        
        interpolation_scheme = self._al_objective._analyzer.interpolation_scheme
        
        # 第一次调用时，把用户初始设置的 penalty_factor 保存为目标上限
        if not hasattr(self, '_target_penalty'):
            self._target_penalty = interpolation_scheme.penalty_factor
            self._log_info(f"开启惩罚延续策略: 目标最大惩罚因子捕获为 {self._target_penalty}")
        
        # 按全局迭代步数计算当前的惩罚因子
        penalty_update = iter_idx // 30
        current_penalty = min(1.0 + penalty_update * 0.5, self._target_penalty)
        
        # 将更新后的惩罚因子写回插值方案中
        if current_penalty != interpolation_scheme.penalty_factor:
            interpolation_scheme.penalty_factor = current_penalty

    def optimize(self,
                design_variable: Union[Function, TensorLike], 
                density_distribution: Union[Function, TensorLike], 
                enable_timing: bool = None,
                **kwargs
            ) -> Tuple[Union[Function, TensorLike], OptimizationHistory]:
        
        analyzer = self._al_objective._analyzer
        opts = self.options
        
        # --- 问题规模初始化 (constraint=[] -> m=0) ---
        m = 0 
        n = design_variable.shape[0]
        opts.initialize_problem_params(m, n)

        # --- 变量初始化 ---
        if isinstance(design_variable, Function):
            dv = design_variable.space.function(bm.copy(design_variable[:]))
        else:
            dv = bm.copy(design_variable[:])
        
        if isinstance(density_distribution, Function):
            rho = density_distribution.space.function(bm.copy(density_distribution[:]))
        else:
            rho = bm.copy(density_distribution[:])

        # 初始物理密度
        rho_phys = self._filter.get_initial_density(density=rho)

        # --- 被动单元处理 ---
        pde = analyzer.pde
        passive_mask = None
        if hasattr(pde, 'get_passive_element_mask'):
            design_mesh = getattr(self._filter, 'design_mesh', None)
            passive_mask = pde.get_passive_element_mask(mesh=design_mesh)

        if passive_mask is not None:
            dv[passive_mask] = 1.0

        xold1 = bm.copy(dv)
        xold2 = bm.copy(dv)
        
        self.history = OptimizationHistory()
        # 记录第 0 步构型: rho_phys 已含滤波与 passive solid 覆写, 即优化的真实起点
        self.history.log_initial(rho_phys)
        global_iter = 0

        # 停止准则状态: hold_count 记录连续满足 C0-C2 的 ALM 外层步数 (C3);
        # converged/termination_reason 是终止判据的唯一出处, 供 driver 汇总,
        # 避免消费端各自用单条件重新拼一个收敛标志。
        hold_count = 0
        self.converged = False
        self.termination_reason = 'not-started'
        change = 2 * opts.change_tolerance
        max_relative_violation = float('inf')
        max_relative_violation_solid = float('inf')
        # 6.4 节诊断量: 乘子相对变化, 只报告不进判据。
        self.last_multiplier_change = float('nan')
        # 末期移动限制衰减的路径相关状态 (存在 optimizer 实例上, 不回写 options:
        # _build_al_options 被调用两次产出两个对象, 改一个会静默失步)。
        self._move_limit_now = float(getattr(opts, 'move_limit', ALMMMAOptions.move_limit))
        self._move_limit_decays = 0
        self._c0_first_iter = None
        change_phys = 2 * opts.change_tolerance
        change_mean = 2 * opts.change_tolerance
        change_outer_mean = 2 * opts.change_tolerance
        self.last_change_outer_mean = float('nan')
        self.last_change_outer_max = float('nan')
        self.last_kkt_diagnostics = None
        self.outer_history = []
        self.last_inner_diagnostics = None
        inner_iteration_failed = False
        rho_window = []
        # 状态复用 (算法 2 步骤 2/4): 内层步末已对新设计求解, 下一步头部直接沿用;
        # 只有刚度 (罚指数连续化) 或物理密度 (beta 更新后重过滤) 变化时才重解。
        # AL 参数 (lambda, mu) 只影响 Phi 不影响状态, 乘子更新后只需重算目标缓存。
        state = None
        state_penalty = None
        objective_valid = False

        # =========================================================================
        # 外层循环 (ALM 步) - 控制 λ 和 μ
        # =========================================================================
        for al_iter in range(opts.max_al_iterations):
            # _epoch 使用 AL 步计数 (非全局迭代计数)
            # 控制渐近线初始化: 前 2 个 AL 步使用 AsymInit
            self._epoch = al_iter + 1

            inner_stop_rule = getattr(opts, 'inner_stop_rule', 'legacy')
            use_projected_gradient = inner_stop_rule == 'projected_gradient'
            inner_initial_residual = None
            inner_final_residual = None
            inner_target = None
            inner_steps = 0
            inner_accepted = not use_projected_gradient
            outer_mu = float(self._al_objective.mu)
            beta_value = getattr(self._filter, 'beta', None)
            outer_beta = None if beta_value is None else float(beta_value)

            change = 2 * opts.change_tolerance
            # 外层步起点设计 rho^{k,0}: C1 的 'mean' 度量按外层步首末设计计。
            dv_outer_start = bm.copy(dv[:])

            # 初始化为一个远大于 1.0+TolS 的“魔法数字”（防守型编程）
            # 作用：作为逻辑屏障，人为伪造一个“结构严重超载”的初始假象。
            # 这能强制逼迫内层 MMA 循环去调用有限元求解器计算真实应力，
            # 绝对防止在获取真实应力数据前意外触发底部的全局收敛条件（避免过早假收敛）。
            max_stress_measure = 2.0 
            
            # =====================================================================
            # 内层循环 (MMA 步) - 对当前的 lambda, mu 进行极小化
            # =====================================================================
            for mma_iter in range(opts.mma_iters_per_al):
                t = None
                if enable_timing:
                    t = timer(f"拓扑优化单次迭代")
                    next(t)

                start_time = time()

                # --- 惩罚因子更新 (基于全局迭代步数) ---
                self._update_penalty(iter_idx=global_iter)
                current_penalty = analyzer.interpolation_scheme.penalty_factor

                # 更新迭代计数
                global_iter += 1
                
                # --- 计算增广拉格朗日目标函数及灵敏度 ---
                # a. 状态求解 (FEA): 沿用上一内层步末对当前设计的分析, 刚度或物理
                #    密度变化后重解。
                if state is None or state_penalty != current_penalty:
                    state = analyzer.solve_state(rho_val=rho_phys)
                    state_penalty = current_penalty
                    objective_valid = False
                if enable_timing:
                    t.send('求解')
                
                # b. 评估增广拉格朗日目标函数及其物理敏度: AL 参数变化后重算缓存
                #    (_cache_g/_cache_h 与体积缓存), 供 jac 使用。
                if not objective_valid:
                    self._al_objective.fun(density=rho_phys, state=state)
                    objective_valid = True
                if enable_timing:
                    t.send('目标函数计算')
                dJ_drho = self._al_objective.jac(density=rho_phys, state=state)
                if enable_timing:
                    t.send('目标函数灵敏度分析')
                
                # c. 通过过滤器应用链式法则 (获取对设计变量 dv 的敏度)
                dJ_dv = self._filter.filter_objective_sensitivities(design_variable=dv, obj_grad_rho=dJ_drho)
                if enable_timing:
                    t.send('灵敏度过滤')

                # 从缓存中读取体积分数
                volfrac = self._al_objective._volume_objective._v / self._al_objective._volume_objective._v0
                
                # 被动单元灵敏度置零
                if passive_mask is not None:
                    dJ_dv[passive_mask] = 0.0

                if use_projected_gradient:
                    current_residual = projected_gradient_residual(
                        design=dv,
                        gradient=dJ_dv,
                        passive_mask=passive_mask,
                    )
                    if inner_initial_residual is None:
                        inner_initial_residual = current_residual
                        inner_target = max(
                            float(getattr(opts, 'inner_absolute_tolerance', 1e-6)),
                            float(getattr(opts, 'inner_relative_tolerance', 0.1))
                            * inner_initial_residual,
                        )
                    elif current_residual <= inner_target:
                        # global_iter/history 只统计已经接受的 MMA 更新；本次只评估
                        # 了接受态残差，没有生成新的候选设计。
                        global_iter -= 1
                        inner_final_residual = current_residual
                        inner_accepted = True
                        if enable_timing:
                            t.send(None)
                        break

                # --- 求解无约束 MMA 子问题 ---
                if use_projected_gradient:
                    # PolyStress 的 legacy 路径按 AL 外层步更新 _epoch。新的
                    # fixed-AL 内层模式按已接受的 MMA 更新计数，使第三次更新起
                    # 能使用历史渐近线；这是一项内层算法选择，不改变 legacy。
                    self._epoch = global_iter
                if passive_mask is not None:
                    dv_new = bm.copy(dv)
                    active_mask = ~passive_mask
                    dv_new[active_mask] = self._solve_unconstrained_subproblem(
                                                    dfdz=dJ_dv[active_mask],
                                                    z=dv[active_mask],
                                                    zold1=xold1[active_mask],
                                                    zold2=xold2[active_mask],
                                                )
                else:
                    dv_new = self._solve_unconstrained_subproblem(
                                        dfdz=dJ_dv, z=dv, zold1=xold1, zold2=xold2,
                                    )
                if enable_timing:
                    t.send('MMA 优化')

                # --- 密度过滤与设计更新 (算法 2 步骤 3) ---
                # rho_phys 才是进 FEA / 体积 / 应力约束与全部出图的场, 物理变化量
                # 必须在下一行重绑之前算。
                rho_prev = bm.copy(rho_phys[:])
                rho_phys = self._filter.filter_design_variable(
                    design_variable=dv_new, physical_density=rho_phys)
                if enable_timing:
                    t.send('密度过滤')
                change = float(bm.max(bm.abs(dv_new - dv)))
                change_phys = float(bm.max(bm.abs(rho_phys[:] - rho_prev)))
                # 相邻两次 MMA 更新间设计变量的平均绝对变化, 只用于内层提前退出。
                change_mean = float(bm.mean(bm.abs(dv_new - dv)))
                xold2, xold1 = xold1, bm.copy(dv)
                dv = dv_new

                # --- 对新设计重新分析 (算法 2 步骤 4) ---
                # 内层退出判据、历史记录、外层 C2 与乘子更新全部使用新设计的约束值;
                # 该状态同时供下一内层步头部复用, 每次 MMA 更新仍只求解一次。
                state = analyzer.solve_state(rho_val=rho_phys)
                state_penalty = current_penalty
                J_val = float(self._al_objective.fun(density=rho_phys, state=state))
                objective_valid = True
                if enable_timing:
                    t.send('新设计求解与目标评估')
                volfrac = self._al_objective._volume_objective._v / self._al_objective._volume_objective._v0
                max_constraint = float(bm.max(self._al_objective._cache_g))
                SM = self._al_objective._stress_constraint.compute_stress_measure(
                                                rho=rho_phys, state=state)
                max_vm_stress = float(bm.max(SM))
                relative_violation = self._al_objective._stress_constraint.compute_relative_violation(
                    rho=rho_phys, state=state)
                max_relative_violation = float(bm.max(relative_violation))
                # C2 实际比较的量: 实体子集 (rho_phys >= acceptance_solid_threshold)
                # 上的最大相对违反度; 阈值为 None 时等于全域值。豁免单元已由约束
                # 返回 -1, 无需再处理。
                max_relative_violation_solid = self._solid_max_relative_violation(
                    relative_violation, rho_phys, max_relative_violation)
                max_multiplier = float(bm.max(self._al_objective.lamb))
                # 未加权实体应力比: 只作诊断打印, 不参与 C2。加权验收量会被空洞
                # 单元的 m_E ~ 1e-9 压到阈值之下, 未加权量才反映实体材料真实应
                # 力水平。见方法说明 (dut-postdoc) 第 6.4 节。
                solid_ratio_fn = getattr(
                    self._al_objective._stress_constraint,
                    'compute_solid_stress_ratio', None)
                max_solid_stress_ratio = (
                    float('nan') if solid_ratio_fn is None
                    else float(bm.max(solid_ratio_fn(rho=rho_phys, state=state))))

                iteration_time = time() - start_time

                dJ_norm = float(bm.linalg.norm(dJ_dv))
                if enable_timing:
                    t.send('后处理')
                    t.send(None)

                self._log_info(
                        f"It:{al_iter+1:3d}_{mma_iter+1:1d} "
                        f"Obj: {volfrac:.6f} "
                        f"Max_App: {max_vm_stress:.6f} "
                        f"Max_Rel: {max_relative_violation:.6f} "
                        f"Max_Rel_S: {max_relative_violation_solid:.6f} "
                        f"Max_Solid: {max_solid_stress_ratio:.6f} "
                        f"|dJ|: {dJ_norm:.6f} "
                        f"Ch/Tol: {change/opts.change_tolerance:.6f} "
                        f"mu: {self._al_objective.mu:.4e} "
                        f"lam_max: {max_multiplier:.4e} "
                        f"p: {current_penalty:.1f} "
                        f"Time: {iteration_time:.3f} sec "
                    )
                
                # 调用确切的 log_iteration 接口保存历史数据: 密度帧、标量与应力场
                # 同属 MMA 更新后并已重新分析的设计。
                self.history.log_iteration(
                        iter_idx=global_iter,
                        change=change,                              # 设计变量最大绝对变化量
                        time_cost=iteration_time,                   # 本次迭代耗时
                        physical_density=rho_phys,                  # 密度场
                        scalars={
                            'al_objective': float(J_val),
                            'volfrac': volfrac,                     # 体积分数 (目标函数)
                            'max_constraint': max_constraint,
                            'max_apparent_stress_ratio': max_vm_stress,
                            'max_relative_violation': max_relative_violation,
                            'max_relative_violation_solid': max_relative_violation_solid,  # C2 实际比较的量
                            'max_multiplier': max_multiplier,       # max_e lambda_e, 验证乘子有界
                            'max_von_mises': max_vm_stress,         # 归一化的最大 von Mises 应力场
                            'change_physical': change_phys,         # 物理密度最大绝对变化
                            'change_mean': change_mean,             # 设计变量平均绝对变化 (逐 MMA 步)
                            'move_limit': self._effective_move_limit(),
                            'mu': float(self._al_objective.mu),
                        },
                        fields={
                            'von_mises_stress': SM,          # 表观应力比, 不用于可行性判断
                        },
                )
                inner_steps += 1

                # 内层收敛判定: 最大设计变化及相对约束超限均达标
                # change_measure 决定 C1 度量的对象, 默认 'design' 复现今日行为。
                measure = getattr(opts, 'change_measure', 'design')
                change_c1 = (change_mean if measure == 'mean'
                             else change_phys if measure == 'physical'
                             else change)
                if (not use_projected_gradient
                        and change_c1 < opts.change_tolerance
                        and max_relative_violation_solid <= opts.stress_tolerance):
                        break # 跳出内层循环，进入 ALM 更新      

            if use_projected_gradient:
                if not inner_accepted:
                    # 硬预算耗尽后, 在内层末设计 (步骤 4 已分析并评估目标) 上补一次
                    # 伴随评估, 避免用候选步起点的梯度验收内层子问题。
                    dJ_drho = self._al_objective.jac(density=rho_phys, state=state)
                    dJ_dv = self._filter.filter_objective_sensitivities(
                        design_variable=dv,
                        obj_grad_rho=dJ_drho,
                    )
                    if passive_mask is not None:
                        dJ_dv[passive_mask] = 0.0
                    inner_final_residual = projected_gradient_residual(
                        design=dv,
                        gradient=dJ_dv,
                        passive_mask=passive_mask,
                    )
                    inner_accepted = inner_final_residual <= inner_target

                # PG 失败会在本外层更新之前退出；先固定本外层首末变化，避免
                # summary 沿用上一外层值或在首外层报告 NaN。
                change_outer_mean = float(
                    bm.mean(bm.abs(dv[:] - dv_outer_start)))
                self.last_change_outer_mean = change_outer_mean
                self.last_change_outer_max = float(
                    bm.max(bm.abs(dv[:] - dv_outer_start)))

                inner_diagnostics = {
                    'outer_index': al_iter + 1,
                    'beta': outer_beta,
                    'mu': outer_mu,
                    'initial_residual': float(inner_initial_residual),
                    'final_residual': float(inner_final_residual),
                    'target': float(inner_target),
                    'inner_steps': int(inner_steps),
                    'accepted': bool(inner_accepted),
                    'change_outer_mean': change_outer_mean,
                    'change_outer_max': self.last_change_outer_max,
                    'max_constraint': max_constraint,
                    'max_relative_violation': max_relative_violation,
                    'max_relative_violation_solid': max_relative_violation_solid,
                }
                self.outer_history.append(inner_diagnostics)
                self.last_inner_diagnostics = inner_diagnostics

                if not inner_accepted:
                    inner_iteration_failed = True
                    # 本外层未执行乘子更新，因此乘子变化精确为零。
                    self.last_multiplier_change = 0.0
                    self.termination_reason = (
                        'inner-iteration-limit: '
                        f'outer={al_iter + 1}, '
                        f'steps={inner_steps}/{opts.mma_iters_per_al}, '
                        f'residual={inner_final_residual:.3e}, '
                        f'target={inner_target:.3e}, '
                        f'initial={inner_initial_residual:.3e}, '
                        f'mu={outer_mu:.4e}, beta={outer_beta}'
                    )
                    self._log_info(
                        f'ALM Optimization stopped before outer update: '
                        f'{self.termination_reason}')
                    break

            # =====================================================================
            # ALM 乘子与惩罚参数更新 (外层更新)
            # =====================================================================
            # 罚因子条件放大所需的违反度: 与 C2 同量 (实体子集上的 max_e r_e), 由
            # 优化器推入。update_multipliers() 的零参签名保持不变, 覆写者不受影响。
            _set_violation = getattr(self._al_objective, 'set_current_violation', None)
            if _set_violation is not None:
                _set_violation(max_relative_violation_solid)

            lamb_prev = bm.copy(self._al_objective.lamb)
            self._al_objective.update_multipliers()
            # lambda/mu 改变了 Phi 及其缓存, 状态不受影响。
            objective_valid = False
            # 乘子相对变化 max|lambda^(n+1) - lambda^(n)| / max(1, max lambda^(n)):
            # 6.4 节的必报诊断量, 反映外层是否已停止推乘子; 不进 C0-C3。
            self.last_multiplier_change = float(
                bm.max(bm.abs(self._al_objective.lamb - lamb_prev))
                / max(1.0, float(bm.max(lamb_prev))))
            self._log_info(f"ALM Step {al_iter}: "
                f"lambda: norm={bm.linalg.norm(self._al_objective.lamb):.6f},  max={bm.max(self._al_objective.lamb):.6f}, min={bm.min(self._al_objective.lamb):.6f}, "
                f"mu={self._al_objective.mu:.4f}, "
                f"d_lambda_rel={self.last_multiplier_change:.6e}, "
                f"capped={int(getattr(self._al_objective, 'last_capped_count', 0))}")
            
            # =====================================================================
            # Beta 更新后的状态重置 (投影连续化)
            # =====================================================================
            beta_updated = False 
            if hasattr(self._filter, 'continuation_step'):
                change, beta_updated = self._filter.continuation_step(change)
            
            if beta_updated:
                # 重置相关的缩放因子 (如果有)
                if hasattr(self, '_obj_scale_factor'):
                    self._obj_scale_factor = None 
                # 重置 MMA 渐近线和历史步，防止非线性跳跃导致的震荡
                self._low, self._upp = None, None  
                xold1, xold2 = dv[:], dv[:]        
                
                # 基于新的 beta 重新过滤一次物理密度，确保物理场与当前 beta 严格一致
                rho_phys = self._filter.filter_design_variable(design_variable=dv, physical_density=rho_phys)
                # 物理密度已变, 下一内层步头部重解状态 (每次 beta 更新多一次求解)。
                state = None
                objective_valid = False
                
                self._log_info("Beta updated. Resetting MMA asymptotes and scaling for stability.")

            # =====================================================================
            # 全局收敛判定: C0 连续化终止 & C1 设计稳定 & C2 松弛可行 & C3 持续
            # 依据 dut-postdoc/papers/huzhang-topopt/
            #     stress-constrained-topopt-models-and-algorithms.md 第 6 节。
            # =====================================================================
            # 将内层计算出的最新最大应力赋值给外层判定变量
            max_stress_measure = max_vm_stress

            # C0: 全部连续化参数已到达终值且本步未发生更新。
            # beta 只判 not beta_updated 是不够的: 那只说明本步没更新, 不说明
            # 已经到顶, 会把连续化中途的停滞记为收敛。
            beta_now = getattr(self._filter, 'beta', None)
            beta_ceiling = getattr(self._filter, 'beta_max', None)
            if beta_now is None or beta_ceiling is None:
                # 读不到投影状态: 退到"本步未发生连续化更新"这一必要条件。
                # 无连续化的过滤器上 beta_updated 恒为 False, 该分量恒成立;
                # 有连续化但不暴露 beta 的过滤器上, 至少不会把刚更新过投影的
                # 那一步记成 C0 达标 (那一步的状态还没在新参数下重解过)。
                beta_done = not beta_updated
            else:
                beta_done = (not beta_updated) and (float(beta_now) >= float(beta_ceiling))
            penalty_done = (not opts.use_penalty_continuation
                            or current_penalty >= self._target_penalty)
            c0 = beta_done and penalty_done

            # beta 更新步上 continuation_step 强制返回 change = 1.0 (哨兵值),
            # 物理度量必须在该分支同样失格, 否则 history 会把那一步画成"已收敛"。
            # 'mean': 外层步首末设计 rho^{k+1,0} 与 rho^{k,0} 的平均绝对变化
            # (正文算法 2 第 5 步的 Delta_{rho,out}); 最大变化只报告不判据。
            change_outer_mean = float(bm.mean(bm.abs(dv[:] - dv_outer_start)))
            self.last_change_outer_mean = change_outer_mean
            self.last_change_outer_max = float(bm.max(bm.abs(dv[:] - dv_outer_start)))
            measure = getattr(opts, 'change_measure', 'design')
            change_c1 = (change_outer_mean if measure == 'mean'
                         else change_phys if measure == 'physical'
                         else change)
            if beta_updated:
                change_c1 = 1.0

            c1 = change_c1 < opts.change_tolerance
            c2 = max_relative_violation_solid <= opts.stress_tolerance
            c4 = True
            if bool(getattr(opts, 'kkt_acceptance_enabled', False)) and c0 and c1 and c2:
                # C0 成立意味着本步未更新 beta, state 即当前 rho_phys 的状态。
                self.last_kkt_diagnostics = self.kkt_diagnostics(
                    design_variable=dv,
                    density_distribution=rho_phys,
                    state=state,
                    passive_mask=passive_mask,
                )
                c4 = bool(self.last_kkt_diagnostics['accepted'])

            # -----------------------------------------------------------------
            # 末期移动限制衰减: 仅在连续化终止 (C0) 之后、且窗口内的运动被判为
            # 非相干 (极限环) 时才收缩。仍在单调重组的轨迹不收缩, 以免把它冻在
            # 一个不是驻点的点上而误报 criterion-met。
            # -----------------------------------------------------------------
            move_limit_decay = float(getattr(opts, 'move_limit_decay', 1.0))
            if move_limit_decay < 1.0:
                window_size = int(getattr(opts, 'move_limit_progress_window', 10))
                rho_window.append(bm.copy(rho_phys[:]))
                if len(rho_window) > window_size + 1:
                    rho_window.pop(0)

                if c0 and self._c0_first_iter is None:
                    self._c0_first_iter = al_iter

                if (c0
                        and self._c0_first_iter is not None
                        and (al_iter - self._c0_first_iter) >= window_size
                        and len(rho_window) >= window_size + 1):
                    share = coherent_travel_share(
                                rho_window,
                                cell_ratio=float(getattr(opts, 'move_limit_progress_cell', 0.7))
                            )
                    if share < float(getattr(opts, 'move_limit_progress_ratio', 0.3)):
                        move_limit_min = float(getattr(opts, 'move_limit_min', 0.005))
                        self._move_limit_now = max(move_limit_min,
                                                   self._move_limit_now * move_limit_decay)
                        self._move_limit_decays += 1
                        self._log_info(
                            f"移动限制衰减 -> {self._move_limit_now:.4e} "
                            f"(相干行程份额 {share:.3f}, 第 {self._move_limit_decays} 次)"
                        )

            if c0 and c1 and c2 and c4:
                hold_count += 1
            else:
                hold_count = 0

            # C3: 连续 hold_steps 个 ALM 外层步同时满足 C0-C2 才终止, 用于排除
            # 乘子更新与渐近线调整造成的振幅穿越阈值的偶然命中。
            if hold_count >= opts.hold_steps:
                self.converged = True
                criteria = (
                    'C0-C4'
                    if bool(getattr(opts, 'kkt_acceptance_enabled', False))
                    else 'C0-C2'
                )
                self.termination_reason = (
                    f'criterion-met: {criteria} held for {hold_count} consecutive ALM steps '
                    f'(change={change:.3e}, change_phys={change_phys:.3e}, '
                    f'change_outer_mean={change_outer_mean:.3e}, '
                    f'measure={getattr(opts, "change_measure", "design")}, '
                    f'max_rel_solid={max_relative_violation_solid:.3e} '
                    f'(C2 quantity, threshold={self._c2_threshold_label()}), '
                    f'max_rel_global={max_relative_violation:.3e}, '
                    f'beta={beta_now}, p={current_penalty:.1f})'
                )
                self._log_info(
                    f'ALM Optimization converged at global iteration {global_iter}: '
                    f'{self.termination_reason}')
                break

        if not self.converged and not inner_iteration_failed:
            self.termination_reason = (
                f'max-al-iterations-reached: hold={hold_count}/{opts.hold_steps}, '
                f'change={change:.3e}, change_phys={change_phys:.3e}, '
                f'change_outer_mean={change_outer_mean:.3e}, '
                f'measure={getattr(opts, "change_measure", "design")}, '
                f'max_rel_solid={max_relative_violation_solid:.3e} '
                f'(C2 quantity, threshold={self._c2_threshold_label()}), '
                f'max_rel_global={max_relative_violation:.3e}'
            )
            self._log_info(f'ALM Optimization NOT converged: {self.termination_reason}')

        # driver 读取的公有诊断量 (不伸手进 _ 前缀内部)
        self.effective_move_limit = self._effective_move_limit()
        self.move_limit_decays = int(self._move_limit_decays)
        self.final_design_variable = bm.copy(dv[:])
        self.passive_mask = passive_mask

        return rho_phys, self.history
    
    def _c2_threshold_label(self) -> str:
        """返回 C2 验收子集阈值的可读标签 ('global' 或数值)."""
        thr = getattr(self.options, 'acceptance_solid_threshold', None)
        return 'global' if thr is None else f'{float(thr):g}'

    def _solid_max_relative_violation(self,
                                      relative_violation: TensorLike,
                                      rho_phys: TensorLike,
                                      global_max: float) -> float:
        """C2 实际比较的最大相对违反度.

        Parameters
        ----------
        relative_violation : TensorLike
            逐单元相对违反度 r_e (豁免单元为负常数).
        rho_phys : TensorLike
            当前物理密度.
        global_max : float
            全域最大值, 阈值为 None 或实体子集为空时直接返回.

        Returns
        -------
        float
            阈值非 None 时为 max over {e : rho_phys_e >= 阈值} 的 r_e; 否则为 global_max.
        """
        thr = getattr(self.options, 'acceptance_solid_threshold', None)
        if thr is None:
            return float(global_max)
        rho = bm.reshape(rho_phys[:], (-1,))
        # 逐单元先取最大 (HuZhang 链路返回 (NC, NQ) 的逐积分点量), 再按密度筛选。
        rv_cell = bm.max(bm.reshape(relative_violation, (rho.shape[0], -1)), axis=1)
        mask = rho >= float(thr)
        if int(bm.sum(mask)) == 0:
            return float(global_max)
        return float(bm.max(rv_cell[mask]))

    def kkt_diagnostics(
        self,
        design_variable,
        density_distribution,
        state=None,
        passive_mask=None,
    ):
        """计算原设计盒约束上的一阶 KKT 诊断量.

        驻定残差使用原设计空间 ``[0, 1]^n`` 上的 projected gradient

        ``||z - clip(z - grad L, 0, 1)||_inf``.

        其中 ``grad L`` 由原 Lagrangian 的物理密度梯度经过当前过滤/投影链回传.
        该定义不含 MMA 移动限制，也不使用 AL 罚权 ``lambda + mu*h``，因此移动
        限制缩小不会把非驻点伪装成一阶最优点.

        每次调用需要一次伴随求解. 当前应力算例每个单元仅在形心施加一个约束，
        因而 ``N_c=N_e``；实现仍按约束数组的实际元素数归一化并显式报告两者.
        """
        if state is None:
            state = self._al_objective._analyzer.solve_state(rho_val=density_distribution)

        grad_rho = self._al_objective.lagrangian_jac(
            density=density_distribution,
            state=state,
        )
        grad_design = self._filter.filter_objective_sensitivities(
            design_variable=design_variable,
            obj_grad_rho=grad_rho,
        )
        if passive_mask is not None:
            grad_design = bm.copy(grad_design)
            grad_design[passive_mask] = 0.0

        projected = design_variable - bm.clip(design_variable - grad_design, 0.0, 1.0)
        stationarity = float(bm.max(bm.abs(projected)))

        g = self._al_objective._cache_g
        lamb = self._al_objective.lamb
        if tuple(g.shape) != tuple(lamb.shape):
            raise RuntimeError(
                f"KKT 诊断要求乘子与约束同形, 实际 lambda={lamb.shape}, g={g.shape}."
            )
        n_constraints = g.numel() if hasattr(g, 'numel') else g.size
        n_cells = int(self._al_objective._NC)
        complementarity_raw = float(bm.max(bm.abs(lamb * g)))
        dual_raw = float(bm.max(bm.maximum(-lamb, 0.0)))
        feasibility = float(bm.max(bm.maximum(g, 0.0)))
        complementarity = complementarity_raw / n_constraints
        dual = dual_raw / n_constraints

        opts = self.options
        acceptance_enabled = bool(getattr(opts, 'kkt_acceptance_enabled', False))
        accepted = None
        if acceptance_enabled:
            accepted = bool(
                feasibility <= opts.stress_tolerance
                and stationarity <= opts.kkt_stationarity_tolerance
                and complementarity <= opts.kkt_complementarity_tolerance
                and dual <= opts.kkt_dual_tolerance
            )

        return {
            'enabled': bool(getattr(opts, 'kkt_diagnostics_enabled', False)),
            'acceptance_enabled': acceptance_enabled,
            'accepted': accepted,
            'n_constraints': int(n_constraints),
            'n_cells': n_cells,
            'one_constraint_per_cell': bool(n_constraints == n_cells),
            'feasibility': feasibility,
            'stationarity': stationarity,
            'complementarity_raw': complementarity_raw,
            'complementarity_normalized': complementarity,
            'dual_feasibility_raw': dual_raw,
            'dual_feasibility_normalized': dual,
            'tolerances': {
                'feasibility': float(opts.stress_tolerance),
                'stationarity': (
                    float(opts.kkt_stationarity_tolerance) if acceptance_enabled else None
                ),
                'complementarity_normalized': (
                    float(opts.kkt_complementarity_tolerance) if acceptance_enabled else None
                ),
                'dual_feasibility_normalized': (
                    float(opts.kkt_dual_tolerance) if acceptance_enabled else None
                ),
            },
        }


    def _solve_unconstrained_subproblem(self, 
                                        dfdz: TensorLike, 
                                        z: TensorLike, 
                                        zold1: TensorLike, 
                                        zold2: TensorLike,
                                    ) -> TensorLike:
        """完全复刻 MATLAB PolyStress 中的 MMA_unconst 解析求解
        
        与 MATLAB 的对应关系:
          z       <-> z(Eid)         活跃单元的设计变量
          dfdz    <-> dJdz(Eid)      活跃单元的灵敏度
          _epoch  <-> Iter           AL 步计数器 (控制渐近线初始化)
          _low_active, _upp_active <-> L(Eid), U(Eid)
        
        关键区别于父类 _solve_subproblem:
          1. 解析闭式解, 无需 KKT 迭代求解
          2. 无显式约束 (m=0), 对应 PolyStress 的无约束 AL 子问题
          3. 渐近线仅存储活跃单元部分
        """
        opts = self.options  
        dfdz = dfdz.reshape(-1)
        z = z.reshape(-1)
        zold1 = zold1.reshape(-1)
        zold2 = zold2.reshape(-1)

        zMin = 0.0
        zMax = 1.0
        move = self._effective_move_limit() * (zMax - zMin)
        Osc = opts.osc
        AsymInit = opts.asymp_init

        #TODO 修改
        xmin = bm.maximum(z - move, bm.full_like(z, zMin))
        xmax = bm.minimum(z + move, bm.full_like(z, zMax))
        # xmin = bm.maximum(zMin, z - move)
        # xmax = bm.minimum(zMax, z + move)

        # 动态截断 AsymInc / AsymDecr
        self._asym_inc_dynamic = min(1 + Osc, self._asym_inc_dynamic)
        self._asym_decr_dynamic = max(1 - 2 * Osc, self._asym_decr_dynamic)
        AsymInc = self._asym_inc_dynamic
        AsymDecr = self._asym_decr_dynamic

        # 1. 更新渐近线 L 和 U
        # 前 2 个 AL 步使用 AsymInit
        if self._epoch <= 2 or self._low is None or self._upp is None:
            L = z - AsymInit * (xmax - xmin)
            U = z + AsymInit * (xmax - xmin)
        else:
            low_prev = self._low.reshape(-1)
            upp_prev = self._upp.reshape(-1)
            
            sgn = (z - zold1) * (zold1 - zold2)
            s = bm.ones_like(z)
            s = bm.where(sgn > 0, AsymInc, s)
            s = bm.where(sgn < 0, AsymDecr, s)
            
            L = z - s * (zold1 - low_prev)
            U = z + s * (upp_prev - zold1)

        # 渐近线距离界 [asymptote_min_distance, 10] x (zMax - zMin)。
        # PolyStress 的 MMA_unconst 没有下限: 两值振荡的变量按 AsymDecr 逐步收缩
        # U - L, 步幅几何衰减直至停在折点上, 这是 MMA 对 AL 罚项折点 (曲率跳 mu)
        # 的唯一逐变量阻尼。2026-09-11 曾因 Hu-Zhang k=2 第 152 外层步 U - L 收缩到
        # 原式分母 p - q 相消归零而设下限 0.01, 但 0.01 高于末期移动限 (0.005),
        # 使该阻尼失效, 灰度带单元被钉在移动限幅度上翻转 (2026-09-15 LFEM k=1 极限环)。
        # 相消问题已由 _stable_mma_candidate 解决, 下限退回 1e-4 只防 U - L 下溢。
        span = zMax - zMin
        L, U = _bound_mma_asymptotes(
            z,
            L,
            U,
            span=span,
            min_distance=opts.asymptote_min_distance,
        )

        self._low = bm.copy(L)
        self._upp = bm.copy(U)

        # 2. 计算有效边界 alpha 和 beta
        alpha = 0.9 * L + 0.1 * z
        beta = 0.9 * U + 0.1 * z
        alpha = bm.maximum(xmin, alpha)
        beta = bm.minimum(xmax, beta)

        # 3. 求解无约束子问题 (解析解)
        feps = 1e-6
        #TODO 修改
        p = (U - z)**2 * (bm.maximum(dfdz, bm.zeros_like(dfdz)) 
                            + 0.001 * bm.abs(dfdz) 
                            + feps / (U - L))
        q = (z - L)**2 * (-bm.minimum(dfdz, bm.zeros_like(dfdz)) 
                            + 0.001 * bm.abs(dfdz) 
                            + feps / (U - L))
        # p = (U - z)**2 * (bm.maximum(dfdz, 0.0) 
        #                   + 0.001 * bm.abs(dfdz) 
        #                   + feps / (U - L))
        # q = (z - L)**2 * (-bm.minimum(dfdz, 0.0) 
        #                   + 0.001 * bm.abs(dfdz) 
        #                   + feps / (U - L))

        zCnd = _stable_mma_candidate(L, U, p, q)
        zNew = bm.maximum(alpha, bm.minimum(beta, zCnd))

        return zNew
