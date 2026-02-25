from dataclasses import dataclass
from typing import Union, Tuple, Optional
from time import time
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.tools import OptimizationHistory
from soptx.regularization.filter import Filter
from soptx.utils import timer
from soptx.optimization.mma_optimizer import MMAOptions, MMAOptimizer
from soptx.optimization.augmented_lagrangian_objective import AugmentedLagrangianObjective

@dataclass
class ALMMMAOptions(MMAOptions):
    """专门针对 ALM-MMA 双层优化算法的配置选项"""
    # =========================================================================
    # 1. 覆盖父类默认值 (应力问题更保守)
    # =========================================================================
    change_tolerance: float = 0.002         # 设计变量收敛阈值 (Tol)
    move_limit: float = 0.15                # 更保守的移动限制
    asymp_init: float = 0.2                 # 更紧的初始渐近线距离
    use_penalty_continuation: bool = False  # ALM 框架下不使用 SIMP 连续化

    # =========================================================================
    # 2. ALM 外层循环控制
    # =========================================================================
    max_al_iterations: int = 150           # ALM 外层步数
    mma_iters_per_al: int = 5              # 每个 ALM 步中 MMA 内层迭代次数
    stress_tolerance: float = 0.003        # 应力约束容差 (TolS)

    # =========================================================================
    # 3. 渐近线振荡控制 (与父类 asymp_incr/asymp_decr 配合使用)
    # =========================================================================
    osc: float = 0.2                       # 振荡控制参数

    # =========================================================================
    # 4. 增广拉格朗日罚参数
    # =========================================================================
    mu_0: float = 10.0                     # 初始罚因子 μ^(0)
    mu_max: float = 10000.0                # 最大罚因子 μ_max
    alpha: float = 1.1                     # 罚因子更新参数 α > 1
    lambda_0_init_val: float = 0.0         # 初始拉格朗日乘子标量值 λ^(0)

    # =========================================================================
    # 5. 阈值投影连续化参数 (对应 opt.contB = [BFreq, B0, Binc, Bmax])
    # =========================================================================
    beta_freq: int = 5                     # 每隔多少 AL 步更新一次 β
    beta_init: float = 1.0                 # β 初始值
    beta_incr: float = 1.0                 # β 每次增量
    beta_max: float = 10.0                 # β 最大值

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
        global_iter = 0

        # =========================================================================
        # 外层循环 (ALM 步) - 控制 λ 和 μ
        # =========================================================================
        for al_iter in range(opts.max_al_iterations):
            # _epoch 使用 AL 步计数 (非全局迭代计数)
            # 控制渐近线初始化: 前 2 个 AL 步使用 AsymInit
            self._epoch = al_iter + 1

            change = 2 * opts.change_tolerance

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
                # a. 状态求解 (FEA)
                state = analyzer.solve_state(rho_val=rho_phys)
                if enable_timing:
                    t.send('求解')
                
                # b. 评估增广拉格朗日目标函数及其物理敏度
                J_val = self._al_objective.fun(density=rho_phys, state=state)
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

                # --- 求解无约束 MMA 子问题 ---
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
                
                # 密度过滤与更新
                rho_phys = self._filter.filter_design_variable(design_variable=dv_new, physical_density=rho_phys)
                if enable_timing:
                    t.send('密度过滤')
                
                # 更新历史设计变量
                xold2 = xold1
                xold1 = dv[:]

                # 计算收敛性
                change = float(bm.mean(bm.abs(dv_new - dv)))

                # 更新设计变量
                dv = dv_new
                
                # --- 计算材料刚度插值系数 (基于新的物理密度)---
                E_rho = analyzer.interpolation_scheme.interpolate_material(
                                                            material=analyzer.material,
                                                            rho_val=rho_phys,
                                                            integration_order=analyzer.integration_order,
                                                        )
                E0 = analyzer.material.youngs_modulus
                E = E_rho / E0 # (NC, )

                # --- 计算归一化的 von Mises 应力 ---
                vm = state['von_mises']
                slim = self._al_objective._stress_constraint._stress_limit
                SM = E[..., None] * vm / slim
                
                # 更新当前最大应力测度，用于收敛判定
                max_vm_stress = float(bm.max(SM))
                
                # 完善日志记录和历史保存
                iteration_time = time() - start_time

                # 按照 PolyStress.m 的格式自定义输出日志
                dJ_norm = float(bm.linalg.norm(dJ_dv))
                if enable_timing:
                    t.send('后处理')
                    t.send(None)

                self._log_info(
                        f"It:{al_iter+1:3d}_{mma_iter+1:1d} "
                        f"Obj: {volfrac:.6f} "
                        f"Max_VM: {max_vm_stress:.6f} "
                        f"|dJ|: {dJ_norm:.6f} "
                        f"Ch/Tol: {change/opts.change_tolerance:.6f} "
                        f"mu: {self._al_objective.mu:.4e} "
                        f"p: {current_penalty:.1f} "
                        f"Time: {iteration_time:.3f} sec "
                    )
                
                # 调用确切的 log_iteration 接口保存历史数据
                self.history.log_iteration(
                        iter_idx=global_iter,
                        change=change,                              # 设计变量最大变化量
                        time_cost=iteration_time,                   # 本次迭代耗时
                        physical_density=rho_phys,                  # 密度场
                        scalars={
                            'volfrac': volfrac,                     # 体积分数 (目标函数)
                            'max_von_mises': max_vm_stress,         # 归一化的最大 von Mises 应力场
                        },
                        fields={
                            'von_mises_stress': SM,                 # 归一化的 von Mises 应力场 (约束函数)
                        },
                )

                # 内层收敛判定: 变化率达标 且 最大应力满足 (1 + TolS)
                if change <= opts.change_tolerance and max_vm_stress <= 1.0 + opts.stress_tolerance:
                        break # 跳出内层循环，进入 ALM 更新      

            # =====================================================================
            # ALM 乘子与惩罚参数更新 (外层更新)
            # =====================================================================
            self._al_objective.update_multipliers()
            self._log_info(f"ALM Step {al_iter}: "
                f"lambda: norm={bm.linalg.norm(self._al_objective.lamb):.6f},  max={bm.max(self._al_objective.lamb):.6f}, min={bm.min(self._al_objective.lamb):.6f}, "
                f"mu={self._al_objective.mu:.4f}")
            
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
                
                self._log_info("Beta updated. Resetting MMA asymptotes and scaling for stability.")

            # =====================================================================
            # 全局收敛判定
            # =====================================================================
            # 将内层计算出的最新最大应力赋值给外层判定变量
            max_stress_measure = max_vm_stress

            if change <= opts.change_tolerance and max_stress_measure <= 1.0 + opts.stress_tolerance:
                if not opts.use_penalty_continuation or current_penalty >= self._target_penalty:
                    self._log_info(f"ALM Optimization converged perfectly at global iteration {global_iter} with p={current_penalty:.1f}.")
                    break
                
        return rho_phys, self.history
    

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
        move = opts.move_limit * (zMax - zMin)
        Osc = opts.osc
        AsymInit = opts.asymp_init

        xmin = bm.maximum(zMin, z - move)
        xmax = bm.minimum(zMax, z + move)

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

        self._low = bm.copy(L)
        self._upp = bm.copy(U)

        # 2. 计算有效边界 alpha 和 beta
        alpha = 0.9 * L + 0.1 * z
        beta = 0.9 * U + 0.1 * z
        alpha = bm.maximum(xmin, alpha)
        beta = bm.minimum(xmax, beta)

        # 3. 求解无约束子问题 (解析解)
        feps = 1e-6
        p = (U - z)**2 * (bm.maximum(dfdz, 0.0) 
                          + 0.001 * bm.abs(dfdz) 
                          + feps / (U - L))
        q = (z - L)**2 * (-bm.minimum(dfdz, 0.0) 
                          + 0.001 * bm.abs(dfdz) 
                          + feps / (U - L))

        zCnd = (L * p - U * q + (U - L) * bm.sqrt(p * q)) / (p - q)
        zNew = bm.maximum(alpha, bm.minimum(beta, zCnd))

        return zNew