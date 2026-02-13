import warnings

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

class ALMMMAOptions(MMAOptions):
    """专门针对 ALM-MMA 双层优化算法的配置选项"""
    def __init__(self):
        super().__init__()
        
        # =========================================================================
        # ALM 专属外层控制参数 (对应 PolyStress.m 设置)
        # =========================================================================
        
        # ALM 外层最大迭代步数
        self.max_al_iterations = 200     
        
        # 每个 ALM 外层步中，MMA 内层优化的最大次数 (对应 opt.MMA_Iter)
        self.mma_iters_per_al = 10       
        
        # 应力约束的容差界限 (对应 opt.TolS)
        self.stress_tolerance = 0.01     
        
        # 惩罚因子 mu 的放大倍数 (对应 opt.alpha)
        self.alpha = 1.1                 
        
        # 收敛阈值 (对应 opt.Tol)
        self.change_tolerance = 0.01

        self.osc = 0.2           # 振荡控制参数


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
        # ==================== 新增的选项拦截与解析逻辑 ====================
        if options is None:
            parsed_options = ALMMMAOptions()
        elif isinstance(options, dict):
            parsed_options = ALMMMAOptions()
            for key, value in options.items():
                if hasattr(parsed_options, key) and not key.startswith('_'):
                    setattr(parsed_options, key, value)
                else:
                    warnings.warn(f"Ignored unknown or private parameter in options: '{key}'")
        elif isinstance(options, ALMMMAOptions):
            parsed_options = options
        else:
            raise TypeError("The 'options' parameter must be a dict or an ALMMMAOptions instance.")
        # ================================================================      
            
        # 调用父类初始化，传入空约束列表 []，使得 MMA 子问题退化为无约束问题 (m=0)
        super().__init__(objective=al_objective, 
                        constraint=[], 
                        filter=filter, 
                        options=parsed_options, 
                        enable_logging=enable_logging, 
                        logger_name=logger_name)
        
        self._al_objective = al_objective

        # 渐近线增减系数作为实例变量, 支持动态截断
        self._asym_inc = parsed_options.asymp_incr
        self._asym_decr = parsed_options.asymp_decr

        # 活跃单元局部渐近线 (仅存储 Eid 对应的子集)
        self._low_active = None
        self._upp_active = None

        self._analyzer = al_objective._analyzer

    def optimize(self,
                design_variable: Union[Function, TensorLike], 
                density_distribution: Union[Function, TensorLike], 
                enable_timing: bool = False,
                **kwargs
            ) -> Tuple[Union[Function, TensorLike], OptimizationHistory]:
        
        analyzer = self._al_objective._analyzer
        
        # --- 问题规模初始化 (constraint=[] -> m=0) ---
        m = 0 
        n = design_variable.shape[0]
        self.options._initialize_problem_params(m, n)

        # --- 变量初始化 ---
        if isinstance(design_variable, Function):
            dv = design_variable.space.function(bm.copy(design_variable[:]))
        else:
            dv = bm.copy(design_variable[:])
        
        from soptx.interpolation.interpolation_scheme import DensityDistribution
        if isinstance(density_distribution, Function):
            rho = density_distribution.space.function(bm.copy(density_distribution[:]))
        elif isinstance(density_distribution, DensityDistribution):
            rho = density_distribution
        else:
            rho = bm.copy(density_distribution[:])

        # 初始物理密度
        rho_phys = self._filter.get_initial_density(density=rho)

        # --- 被动单元处理 ---
        passive_mask = self._passive_mask  # 继承自父类
        if passive_mask is not None:
            dv[passive_mask] = 1.0
            all_ids = bm.arange(n)
            active_ids = all_ids[~passive_mask]
        else:
            active_ids = None  # None 表示全部单元均为活跃

        xold1 = bm.copy(dv)
        xold2 = bm.copy(dv)
        
        self.history = OptimizationHistory()
        global_iter = 0

        # 收敛参数
        Tol = self.options.change_tolerance 
        TolS = self.options.stress_tolerance
        change = 2 * Tol
        max_stress_measure = 10.0

        # =========================================================================
        # 1. 外层循环 (ALM 步) - 控制 lambda 和 mu
        # =========================================================================
        for al_iter in range(self.options.max_al_iterations):
            # _epoch 使用 AL 步计数 (非全局迭代计数)
            # 控制渐近线初始化: 前 2 个 AL 步使用 AsymInit
            self._epoch = al_iter + 1

            change = 2 * self.options.change_tolerance
            max_stress_measure = 2.0 
            
            # =====================================================================
            # 2. 内层循环 (MMA 步) - 对当前的 lambda, mu 进行极小化
            # =====================================================================
            for mma_iter in range(self.options.mma_iters_per_al):
                start_time = time()
                
                volfrac = self._al_objective._volume_objective.fun(density=rho_phys)

                # --- 计算增广拉格朗日目标函数及灵敏度 ---
                # a. 状态求解 (FEA)
                state = analyzer.solve_state(rho_val=rho_phys)
                
                # b. 评估增广拉格朗日目标函数及其物理敏度
                J_val = self._al_objective.fun(density=rho_phys, state=state)
                dJ_drho = self._al_objective.jac(density=rho_phys, state=state)
                
                # c. 通过过滤器应用链式法则 (获取对设计变量 dv 的敏度)
                dJ_dv = self._filter.filter_objective_sensitivities(design_variable=dv, obj_grad_rho=dJ_drho)
                
                # 被动单元灵敏度置零
                if passive_mask is not None:
                    dJ_dv[passive_mask] = 0.0

                # --- 求解无约束 MMA 子问题 ---
                if active_ids is not None:
                    dv_new = bm.copy(dv)  # 被动单元保持不变
                    z_active_new = self._solve_unconstrained_subproblem(
                        dfdz=dJ_dv[active_ids],
                        z=dv[active_ids],
                        zold1=xold1[active_ids],
                        zold2=xold2[active_ids],
                    )
                    dv_new[active_ids] = z_active_new
                else:
                    dv_new = self._solve_unconstrained_subproblem(
                                                        dfdz=dJ_dv,
                                                        z=dv,
                                                        zold1=xold1,
                                                        zold2=xold2,
                                                    )
                
                # 密度过滤与更新
                rho_phys = self._filter.filter_design_variable(design_variable=dv_new, physical_density=rho_phys)
                
                # 更新历史设计变量
                xold2 = xold1
                xold1 = dv[:]

                # 计算收敛性
                change = float(bm.mean(bm.abs(dv_new[active_ids] - dv[active_ids])))

                # 更新设计变量
                dv = dv_new
                
                # --- 计算材料刚度插值系数 (基于新的物理密度)---
                E_rho = self._analyzer.interpolation_scheme.interpolate_material(
                                                            material=self._analyzer.material,
                                                            rho_val=rho_phys,
                                                            integration_order=self._analyzer.integration_order,
                                                        )
                E0 = self._analyzer.material.youngs_modulus
                E = E_rho / E0 # (NC, )

                # --- 计算归一化的 von Mises 应力 ---
                vm = state['von_mises']
                slim = self._al_objective._stress_constraint._stress_limit
                SM = E[..., None] * vm / slim
                
                # 更新当前最大应力测度，用于收敛判定
                max_stress_measure = float(bm.max(SM))
                
                # [修正 2] 完善日志记录和历史保存
                iteration_time = time() - start_time

                # 按照 PolyStress.m 的格式自定义输出日志
                dJ_norm = float(bm.linalg.norm(dJ_dv))
                self._log_info(
                    f"It:{al_iter+1:3d}_{mma_iter+1:1d} "
                    f"Obj: {volfrac:.3f} "
                    f"Max_VM: {max_stress_measure:.3f} "
                    f"|dJ|: {dJ_norm:.3f} "
                    f"Ch/Tol: {change/Tol:.3f}"
                )
                
                # 调用确切的 log_iteration 接口保存历史数据
                self.history.log_iteration(
                    iter_idx=global_iter,
                    obj_val=float(J_val),                       # 记录 AL 总目标函数值
                    volfrac=volfrac,                            # 记录体积分数约束/目标
                    change=float(change),                       # 设计变量最大变化量
                    penalty_factor=float(self._al_objective.mu),# 当前的 AL 罚因子
                    time_cost=iteration_time,
                    physical_density=rho_phys,                  # 保存密度场
                    von_mises_stress=SM,                        # 保存 von Mises 应力场
                    verbose=False                               # 关闭默认打印，避免重复
                )
                global_iter += 1

                # 内层收敛判定: 变化率达标 且 最大应力满足 (1 + TolS)
                if change <= self.options.change_tolerance and max_stress_measure <= 1.0 + self.options.stress_tolerance:
                    break # 跳出内层循环，进入 ALM 更新
            
            # =====================================================================
            # 3. ALM 乘子与惩罚参数更新 (外层更新)
            # =====================================================================
            h_val = self._al_objective._cache_h
            
            # lambda = lambda + mu * h
            self._al_objective.lamb = self._al_objective.lamb + self._al_objective.mu * h_val
            
            # mu = min(alpha * mu, mu_max)
            self._al_objective.mu = min(self.options.alpha * self._al_objective.mu, self._al_objective.mu_max)
            
            self._log_info(f"ALM Step {al_iter}: max(SM)={max_stress_measure:.3f}, mu={self._al_objective.mu:.1f}")

            # =====================================================================
            # 4. 全局收敛判定
            # =====================================================================
            if change <= self.options.change_tolerance and max_stress_measure <= 1.0 + self.options.stress_tolerance:
                self._log_info(f"ALM Optimization converged perfectly at global iteration {global_iter}.")
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
        dfdz = dfdz.reshape(-1)
        z = z.reshape(-1)
        zold1 = zold1.reshape(-1)
        zold2 = zold2.reshape(-1)

        zMin = 0.0
        zMax = 1.0
        move = self.options.move_limit * (zMax - zMin)
        Osc = self.options.osc
        AsymInit = self.options.asymp_init

        # [修正5] 渐近线增减系数截断 (对应 MATLAB MMA_unconst 中的动态更新)
        #   MATLAB: AsymInc = min(1+Osc, AsymInc);
        #           AsymDecr = max(1-2*Osc, AsymDecr);
        self._asym_inc = min(1.0 + Osc, self._asym_inc)
        self._asym_decr = max(1.0 - 2.0 * Osc, self._asym_decr)
        AsymInc = self._asym_inc
        AsymDecr = self._asym_decr

        xmin = bm.maximum(zMin, z - move)
        xmax = bm.minimum(zMax, z + move)
        xmami = xmax - xmin

        # ---- 1. 更新渐近线 L 和 U ----
        #   对应 MATLAB: if Iter<=2 ... else ... end
        if self._epoch <= 2 or self._low_active is None or self._upp_active is None:
            L = z - AsymInit * xmami
            U = z + AsymInit * xmami
        else:
            L_prev = self._low_active.reshape(-1)
            U_prev = self._upp_active.reshape(-1)

            sgn = (z - zold1) * (zold1 - zold2)
            s = bm.ones_like(z)
            s = bm.where(sgn > 0, AsymInc, s)
            s = bm.where(sgn < 0, AsymDecr, s)
            
            L = z - s * (zold1 - L_prev)
            U = z + s * (U_prev - zold1)

        # 保存活跃单元的渐近线
        self._low_active = bm.copy(L)
        self._upp_active = bm.copy(U)

        # ---- 2. 计算有效边界 alpha 和 beta ----
        #   对应 MATLAB: alpha = 0.9*L + 0.1*z; beta = 0.9*U + 0.1*z;
        alpha = 0.9 * L + 0.1 * z
        beta = 0.9 * U + 0.1 * z
        alpha = bm.maximum(xmin, alpha)
        beta = bm.minimum(xmax, beta)

        # ---- 3. 解析求解无约束子问题 ----
        #   对应 MATLAB: p, q, zCnd, zNew 的计算
        feps = 1e-6
        dfdz_pos = bm.maximum(dfdz, 0.0)
        dfdz_neg = -bm.minimum(dfdz, 0.0)
        dfdz_abs = bm.abs(dfdz)

        p = (U - z)**2 * (dfdz_pos + 0.001 * dfdz_abs + feps / (U - L))
        q = (z - L)**2 * (dfdz_neg + 0.001 * dfdz_abs + feps / (U - L))

        zCnd = (L * p - U * q + (U - L) * bm.sqrt(p * q)) / (p - q)

        zNew = bm.maximum(alpha, bm.minimum(beta, zCnd))

        return zNew