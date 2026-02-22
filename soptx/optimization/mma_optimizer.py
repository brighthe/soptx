import warnings
from time import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Any, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.compliant_mechanism_objective import CompliantMechanismObjective
from soptx.optimization.compliance_objective import ComplianceObjective
from soptx.optimization.volume_constraint import VolumeConstraint
from soptx.optimization.stress_constraint import StressConstraint
from soptx.optimization.tools import OptimizationHistory
from soptx.optimization.utils import solve_mma_subproblem
from soptx.optimization.utils import compute_volume
from soptx.regularization.filter import Filter
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

@dataclass
class MMAOptions:
    """MMA 算法的配置选项"""
    # =========================================================================
    # 1. 用户常用控制参数
    # =========================================================================
    # 最大迭代次数 ([柔顺度]: ~200, [应力/ALM]: ~150)
    max_iterations: int = 200
    
    # 收敛阈值 ([柔顺度]: 1e-2, [应力]: 2e-3)
    change_tolerance: float = 1e-2
    
    # 是否使用惩罚因子连续化 (SIMP)
    use_penalty_continuation: bool = True

    # =========================================================================
    # 2. 几何/渐近线控制参数 (Asymptotes)
    # =========================================================================
    # 移动限制 (Move limit)
    move_limit: float = 0.2
    
    # 初始渐近线距离因子 (Initial asymptote)
    asymp_init: float = 0.5
    
    # 渐近线扩张/收缩系数
    asymp_incr: float = 1.2
    asymp_decr: float = 0.7

    # =========================================================================
    # 3. 子问题数值参数 (Subproblem)
    # =========================================================================
    # 边界计算因子
    albefa: float = 0.1
    # 近似精度参数
    raa0: float = 1e-5
    # 最小数值容差
    epsilon_min: float = 1e-7

    # =========================================================================
    # 4. 辅助变量参数 (Auxiliary Variables)
    # =========================================================================
    # a0 常数 (目标函数项权重)
    a0: float = 1.0

    # -------------------------------------------------------------------------
    # 内部状态 (由 initialize_problem_params 填充)
    # -------------------------------------------------------------------------
    m: int = field(default=0, init=False)  # 约束数量
    n: int = field(default=0, init=False)  # 变量数量
    
    # 以下向量在初始化问题规模后分配
    xmin: Optional[TensorLike] = field(default=None, init=False)
    xmax: Optional[TensorLike] = field(default=None, init=False)
    a: Optional[TensorLike] = field(default=None, init=False)
    c: Optional[TensorLike] = field(default=None, init=False)
    d: Optional[TensorLike] = field(default=None, init=False)

    def initialize_problem_params(self, m: int, n: int) -> None:
        """
        初始化与问题规模相关的向量 (m: 约束数, n: 变量数)
        """
        self.m = m
        self.n = n
        
        # 初始化上下界与辅助向量
        self.xmin = bm.zeros((n, 1), dtype=bm.float64)
        self.xmax = bm.ones((n, 1), dtype=bm.float64)
        
        self.a = bm.zeros((m, 1), dtype=bm.float64)
        self.d = bm.zeros((m, 1), dtype=bm.float64)
        self.c = 1e4 * bm.ones((m, 1), dtype=bm.float64) 


class MMAOptimizer(BaseLogged):
    def __init__(self,
                objective: Union[ComplianceObjective, CompliantMechanismObjective],
                constraint: VolumeConstraint,
                filter: Filter,
                options: Union[MMAOptions, Dict[str, Any], None] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """Method of Moving Asymptotes (MMA) 优化器
    
        用于求解拓扑优化问题的 MMA 方法实现. 该方法通过动态调整渐近线位置
        来控制优化过程, 具有良好的收敛性能
        
        Parameters
        ----------
        objective: 目标函数对象
        constraint: 约束条件对象
        filter: 过滤器对象
        options: 优化器配置选项
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._objective = objective
        self._constraint = constraint
        self._filter = filter

        # 选项初始化与适配
        if isinstance(options, MMAOptions):
            self.options = options
        elif isinstance(options, dict):
            # 如果传入字典，先创建默认实例，再更新
            self.options = MMAOptions()
            # 获取 dataclass 的有效字段集合
            valid_configs = {f for f in self.options.__dict__ if not f.startswith('_')}
            
            for key, value in options.items():
                if key in valid_configs:  
                    setattr(self.options, key, value)
                else:
                    self._log_warning(f"Ignored unknown or internal parameter in options: '{key}'")
        else:
            self.options = MMAOptions()

        # 初始化问题规模参数为 None
        self._n = None
        self._m = None

        # MMA 内部状态初始化
        self._epoch = 0
        self._low: Optional[TensorLike] = None
        self._upp: Optional[TensorLike] = None

    def _update_penalty(self, iter_idx: int) -> None:
        """连续化技术对幂指数惩罚因子进行更新
        
        两种方式:
        方式1: 初始 1, 每 30 次迭代增加 0.5, 最终 3
        方式2: 初始 1, 每步增加 0.04, 第 50 步增至 3
        """
        if not self.options.use_penalty_continuation:
            return
        
        interpolation_scheme = self._objective._analyzer._interpolation_scheme
        
        # 方式 1: 每 30 次迭代增加 0.5
        penalty_update = iter_idx // 30
        current_penalty = min(1.0 + penalty_update * 0.5, 3.0)
        
        # 方式 2: 每步增加 0.04，第 50 步达到 3
        # current_penalty = min(1.0 + iter_idx * 0.04, 3.0)
        
        # 更新插值方案中的惩罚因子
        if current_penalty != interpolation_scheme.penalty_factor:
            interpolation_scheme.penalty_factor = current_penalty

    def optimize(self,
                design_variable: Union[Function, TensorLike], 
                density_distribution: Union[Function, TensorLike], 
                is_store_stress: bool = True,
                enable_timing: bool = False,
                **kwargs
            ) -> Tuple[Union[Function, TensorLike], OptimizationHistory]:
        """运行 MMA 优化算法
        
        Parameters
        ----------
        design_variable : 设计变量
        density_distribution : 密度分布
        **kwargs : 其他参数
        """
        # ==================== 获取分析器引用 ====================
        analyzer = self._objective._analyzer
        interpolation_scheme = analyzer.interpolation_scheme

        # ==================== 问题规模参数初始化 ====================
        m = 1
        n = design_variable.shape[0]
        self.options.initialize_problem_params(m, n)

        # ==================== 优化参数获取 ====================
        max_iters = self.options.max_iterations
        change_tol = self.options.change_tolerance

        # ==================== 变量初始化 ====================
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

        # ==================== MMA 历史变量初始化 ====================
        xold1 = bm.copy(dv[:])
        xold2 = bm.copy(xold1[:])

        # ==================== 优化状态初始化 ====================
        self.history = OptimizationHistory()
        # 初始化目标函数缩放因子 (投影时使用)
        self._obj_scale_factor = None

        # 优化主循环
        for iter_idx in range(max_iters):
            t = None
            if enable_timing:
                t = timer(f"拓扑优化单次迭代")
                next(t)

            start_time = time()

            # ==================== 惩罚因子更新 ====================
            self._update_penalty(iter_idx=iter_idx)
            current_penalty = interpolation_scheme.penalty_factor
            
            # 更新迭代计数
            self._epoch = iter_idx + 1

            # 基于物理密度求解状态变量
            if hasattr(analyzer, 'solve_state'):
                state = analyzer.solve_state(rho_val=rho_phys)
            elif isinstance(self._objective, CompliantMechanismObjective):
                state = analyzer.solve_state(rho_val=rho_phys, adjoint=True)
            if enable_timing:
                t.send('位移场求解')

            # ==================== 目标函数计算 ====================
            obj_val_raw = self._objective.fun(density=rho_phys, state=state)

            # 动态初始化缩放因子 (投影时使用)
            if self._obj_scale_factor is None:   
                obj0 = float(obj_val_raw.item()) 
                denom = max(abs(obj0), 1e-10)
                # 使用投影滤波 -> 缩放至 10.0
                if self._filter._filter_type == 'projection':
                    target_initial_val = 10.0
                    self._obj_scale_factor = min(1e6, target_initial_val / denom)
                # 其他情况 -> 默认 1.0 不缩放
                else:
                    self._obj_scale_factor = 1.0

            # 目标函数应用缩放因子
            obj_val = obj_val_raw * self._obj_scale_factor

            if enable_timing:
                t.send('目标函数计算')

            # ==================== 目标函数灵敏度 ====================
            # 1. 计算目标函数相对于物理密度的灵敏度
            obj_grad_rho_raw = self._objective.jac(density=rho_phys, state=state)
            # 灵敏度应用缩放因子
            obj_grad_rho = obj_grad_rho_raw * self._obj_scale_factor 
            # 2. 计算目标函数相对于设计变量的灵敏度
            obj_grad_dv = self._filter.filter_objective_sensitivities(design_variable=dv, obj_grad_rho=obj_grad_rho)
                
            if enable_timing:
                t.send('目标函数灵敏度分析')

            # ==================== 约束函数计算 ====================
            # 使用物理密度计算约束函数
            con_val = self._constraint.fun(rho_phys)
            if enable_timing:
                t.send('约束函数计算')

            # 计算约束函数相对于物理密度的灵敏度
            con_grad_rho = self._constraint.jac(rho_phys)
            if enable_timing:
                t.send('约束函数灵敏度分析 1')

            # 计算约束函数相对于设计变量的灵敏度
            con_grad_dv = self._filter.filter_constraint_sensitivities(design_variable=dv, con_grad_rho=con_grad_rho)
            if enable_timing:
                t.send('约束函数灵敏度分析 2')
            
            # ==================== MMA 子问题求解 ====================
            # 求解子问题
            dv_new = self._solve_subproblem(
                                        xval=dv[:, None],
                                        fval=con_val,
                                        df0dx=obj_grad_dv[:, None],
                                        dfdx=con_grad_dv,
                                        xold1=xold1[:, None],
                                        xold2=xold2[:, None]
                                    )

            if enable_timing:
                t.send('MMA 优化')

            # 过滤后得到的物理密度
            rho_phys = self._filter.filter_design_variable(design_variable=dv_new, physical_density=rho_phys)

            if enable_timing:
                t.send('密度过滤')
            
            # 更新历史设计变量
            xold2 = xold1
            xold1 = dv[:]

            # 计算收敛性
            change = bm.max(bm.abs(dv_new - dv))
            # print(f"设计变量最大变化量: {change}")
            
            # 更新设计变量
            dv = dv_new
                
            # 当前体积分数
            mesh = self._objective._analyzer._mesh            
            cell_measure = mesh.entity_measure('cell')
            current_volume = bm.einsum('c, c -> ', cell_measure, rho_phys[:])
            total_volume = bm.sum(cell_measure)
            volfrac = current_volume / total_volume

            # 记录当前迭代信息
            iteration_time = time() - start_time

            von_mises_stress = None
            max_vm_stress = None
            if is_store_stress:
                stress_solid = analyzer.compute_stress_state(state=state)['stress_solid']
                raw_von_mises = analyzer.material.calculate_von_mises_stress(stress_solid) / 100.0
                von_mises_stress = raw_von_mises * rho_phys[:, None]  
                max_vm_stress = float(bm.max(von_mises_stress))

                if enable_timing:
                    t.send('von Mises 应力计算')

            self._log_info(
                    f"Iteration: {iter_idx + 1}, "
                    f"Objective: {obj_val:.4f}, "
                    f"Volfrac: {volfrac:.4f}, "
                    f"Change: {change:.4f}, "
                    f"Time: {iteration_time:.3f} sec"
                )
            
            scalars = {
                        'compliance': obj_val_raw,
                        'volfrac': volfrac,
                    }
            fields = {}

            if is_store_stress:
                scalars['max_von_mises_stress'] = max_vm_stress
                fields['von_mises_stress'] = von_mises_stress

            self.history.log_iteration(
                            iter_idx=self._epoch,
                            change=change,
                            time_cost=iteration_time,
                            physical_density=rho_phys,
                            scalars=scalars,
                            fields=fields,
                        )
            
            # ==================== Beta 更新后的状态重置 (投影时使用) ====================
            beta_updated = False 
            if current_penalty >= 3.0:
                change, beta_updated = self._filter.continuation_step(change)
            
            if beta_updated:
                self._obj_scale_factor = None 
                self._low, self._upp = None, None  # 重置渐近线
                xold1, xold2 = dv[:], dv[:]        # 重置历史步
                rho_phys = self._filter.filter_design_variable(design_variable=dv, physical_density=rho_phys)
                self._log_info(f"Beta updated. Resetting MMA asymptotes and scaling for stability.")
                continue
            
            if enable_timing:
                t.send('后处理')
                t.send(None)

            # 收敛检查
            if change <= change_tol and current_penalty >= 3.0:
                msg = (f"Converged after {self._epoch} iterations "
                       f"(design variable change <= {change_tol}).")
                self._log_info(msg)
                break

        else:
            self._log_info(
                "Maximum number of iterations reached before satisfying "
                "design-change tolerance (quasi-convergence)."
            )
                
        # 打印时间统计信息
        self.history.print_time_statistics()
        
        return rho_phys, self.history
        
    def _update_asymptotes(self, 
                          xval: TensorLike, 
                          xmin: TensorLike,
                          xmax: TensorLike,
                          xold1: TensorLike,
                          xold2: TensorLike
                        ) -> Tuple[TensorLike, TensorLike]:
        """更新渐近线位置
        
        Parameters
        ----------
        xval : TensorLike (n, 1)
            当前设计变量
        xmin : TensorLike (n, 1)
            设计变量下界
        xmax : TensorLike (n, 1)
            设计变量上界
        xold1 : TensorLike (n, 1)
            前一步设计变量
        xold2 : TensorLike (n, 1)
            前两步设计变量
            
        Returns
        -------
        Tuple[TensorLike, TensorLike]
            更新后的下渐近线和上渐近线
        """
        asyinit = self.options.asymp_init
        asyincr = self.options.asymp_incr
        asydecr = self.options.asymp_decr

        xmami = xmax - xmin
        
        if self._epoch <= 2 or self._low is None or self._upp is None:
            self._low = xval - asyinit * xmami
            self._upp = xval + asyinit * xmami
        else:
            factor = bm.ones((xval.shape[0], 1))
            xxx = (xval - xold1) * (xold1 - xold2)
            epsilon = 1e-12
            factor[xxx > epsilon] = asyincr
            factor[xxx < -epsilon] = asydecr
            
            self._low = xval - factor * (xold1 - self._low)
            self._upp = xval + factor * (self._upp - xold1)
            
            lowmin = xval - 10 * xmami
            lowmax = xval - 0.01 * xmami
            uppmin = xval + 0.01 * xmami
            uppmax = xval + 10 * xmami
            
            self._low = bm.maximum(self._low, lowmin)
            self._low = bm.minimum(self._low, lowmax)
            self._upp = bm.minimum(self._upp, uppmax)
            self._upp = bm.maximum(self._upp, uppmin)

        return self._low, self._upp
        
    def _solve_subproblem(self, 
                        xval: TensorLike,
                        fval: TensorLike,
                        df0dx: TensorLike,
                        dfdx: TensorLike,
                        xold1: TensorLike,
                        xold2: TensorLike
                    ) -> TensorLike:
        """求解 MMA 子问题
        
        Parameters
        ----------
        xval :  (n, 1) - 当前设计变量
        fval :  (m, 1) - 标准化后的约束函数值
        df0dx:  (n, 1) - 目标函数相对于设计变量的梯度
        dfdx :  (m, n) - 约束函数相对于设计变量的梯度 (第 i 行对应第 i 个约束)
        xold1 : (n, 1) - 前一步设计变量
        xold2 : (n, 1) - 前两步设计变量

        Returns
        -------
        xmma :  (n, ) MMA 子问题的最优解 (新的设计变量)
        """
        xmin = self.options.xmin
        xmax = self.options.xmax
        m = self.options.m
        n = self.options.n

        a0 = self.options.a0
        a = self.options.a
        c = self.options.c
        d = self.options.d

        albefa = self.options.albefa
        raa0 = self.options.raa0
        epsimin = self.options.epsilon_min

        eeen = bm.ones((n, 1), dtype=bm.float64)
        eeem = bm.ones((m, 1), dtype=bm.float64)

        # --- 动态移动限制 (基于 Beta 连续化参数调整) ---
        move = self.options.move_limit
        beta_val = getattr(self._filter, 'beta', None)

        if beta_val is not None:
            move = move / (1.0 + 0.3 * bm.log(beta_val))

        # 更新渐近线
        low, upp = self._update_asymptotes(xval, xmin, xmax, xold1, xold2)
        
        # 计算变量边界 alfa, beta
        xxx1 = low + albefa * (xval - low)
        xxx2 = xval - move * (xmax - xmin)
        xxx = bm.maximum(xxx1, xxx2)
        alfa = bm.maximum(xmin, xxx)
        
        xxx1 = upp - albefa * (upp - xval)
        xxx2 = xval + move * (xmax - xmin)
        xxx = bm.minimum(xxx1, xxx2)
        beta = bm.minimum(xmax, xxx)

        # 计算 p0, q0 构建目标函数的近似
        xmami = xmax - xmin
        xmami_eps = raa0 * eeen
        xmami = bm.maximum(xmami, xmami_eps)
        xmami_inv = eeen / xmami
        
        ux1 = upp - xval
        xl1 = xval - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv = eeen / ux1
        xlinv = eeen / xl1
        
        p0 = bm.maximum(df0dx, bm.tensor(0, dtype=bm.float64))   
        q0 = bm.maximum(-df0dx, bm.tensor(0, dtype=bm.float64)) 
        pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2
        
        # 构建 P, Q 和 b 构建约束函数的近似
        P = bm.zeros((m, n), dtype=bm.float64)
        Q = bm.zeros((m, n), dtype=bm.float64)
        P = bm.maximum(dfdx, bm.tensor(0, dtype=bm.float64))
        Q = bm.maximum(-dfdx, bm.tensor(0, dtype=bm.float64))
        PQ = 0.001 * (P + Q) + raa0 * bm.dot(eeem, xmami_inv.T)
        P = P + PQ
        Q = Q + PQ
        
        # 使用 einsum 替代对角矩阵乘法
        P = bm.einsum('j, ij -> ij', ux2.flatten(), P)
        Q = bm.einsum('j, ij -> ij', xl2.flatten(), Q)
        b = bm.dot(P, uxinv) + bm.dot(Q, xlinv) - fval
        
        # 求解子问题
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = solve_mma_subproblem(
                                                                    m=m, n=n, 
                                                                    epsimin=epsimin, 
                                                                    low=low, upp=upp, 
                                                                    alfa=alfa, beta=beta,
                                                                    p0=p0, q0=q0, P=P, Q=Q,
                                                                    a0=a0, a=a, b=b, c=c, d=d
                                                                )
        
        if (bm.any(bm.isnan(xmma[:])) or bm.any(bm.isinf(xmma[:])) or 
            bm.any(xmma[:] < -1e-12) or bm.any(xmma[:] > 1 + 1e-12)):
            self._log_error(f"输出密度超出合理范围 [0, 1]: "
                            f"range=[{bm.min(xmma):.2e}, {bm.max(xmma):.2e}]")
        
        return xmma.reshape(-1)