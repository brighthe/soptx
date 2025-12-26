import warnings
from time import time
from typing import Optional, Tuple, Union

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.compliant_mechanism_objective import CompliantMechanismObjective
from soptx.optimization.compliance_objective import ComplianceObjective
from soptx.optimization.volume_constraint import VolumeConstraint
from soptx.optimization.tools import OptimizationHistory
from soptx.optimization.utils import solve_mma_subproblem
from soptx.regularization.filter import Filter
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer


class MMAOptions:
    """MMA 算法的配置选项"""

    def __init__(self):
        """初始化参数的默认值"""
        # MMA 算法的用户级参数
        self.max_iterations = 200             # 最大迭代次数
        self.change_tolerance = 1e-2          # 设计变量无穷范数阈值
        self.use_penalty_continuation = True  # 是否使用惩罚因子连续化技术

        # MMA 算法的高级参数
        self._m = 1
        self._n = None
        self._xmin = None
        self._xmax = None
        self._a0 = 1.0
        self._a = None
        self._c = None
        self._d = None
        
        # MMA 算法的固定参数
        self._asymp_init = 0.5      # 渐近线初始距离的因子
        self._asymp_incr = 1.2      # 渐近线矩阵减小的因子
        self._asymp_decr = 0.7      # 渐近线矩阵增加的因子
        self._move_limit = 0.2      # 移动限制
        self._albefa = 0.1          # 计算边界 alfa 和 beta 的因子
        self._raa0 = 1e-5           # 函数近似精度的参数
        self._epsilon_min = 1e-7    # 最小容差

    def set_advanced_options(self, **kwargs):
        """设置高级选项，仅供专业用户使用
        
        Parameters:
        -----------
        - **kwargs : 高级参数设置，可包含：
            - m : 约束函数的数量
            - n : 设计变量的数量
            - xmin : 设计变量的下界
            - xmax : 设计变量的上界
            - a0 : a_0*z 项的常数系数
            - a : a_i*z 项的线性系数
            - c : c_i*y_i 项的线性系数
            - d : 0.5*d_i*(y_i)**2 项的二次项系数
            - asymp_init : 渐近线初始距离的因子
            - asymp_incr : 渐近线矩阵减小的因子
            - asymp_decr : 渐近线矩阵增加的因子
            - move_limit : 移动限制
            - albefa : 计算边界 alfa 和 beta 的因子
            - raa0 : 函数近似精度的参数
            - epsilon_min : 最小容差
        """
        warnings.warn("Modifying advanced options may affect algorithm stability",
                     UserWarning)
        
        valid_params = {
            'm': '_m',
            'n': '_n',
            'xmin': '_xmin',
            'xmax': '_xmax',
            'a0': '_a0',
            'a': '_a',
            'c': '_c',
            'd': '_d',
            'asymp_init': '_asymp_init',
            'asymp_incr': '_asymp_incr',
            'asymp_decr': '_asymp_decr',
            'move_limit': '_move_limit',
            'albefa': '_albefa',
            'raa0': '_raa0',
            'epsilon_min': '_epsilon_min'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, valid_params[key], value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    @property
    def m(self) -> int:
        """约束函数的数量"""
        return self._m

    @property
    def n(self) -> Optional[int]:
        """设计变量的数量"""
        return self._n

    @property
    def xmin(self) -> Optional[TensorLike]:
        """设计变量的下界"""
        return self._xmin

    @property
    def xmax(self) -> Optional[TensorLike]:
        """设计变量的上界"""
        return self._xmax

    @property
    def a0(self) -> float:
        """a_0*z 项的常数系数 a_0"""
        return self._a0

    @property
    def a(self) -> Optional[TensorLike]:
        """a_i*z 项的线性系数 a_i"""
        return self._a

    @property
    def c(self) -> Optional[TensorLike]:
        """c_i*y_i 项的线性系数 c_i"""
        return self._c

    @property
    def d(self) -> Optional[TensorLike]:
        """0.5*d_i*(y_i)**2 项的二次项系数 d_i"""
        return self._d

    @property
    def asymp_init(self) -> float:
        """渐近线初始距离的因子"""
        return self._asymp_init
    
    @property
    def asymp_incr(self) -> float:
        """渐近线矩阵减小的因子"""
        return self._asymp_incr
    
    @property
    def asymp_decr(self) -> float:
        """渐近线矩阵增加的因子"""
        return self._asymp_decr
    
    @property
    def move_limit(self) -> float:
        """移动限制"""
        return self._move_limit
    
    @property
    def albefa(self) -> float:
        """计算边界 alfa 和 beta 的因子"""
        return self._albefa
    
    @property
    def raa0(self) -> float:
        """函数近似精度的参数"""
        return self._raa0
    
    @property
    def epsilon_min(self) -> float:
        """最小容差"""
        return self._epsilon_min


class MMAOptimizer(BaseLogged):
    """Method of Moving Asymptotes (MMA) 优化器
    
    用于求解拓扑优化问题的 MMA 方法实现. 该方法通过动态调整渐近线位置
    来控制优化过程, 具有良好的收敛性能
    
    Parameters:
    -----------
    objective : ComplianceObjective
        目标函数对象
    constraint : VolumeConstraint
        约束条件对象
    filter : Filter
        过滤器对象
    options : MMAOptions, optional
        优化器配置选项
    enable_logging : bool, optional
        是否启用日志记录，默认为 False
    logger_name : str, optional
        日志记录器名称
    """
    
    def __init__(self,
                objective: Union[ComplianceObjective, CompliantMechanismObjective],
                constraint: VolumeConstraint,
                filter: Filter,
                options: MMAOptions = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._objective = objective
        self._constraint = constraint
        self._filter = filter

        # 设置基本参数
        self.options = MMAOptions()
        if options is not None:
            # 只允许设置用户级参数
            user_params = ['max_iterations', 'change_tolerance', 'use_penalty_continuation']
            for key, value in options.items():
                if key in user_params:
                    setattr(self.options, key, value)
                else:
                    error_msg = f"Invalid parameter in options111: {key}. " \
                                f"Use set_advanced_options() for advanced parameters."
                    self._log_error(error_msg)
                    
        # MMA 内部状态
        self._epoch = 0
        self._low = None
        self._upp = None
        
    def _initialize_problem_dependent_params(self, n: int) -> None:
        """初始化依赖于问题规模的参数"""

        # 只在参数未设置时初始化默认值
        if self.options.n is None:
            # 设置所有默认值
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.options.set_advanced_options(
                                                n=n,
                                                xmin=bm.zeros((n, 1)),
                                                xmax=bm.ones((n, 1)),
                                                a=bm.zeros((self.options.m, 1)),
                                                c=1e4 * bm.ones((self.options.m, 1)),
                                                d=bm.zeros((self.options.m, 1))
                                            )

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
        # 获取优化参数
        max_iters = self.options.max_iterations
        change_tol = self.options.change_tolerance   # 设计变量无穷范数阈值

        # 检查或初始化问题相关参数
        if self.options.n is None:
            # 未设置高级参数，使用默认值
            self._initialize_problem_dependent_params(n=design_variable.shape[0])
        else:
            # 已设置高级参数，检查一致性
            if self.options.n != design_variable.shape[0]:
                error_msg = (f"设计变量数量不匹配: "
                            f"设置的 n={self.options.n}, "
                            f"实际的 n={design_variable.shape[0]}")
                self._log_error(error_msg)

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

        rho_phys = self._filter.get_initial_density(density=rho)

        # 初始化历史记录
        history = OptimizationHistory()

        # 初始化历史设计变量
        xold1 = bm.copy(dv[:])
        xold2 = bm.copy(xold1[:])

        #! 目标函数缩放因子
        self._obj_scale_factor = None

        # 优化主循环
        for iter_idx in range(max_iters):

            t = None
            if enable_timing:
                t = timer(f"拓扑优化单次迭代")
                next(t)

            start_time = time()

            self._update_penalty(iter_idx=iter_idx)
            current_penalty = self._objective._analyzer._interpolation_scheme.penalty_factor
            
            # 更新迭代计数
            self._epoch = iter_idx + 1

            # 使用物理密度求解位移场
            if isinstance(self._objective, CompliantMechanismObjective):
                uh = self._objective._analyzer.solve_displacement(rho_val=rho_phys, adjoint=True)
            else:
                uh = self._objective._analyzer.solve_displacement(rho_val=rho_phys)
            if enable_timing:
                t.send('位移场求解')
            
            obj_val_raw = self._objective.fun(rho_phys, displacement=uh)

            #! 动态重置缩放因子
            if self._filter._filter_type == 'projection' and self._obj_scale_factor is None:
                obj0 = float(obj_val_raw.item()) 
                denom = max(abs(obj0), 1e-10)
                self._obj_scale_factor = min(1e6, 10.0 / denom)
                self._log_info(f"Objective scaling factor (re)initialized to: {self._obj_scale_factor:.4e}")
            elif self._obj_scale_factor is None:
                self._obj_scale_factor = 1.0

            obj_val = obj_val_raw * self._obj_scale_factor
            if enable_timing:
                t.send('目标函数计算')
                
            # 灵敏度应用缩放因子
            obj_grad_rho_raw = self._objective.jac(rho_phys, displacement=uh)
            obj_grad_rho = obj_grad_rho_raw * self._obj_scale_factor

            # 计算目标函数相对于设计变量的灵敏度
            obj_grad_dv = self._filter.filter_objective_sensitivities(design_variable=dv, obj_grad_rho=obj_grad_rho)
            if enable_timing:
                t.send('目标函数灵敏度分析 2')

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
            
            # MMA 算法: 
            # 标准化约束函数及其梯度
            cm = self._filter._mesh.entity_measure('cell')
            fval = con_val / (self._constraint.volume_fraction * bm.sum(cm))
            dfdx = con_grad_dv[:, None].T / (self._constraint.volume_fraction * bm.sum(cm))

            # 求解子问题
            dv_new = self._solve_subproblem(
                                        xval=dv[:, None],
                                        fval=fval,
                                        df0dx=obj_grad_dv[:, None],
                                        dfdx=dfdx,
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
            
            # 更新设计变量
            dv = dv_new
                
            # 当前体积分数
            volfrac = self._constraint.get_volume_fraction(rho_phys)

            # 记录当前迭代信息
            iteration_time = time() - start_time

            history.log_iteration(iter_idx=iter_idx, 
                                obj_val=obj_val_raw, 
                                volfrac=volfrac, 
                                change=change,
                                penalty_factor=current_penalty, 
                                time_cost=iteration_time, 
                                physical_density=rho_phys)
            
            #! Beta 更新后的状态重置
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
        history.print_time_statistics()
        
        return rho_phys, history
        
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
        
        # if self._epoch <= 2:
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
                    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """求解 MMA 子问题
        
        Parameters
        ----------
        xval : TensorLike (n, 1)
            当前设计变量
        fval : TensorLike
            约束函数值
        df0dx : TensorLike (n, 1)
            目标函数的梯度
        dfdx : TensorLike (m, n)
            约束函数的梯度
        # low : TensorLike (n, 1)
            # 下渐近线
        # upp : TensorLike (n, 1)
            # 上渐近线
        xold1 : TensorLike (n, 1)
            前一步设计变量
        xold2 : TensorLike (n, 1)
            前两步设计变量

        Returns
        -------
        Tuple[TensorLike, TensorLike, TensorLike]
            MMA 子问题的最优解、下渐近线、上渐近线
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

        #! 动态移动限制
        move = self.options.move_limit
        beta_val = getattr(self._filter, 'beta', None)

        if beta_val is not None:
            move = move / (1.0 + 0.3 * bm.log(beta_val))
        # move = self.options.move_limit

        eeen = bm.ones((n, 1), dtype=bm.float64)
        eeem = bm.ones((m, 1), dtype=bm.float64)
        
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
        
        p0 = bm.maximum(df0dx, 0)   
        q0 = bm.maximum(-df0dx, 0) 
        pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2
        
        # 构建 P, Q 和 b 构建约束函数的近似
        P = bm.zeros((m, n), dtype=bm.float64)
        Q = bm.zeros((m, n), dtype=bm.float64)
        P = bm.maximum(dfdx, 0)
        Q = bm.maximum(-dfdx, 0)
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
