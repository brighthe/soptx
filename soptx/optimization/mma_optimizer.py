import warnings
from time import time
from typing import Optional, Tuple, Union, List, Any

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.compliant_mechanism_objective import CompliantMechanismObjective
from soptx.optimization.compliance_objective import ComplianceObjective
from soptx.optimization.volume_constraint import VolumeConstraint
from soptx.optimization.stress_constraint import StressConstraint
from soptx.optimization.tools import OptimizationHistory
from soptx.optimization.utils import solve_mma_subproblem
from soptx.regularization.filter import Filter
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

class MMAOptions:
    """MMA 算法的配置选项"""

    def __init__(self):
        """初始化参数的默认值"""
        # --- 用户常用参数 ---
        self.max_iterations = 200             # 最大迭代次数
        self.change_tolerance = 1e-2          # 设计变量无穷范数阈值
        self.use_penalty_continuation = True  # 是否使用惩罚因子连续化技术

        # --- 几何/渐近线控制参数 ---
        # 控制渐近线的移动速度和范围
        self._move_limit = 0.2        # 移动限制
        self._asymp_init = 0.5        # 初始渐近线距离
        self._asymp_incr = 1.2        # 渐近线扩张系数
        self._asymp_decr = 0.7        # 渐近线收缩系数

        # --- 子问题求解数值参数 ---
        # 控制对偶求解器的行为
        self._albefa = 0.1            # 计算边界 alfa 和 beta 的因子
        self._raa0 = 1e-5             # 函数近似精度的参数
        self._epsilon_min = 1e-7      # 子问题求解的最小数值容差

        # --- 惩罚项策略配置 ---
        # 控制 a, c, d 向量的生成策略
        self._a0 = 1.0               # a0 常数
        self._a = None               # a 向量的默认系数
        self._d = None               # d 向量的默认系数
        self._c = None               # c 向量的默认系数 (惩罚项)

        # --- 内部状态 ---
        self._m = None               # 当前约束函数的数量
        self._n = None               # 当前设计变量的数量
        self._xmin = None
        self._xmax = None

    def _initialize_problem_params(self, m: int, n: int) -> None:
        """初始化问题规模相关参数
                
        Parameters
        ----------
        m : int
            约束函数的数量
        n : int
            设计变量的数量
        """
        self._m = m
        self._n = n
        self._xmin = bm.zeros((n, 1), dtype=bm.float64)
        self._xmax = bm.ones((n, 1), dtype=bm.float64)
        self._a = bm.zeros((m, 1), dtype=bm.float64)
        self._c = 1e3 * bm.ones((m, 1), dtype=bm.float64)
        # self._c = 1000 * bm.ones((m, 1), dtype=bm.float64)
        self._d = bm.zeros((m, 1), dtype=bm.float64)

    def set_advanced_options(self, **kwargs):
        """设置高级选项，仅供专业用户使用
        
        Parameters
        ----------
        **kwargs : 高级参数设置，可包含：
            - a0 : float
                a_0*z 项的常数系数，默认 1.0
            - asymp_init : float
                渐近线初始距离因子，默认 0.5
            - asymp_incr : float
                渐近线距离增大因子，默认 1.2
            - asymp_decr : float
                渐近线距离减小因子，默认 0.7
            - move_limit : float
                移动限制，默认 0.2
            - albefa : float
                计算边界 alfa 和 beta 的因子，默认 0.1
            - raa0 : float
                函数近似精度的参数，默认 1e-5
            - epsilon_min : float
                最小容差，默认 1e-7
        """
        warnings.warn("Modifying advanced options may affect algorithm stability",
                    UserWarning)
        
        valid_params = {
            'a0': '_a0',
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
    def __init__(self,
                objective: Union[ComplianceObjective, CompliantMechanismObjective],
                constraint: Union[VolumeConstraint, StressConstraint, List[Any]],
                filter: Filter,
                options: MMAOptions = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """Method of Moving Asymptotes (MMA) 优化器
    
        用于求解拓扑优化问题的 MMA 方法实现. 该方法通过动态调整渐近线位置
        来控制优化过程, 具有良好的收敛性能
        
        Parameters
        ----------
        objective: 目标函数对象
        constraint: 约束条件对象，支持单约束或约束列表
        filter: 过滤器对象
        options: 优化器配置选项
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._objective = objective

        # 约束标准化处理: 统一转换为 list
        if isinstance(constraint, list):
            self._constraints = constraint
        else:
            self._constraints = [constraint]

        self._filter = filter

        if isinstance(options, MMAOptions):
            self.options = options
        else:
            self.options = MMAOptions()
            # 如果传入的是字典，则覆盖默认值
            if isinstance(options, dict):
                for key, value in options.items():
                    # 仅允许设置公开属性
                    if hasattr(self.options, key) and not key.startswith('_'):
                        setattr(self.options, key, value)
                    else:
                        self._log_warning(f"Ignored unknown or private parameter in options: '{key}'")
            elif options is not None:
                self._log_error("The 'options' parameter must be a dict or an MMAOptions instance.")
        
        # MMA 内部状态初始化
        self._epoch = 0
        self._low = None
        self._upp = None

        #TODO ==================== 被动单元预处理 ====================
        # 判断是否存在应力约束
        self._has_stress_constraint = any(
            isinstance(c, StressConstraint) for c in self._constraints
        )
        
        # 仅当存在应力约束时获取被动单元掩码
        pde = self._objective._analyzer.pde
        self._passive_mask = None
        if self._has_stress_constraint:
            design_mesh = getattr(self._filter, 'design_mesh', None)
            if hasattr(pde, 'get_passive_element_mask'):
                self._passive_mask = pde.get_passive_element_mask(mesh=design_mesh)

                if self._passive_mask is not None:
                    for c in self._constraints:
                        if isinstance(c, StressConstraint):
                            if hasattr(c, 'set_passive_mask'):
                                c.set_passive_mask(self._passive_mask)

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

    def _count_total_constraints(self) -> int:
        """计算所有约束的总数"""
        total = 0
        for constraint in self._constraints:
            if isinstance(constraint, VolumeConstraint):
                total += 1
            elif isinstance(constraint, StressConstraint):
                total += constraint._n_clusters  
            else:
                total += 1 
        return total
    
    def _apply_passive_mask(self, dv, rho) -> None:
        """将被动单元强制设为实体材料"""
        dv[self._passive_mask] = 1.0
        
        if rho.ndim == 1:
            rho[self._passive_mask] = 1.0
        else:
            from soptx.analysis.utils import reshape_multiresolution_data_inverse
            NC, n_sub = rho.shape
            disp_mesh = getattr(self._objective._analyzer, 'disp_mesh', None)
            nx_disp, ny_disp = disp_mesh.meshdata['nx'], disp_mesh.meshdata['ny']
            passive_mask_rho = reshape_multiresolution_data_inverse(
                nx_disp, ny_disp, self._passive_mask, n_sub
            )
            rho[passive_mask_rho] = 1.0

        # design_mesh = getattr(self._filter, 'design_mesh', None)
        # design_mesh.celldata['rho'] = rho
        # from pathlib import Path
        # current_file = Path(__file__)
        # base_dir = current_file.parent.parent / 'vtu'
        # base_dir = str(base_dir)
        # save_path = Path(f"{base_dir}/")
        # design_mesh.to_vtk(f"{save_path}/rho_section5.vtu")

    def optimize(self,
                design_variable: Union[Function, TensorLike], 
                density_distribution: Union[Function, TensorLike], 
                is_store_stress: bool = False,
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
        m = self._count_total_constraints()
        n = design_variable.shape[0]
        self.options._initialize_problem_params(m, n)

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

        #TODO ==================== 被动单元处理 ====================
        if self._passive_mask is not None:
            self._apply_passive_mask(dv, rho)

        # 初始物理密度
        rho_phys = self._filter.get_initial_density(density=rho)

        # ==================== MMA 历史变量初始化 ====================
        xold1 = bm.copy(dv[:])
        xold2 = bm.copy(xold1[:])

        # ==================== 优化状态初始化 ====================
        self.history = OptimizationHistory()
        #TODO 初始化目标函数缩放因子
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
            print(f"初始惩罚因子: {current_penalty}")
            
            # 更新迭代计数
            self._epoch = iter_idx + 1

            #TODO 基于物理密度求解状态变量
            if hasattr(analyzer, 'solve_state'):
                state = analyzer.solve_state(rho_val=rho_phys)
            elif isinstance(self._objective, CompliantMechanismObjective):
                state = analyzer.solve_state(rho_val=rho_phys, adjoint=True)

            if enable_timing:
                t.send('位移场求解')

            #TODO ==================== 目标函数计算 ====================
            obj_val_raw = self._objective.fun(density=rho_phys, state=state)

            # 动态初始化缩放因子
            if self._obj_scale_factor is None:
                
                obj0 = float(obj_val_raw.item()) 
                denom = max(abs(obj0), 1e-10)

                # 情况 A: 存在应力约束 -> 强制缩放至 1.0 以平衡梯度量级
                if self._has_stress_constraint:
                    # target_initial_val = 1.0
                    # self._obj_scale_factor = min(1e6, target_initial_val / denom)
                    self._obj_scale_factor = 1.0
                
                # 情况 B: 无应力约束，但使用投影滤波 -> 缩放至 10.0
                elif self._filter._filter_type == 'projection':
                    target_initial_val = 10.0
                    self._obj_scale_factor = min(1e6, target_initial_val / denom)
                
                # 情况 C: 其他情况 -> 默认 1.0 不缩放
                else:
                    self._obj_scale_factor = 1.0

            # 目标函数应用缩放因子
            obj_val = obj_val_raw * self._obj_scale_factor

            if enable_timing:
                t.send('目标函数计算')

            #TODO ==================== 目标函数灵敏度 ====================
            # 1. 计算目标函数相对于物理密度的灵敏度
            obj_grad_rho_raw = self._objective.jac(density=rho_phys, state=state)
            # 灵敏度应用缩放因子
            obj_grad_rho = obj_grad_rho_raw * self._obj_scale_factor 
            print(f"目标函数相对于物理密度的灵敏度范围: [{bm.min(obj_grad_rho):.4f}, {bm.max(obj_grad_rho):.4f}]")
            # 2. 计算目标函数相对于设计变量的灵敏度
            obj_grad_dv = self._filter.filter_objective_sensitivities(design_variable=dv, obj_grad_rho=obj_grad_rho)
            print(f"目标函数相对于设计变量的灵敏度范围: [{bm.min(obj_grad_dv):.4f}, {bm.max(obj_grad_dv):.4f}]")

            if self._passive_mask is not None:
                obj_grad_dv[self._passive_mask] = 0.0
                
            if enable_timing:
                t.send('目标函数灵敏度分析')

            #TODO ==================== 约束函数计算 ====================
            con_vals = []
            con_grads_dv = []

            for constraint in self._constraints:
                val = constraint.fun(density=rho_phys, state=state, iter_idx=iter_idx)
                # 相对于物理密度的灵敏度
                grad_rho = constraint.jac(density=rho_phys, state=state)
                # 相对于设计变量的灵敏度
                grad_dv = self._filter.filter_constraint_sensitivities(design_variable=dv, con_grad_rho=grad_rho)

                if self._passive_mask is not None:
                    if grad_dv.ndim == 1:
                        grad_dv[self._passive_mask] = 0.0
                    else:
                        grad_dv[:, self._passive_mask] = 0.0

                val_norm, grad_norm = constraint.normalize(val, grad_dv)

                if val_norm.ndim == 0:
                    val_flat = val_norm.reshape(1)
                    grad_flat = grad_norm.reshape(1, -1)  # (1, n)
                elif val_norm.ndim == 1:
                    val_flat = val_norm  # (n_clusters,)
                    grad_flat = grad_norm if grad_norm.ndim == 2 else grad_norm.reshape(1, -1)  # (n_clusters, n)
                else:
                    raise ValueError(f"Unexpected constraint value shape: {val_norm.shape}")

                con_vals.append(val_flat)
                con_grads_dv.append(grad_flat)

            if enable_timing:
                t.send('约束函数灵敏度分析')

            # print(f"体积约束:{con_vals[0]}")
            # print(f"体积约束梯度平均值:{bm.mean(con_grads_dv[0])}")
            print(f"应力约束:{con_vals[0:]}")
            print(f"应力约束梯度平均值:{bm.mean(con_grads_dv[0:])}")
            
            #TODO ==================== MMA 子问题求解 ====================
            fval = bm.concatenate(con_vals).reshape(-1, 1)  # (m, 1)
            dfdx = bm.concatenate(con_grads_dv, axis=0)     # (m, n)

            # 求解子问题
            dv_new = self._solve_subproblem(
                                        xval=dv[:, None],
                                        fval=fval,
                                        df0dx=obj_grad_dv[:, None],
                                        dfdx=dfdx,
                                        xold1=xold1[:, None],
                                        xold2=xold2[:, None]
                                    )
            
            #TODO 强制被动单元为实体材料
            if self._passive_mask is not None:
                dv_new[self._passive_mask] = 1.0

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
            print(f"设计变量最大变化量: {change}")
            
            # 更新设计变量
            dv = dv_new
                
            # 当前体积分数
            volfrac = None
            for constraint in self._constraints:
                if hasattr(constraint, 'get_volume_fraction'):
                    volfrac = constraint.get_volume_fraction(rho_phys)
                    break

            # 记录当前迭代信息
            iteration_time = time() - start_time

            von_mises_stress = None
            if is_store_stress:
                stress_state = analyzer.compute_stress_state(
                                        state=state,
                                        rho_val=rho_phys,
                                    )
                von_mises_stress = stress_state['von_mises_max']
                if enable_timing:
                    t.send('von Mises 应力计算')

            self.history.log_iteration(iter_idx=iter_idx, 
                                obj_val=obj_val_raw, 
                                volfrac=volfrac, 
                                change=change,
                                penalty_factor=current_penalty, 
                                time_cost=iteration_time, 
                                physical_density=rho_phys,
                                von_mises_stress=von_mises_stress
                            )
            
            #TODO Beta 更新后的状态重置
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

        #! 动态移动限制: 根据投影滤波器的 beta 值调整
        move = self.options.move_limit
        beta_val = getattr(self._filter, 'beta', None)

        if beta_val is not None:
            move = move / (1.0 + 0.3 * bm.log(beta_val))

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
