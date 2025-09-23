import warnings
from time import time
from typing import Optional, Union, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from ..optimization.compliance_objective import ComplianceObjective
from ..optimization.volume_constraint import VolumeConstraint
from ..optimization.tools import OptimizationHistory
from ..regularization.filter import Filter
from ..utils.base_logged import BaseLogged

class OCOptions:
    """OC 算法的配置选项"""

    def __init__(self):
        """初始化参数的默认值"""
        # OC 算法的用户级参数
        self.max_iterations = 200     # 最大迭代次数
        self.tolerance = 0.001        # 收敛容差

        # OC 算法的高级参数
        self._move_limit = 0.2
        self._damping_coef = 0.5
        self._initial_lambda = 1e9
        self._bisection_tol = 1e-3
        
    def set_advanced_options(self, **kwargs):
        """设置高级选项，仅供专业用户使用
        
        Parameters:
        -----------
        - **kwargs : 高级参数设置，可包含：
            - move_limit : 移动限制
            - damping_coef : 阻尼系数
            - initial_lambda : 初始 lambda 值
            - bisection_tol : 二分法收敛容差
        """
        warnings.warn("Modifying advanced options may affect algorithm stability",
                     UserWarning)
        
        valid_params = {
            'move_limit': '_move_limit',
            'damping_coef': '_damping_coef',
            'initial_lambda': '_initial_lambda',
            'bisection_tol': '_bisection_tol'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, valid_params[key], value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    @property
    def move_limit(self) -> float:
        """正向移动限制 m"""
        return self._move_limit
        
    @property
    def damping_coef(self) -> float:
        """阻尼系数 η"""
        return self._damping_coef
        
    @property
    def initial_lambda(self) -> float:
        """初始 lambda 值"""
        return self._initial_lambda
        
    @property
    def bisection_tol(self) -> float:
        """二分法收敛容差"""
        return self._bisection_tol

class OCOptimizer(BaseLogged):
    def __init__(self,
                objective: ComplianceObjective,
                constraint: VolumeConstraint,
                filter: Filter,
                options: OCOptions = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._objective = objective
        self._constraint = constraint
        self._filter = filter

        # 设置基本参数
        self.options = OCOptions()
        if options is not None:
            # 只允许设置用户级参数
            user_params = ['max_iterations', 'tolerance']
            for key, value in options.items():
                if key in user_params:
                    setattr(self.options, key, value)
                else:
                    error_msg = f"Invalid parameter in options: {key}. " \
                                f"Use set_advanced_options() for advanced parameters."
                    self._log_error(error_msg)

    def optimize(self, 
                design_variable: Union[Function, TensorLike],
                density_distribution: Union[Function, TensorLike], 
                **kwargs
            ) -> Tuple[TensorLike, OptimizationHistory]:
        """运行 OC 优化算法

        Parameters:
        -----------
        design_variable : 设计变量
        density_distribution : 密度分布
        **kwargs : 其他参数
        """

        # 获取优化参数
        max_iters = self.options.max_iterations
        tol = self.options.tolerance
        bisection_tol = self.options.bisection_tol

        if isinstance(design_variable, Function):
            dv = design_variable.space.function(bm.copy(design_variable[:]))
        else:
            dv = bm.copy(design_variable[:])
        
        if isinstance(density_distribution, Function):
            rho = density_distribution.space.function(bm.copy(density_distribution[:]))
        else:
            rho = bm.copy(density_distribution[:])

        rho_phys = self._filter.get_initial_density(density=rho)

        # 初始化历史记录
        history = OptimizationHistory()

        # 优化主循环
        for iter_idx in range(max_iters):
            
            start_time = time()

            # 使用物理密度计算目标函数
            obj_val = self._objective.fun(rho_phys)

            # 计算目标函数相对于物理密度的灵敏度
            obj_grad_rho = self._objective.jac(rho_phys)

            # 计算目标函数相对于设计变量的灵敏度
            obj_grad_dv = self._filter.filter_objective_sensitivities(design_variable=dv, obj_grad_rho=obj_grad_rho)

            # 使用物理密度计算约束函数
            con_val = self._constraint.fun(rho_phys)

            # 计算约束函数相对于物理密度的灵敏度
            con_grad_rho = self._constraint.jac(rho_phys)

            # 计算约束函数相对于设计变量的灵敏度
            con_grad_dv = self._filter.filter_constraint_sensitivities(design_variable=dv, con_grad_rho=con_grad_rho)

            # OC 算法: 二分法求解拉格朗日乘子
            l1, l2 = 0.0, self.options.initial_lambda
            while (l2 - l1) / (l2 + l1) > bisection_tol:
                lmid = 0.5 * (l2 + l1)
                dv_new = self._update_density(design_variable=dv, dc=obj_grad_dv, dg=con_grad_dv, lmid=lmid)

                # 过滤后得到的物理密度
                rho_phys = self._filter.filter_design_variable(design_variable=dv_new, physical_density=rho_phys)

                # 检查约束函数值
                if self._constraint.fun(rho_phys) > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            # 计算收敛性
            change = bm.max(bm.abs(dv_new - dv))

            # 更新设计变量
            dv = dv_new

            # 当前体积分数
            volfrac = self._constraint.get_volume_fraction(rho_phys)

            iteration_time = time() - start_time

            history.log_iteration(iter_idx=iter_idx, 
                                obj_val=obj_val, 
                                volfrac=volfrac, 
                                change=change,
                                penalty_factor=self._objective._analyzer._interpolation_scheme.penalty_factor, 
                                time_cost=iteration_time, 
                                physical_density=rho_phys)

            # 收敛检查
            if change <= tol:
                msg = f"Converged after {iter_idx + 1} iterations"
                self._log_info(msg)
                break

        # 打印时间统计信息
        history.print_time_statistics()
                
        return rho_phys, history
    
    def _update_density(self,
                        design_variable: Union[Function, TensorLike],
                        dc: TensorLike,
                        dg: TensorLike,
                        lmid: float
                    ) -> Union[Function, TensorLike]:
        """使用 OC 准则更新设计变量"""

        # 获取算法内部参数
        m = self.options.move_limit
        eta = self.options.damping_coef
        kwargs = bm.context(design_variable)

        dv = design_variable

        if (bm.any(bm.isnan(dv[:])) or bm.any(bm.isinf(dv[:])) or
            bm.any(dv[:] < -1e-12) or bm.any(dv[:] > 1 + 1e-12)):
            self._log_error(f"输入设计变量超出合理范围 [0, 1]: "
                            f"range=[{bm.min(dv):.2e}, {bm.max(dv):.2e}]")

        if isinstance(dv, Function):
            dv_new = dv.space.function(bm.copy(dv[:]))
        else:
            dv_new = bm.copy(dv[:])
    

        # 使用绝对值避免负数开方
        B_e = -dc / (dg * lmid)
        B_e_abs = bm.abs(B_e)
        B_e_damped = bm.pow(B_e_abs, eta)

        dv_new[:] = bm.maximum(
            bm.tensor(0.0, **kwargs), 
            bm.maximum(
                dv - m, 
                bm.minimum(
                    bm.tensor(1.0, **kwargs), 
                    bm.minimum(
                        dv + m, 
                        dv * B_e_damped
                    )
                )
            )
        )

        if (bm.any(bm.isnan(dv_new[:])) or bm.any(bm.isinf(dv_new[:])) or
            bm.any(dv_new[:] < -1e-12) or bm.any(dv_new[:] > 1 + 1e-12)):
            self._log_error(f"输出设计变量超出合理范围 [0, 1]: "
                            f"range=[{bm.min(dv_new):.2e}, {bm.max(dv_new):.2e}]")

        return dv_new
        
