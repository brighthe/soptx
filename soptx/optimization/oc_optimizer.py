import warnings
from dataclasses import dataclass
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


@dataclass
class OCOptions:
    """OC 算法的配置选项"""
    # 用户级参数：直接暴露给用户
    max_iterations: int = 100     # 最大迭代次数
    tolerance: float = 1e-3       # 收敛容差

    # 高级参数：通过专门的方法修改
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

    def __init__(self):
        """初始化高级参数的默认值"""
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
                    raise ValueError(error_msg)

    def optimize(self, 
                density_distribution: Union[Function, TensorLike], **kwargs
            ) -> Tuple[TensorLike, OptimizationHistory]:
        """运行 OC 优化算法

        Parameters:
        -----------
        density_distribution : 初始相对密度场
        **kwargs : 其他参数
        """

        # 获取优化参数
        max_iters = self.options.max_iterations
        tol = self.options.tolerance
        bisection_tol = self.options.bisection_tol

        rho = density_distribution
        
        if isinstance(rho, Function):
            rho_Phys = rho.space.function(bm.copy(rho[:]))
        else:
            rho_Phys = bm.copy(rho[:])

        rho_Phys = self._filter.get_initial_density(rho=rho, rho_Phys=rho_Phys)

        # 初始化历史记录
        history = OptimizationHistory()

        # 优化主循环
        for iter_idx in range(max_iters):
            
            start_time = time()

            # 使用物理密度计算约束函数值梯度
            obj_val = self._objective.fun(rho_Phys)
            obj_grad = self._objective.jac(rho_Phys)

            # 过滤目标函数灵敏度
            obj_grad = self._filter.filter_objective_sensitivities(rho_Phys=rho_Phys, obj_grad=obj_grad)

            # 使用物理密度计算约束函数值梯度
            con_val = self._constraint.fun(rho_Phys)
            con_grad = self._constraint.jac(rho_Phys)

            # 过滤约束函数灵敏度
            con_grad = self._filter.filter_constraint_sensitivities(rho_Phys=rho_Phys, con_grad=con_grad)
            
            if iter_idx == 17:
                print("----------------------------")

            # 二分法求解拉格朗日乘子
            l1, l2 = 0.0, self.options.initial_lambda
            while (l2 - l1) / (l2 + l1) > bisection_tol:
                lmid = 0.5 * (l2 + l1)
                rho_new = self._update_density(rho=rho, dc=obj_grad, dg=con_grad, lmid=lmid)

                # 过滤物理密度
                rho_Phys = self._filter.filter_variables(rho=rho_new, rho_Phys=rho_Phys)

                # 检查约束函数值
                if self._constraint.fun(rho_Phys) > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            # 计算收敛性
            change = bm.max(bm.abs(rho_new - rho))

            # 更新设计变量，确保目标函数内部状态同步
            rho = rho_new

            # 当前体积分数
            volfrac = self._constraint.get_volume_fraction(rho_Phys)

            iteration_time = time() - start_time

            history.log_iteration(iter_idx=iter_idx, obj_val=obj_val, volfrac=volfrac, 
                                change=change, time_cost=iteration_time, physical_density=rho_Phys)

            # 收敛检查
            if change <= tol:
                msg = f"Converged after {iter_idx + 1} iterations"
                self._log_info(msg)
                break

        # 打印时间统计信息
        history.print_time_statistics()
                
        return rho, history
    
    def _update_density(self,
                        rho: Union[Function, TensorLike],
                        dc: TensorLike,
                        dg: TensorLike,
                        lmid: float
                    ) -> Union[Function, TensorLike]:
        """使用 OC 准则更新密度"""

        # 获取算法内部参数
        m = self.options.move_limit
        eta = self.options.damping_coef
        kwargs = bm.context(rho)

        if isinstance(rho, Function):
            rho_new = rho.space.function(bm.copy(rho[:]))
        else:
            rho_new = bm.copy(rho[:])
        
        if (bm.any(bm.isnan(rho[:])) or bm.any(bm.isinf(rho[:])) or 
            bm.any(rho[:] < -1e-12) or bm.any(rho[:] > 1 + 1e-12)):
            self._log_error(f"输入密度超出合理范围 [0, 1]: "
                            f"range=[{bm.min(rho):.2e}, {bm.max(rho):.2e}]")
            
        if bm.any(dc > 1e-12):
            self._log_error(f"目标函数梯度中存在正值, 可能导致目标函数上升")

        # 使用绝对值避免负数开方
        B_e = -dc / (dg * lmid)
        B_e_abs = bm.abs(B_e)
        B_e_damped = bm.pow(B_e_abs, eta)

        rho_new[:] = bm.maximum(
            bm.tensor(0.0, **kwargs), 
            bm.maximum(
                rho - m, 
                bm.minimum(
                    bm.tensor(1.0, **kwargs), 
                    bm.minimum(
                        rho + m, 
                        rho * B_e_damped
                    )
                )
            )
        )

        if (bm.any(bm.isnan(rho_new[:])) or bm.any(bm.isinf(rho_new[:])) or 
            bm.any(rho_new[:] < -1e-12) or bm.any(rho_new[:] > 1 + 1e-12)):
            self._log_error(f"输入密度超出合理范围 [0, 1]: "
                            f"range=[{bm.min(rho_new):.2e}, {bm.max(rho_new):.2e}]")
        
        return rho_new
        
