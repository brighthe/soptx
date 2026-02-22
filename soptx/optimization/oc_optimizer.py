import warnings
from time import time
from typing import Optional, Union, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.optimization.compliant_mechanism_objective import CompliantMechanismObjective
from soptx.optimization.compliance_objective import ComplianceObjective
from ..optimization.volume_constraint import VolumeConstraint
from ..optimization.tools import OptimizationHistory
from ..regularization.filter import Filter
from ..utils.base_logged import BaseLogged
from soptx.optimization.utils import compute_volume
from soptx.utils import timer

class OCOptions:
    """OC 算法的配置选项"""

    def __init__(self):
        """初始化参数的默认值"""
        # OC 算法的用户级参数
        self.max_iterations = 200        # 最大迭代次数
        self.change_tolerance = 1e-2     # 设计变量无穷范数阈值

        # OC 算法的高级参数
        # 柔顺度最小化参数组合: m = 0.2, η = 0.5, λ = 1e9, btol = 1e-3, dmin = 1e-9
        # 柔顺机械设计参数组合: m = 0.1, η = 0.3, λ = 1e5, btol = 1e-4, dmin = 1e-3
        self._move_limit = 0.2
        self._damping_coef = 0.5
        self._initial_lambda = 1e9
        self._bisection_tol = 1e-3
        self._design_variable_min = 1e-9 
        
    def set_advanced_options(self, **kwargs):
        """设置高级选项，仅供专业用户使用
        
        Parameters
        ----------
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
            'bisection_tol': '_bisection_tol',
            'design_variable_min': '_design_variable_min'
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
                objective: Union[ComplianceObjective, CompliantMechanismObjective],
                constraint: VolumeConstraint,
                filter: Filter,
                options: OCOptions = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._objective = objective
        self._constraint = constraint
        self._filter = filter

        # 设置基本参数
        self.options = OCOptions()
        if options is not None:
            # 只允许设置用户级参数
            user_params = ['max_iterations', 'change_tolerance']
            for key, value in options.items():
                if key in user_params:
                    setattr(self.options, key, value)
                else:
                    error_msg = f"Invalid parameter in options: {key}. " \
                                f"Use set_advanced_options() for advanced parameters."
                    self._log_error(error_msg)

        # 获取被动单元掩码
        self._passive_mask = None

        analyzer = getattr(objective, '_analyzer', None)
        pde = getattr(analyzer, 'pde', None)
        design_mesh = getattr(self._filter, 'design_mesh', None)

        if pde is not None and design_mesh is not None:
            if hasattr(pde, 'get_passive_element_mask'):
                self._passive_mask = pde.get_passive_element_mask(mesh=design_mesh)

    def optimize(self, 
                design_variable: Union[Function, TensorLike],
                density_distribution: Union[Function, TensorLike],
                enable_timing: bool = False,
                is_store_stress: bool = False,
                **kwargs
            ) -> Tuple[TensorLike, OptimizationHistory]:
        """运行 OC 优化算法

        Parameters
        ----------
        design_variable : 设计变量
        density_distribution : 密度分布
        **kwargs : 其他参数
        """

        # 获取优化参数
        max_iters = self.options.max_iterations
        change_tol = self.options.change_tolerance   # 设计变量无穷范数阈值
        bisection_tol = self.options.bisection_tol

        if isinstance(design_variable, Function):
            dv = design_variable.space.function(bm.copy(design_variable[:]))
        else:
            dv = bm.copy(design_variable[:])
        
        if isinstance(density_distribution, Function):
            rho = density_distribution.space.function(bm.copy(density_distribution[:]))
        else:
            rho = bm.copy(density_distribution[:])

        # 初始化时设置被动单元 
        if self._passive_mask is not None:
            dv[self._passive_mask] = 1.0
            rho[self._passive_mask] = 1.0

        # design_mesh = getattr(self._filter, 'design_mesh', None)
        # # design_mesh.celldata['rho'] = rho
        # design_mesh.nodedata['rho'] = rho
        # from pathlib import Path
        # current_file = Path(__file__)
        # base_dir = current_file.parent.parent / 'vtu'
        # base_dir = str(base_dir)
        # save_path = Path(f"{base_dir}/")
        # design_mesh.to_vtk(f"{save_path}/rho_section3.vtu")

        rho_phys = self._filter.get_initial_density(density=rho)

        analyzer = self._objective._analyzer

        # 初始化历史记录
        history = OptimizationHistory()

        # 优化主循环
        for iter_idx in range(max_iters):

            t = None
            if enable_timing:
                t = timer(f"拓扑优化单次迭代")
                next(t)

            start_time = time()

            # TODO 基于物理密度求解状态变量 (待完善)
            state = analyzer.solve_state(rho_val=rho_phys)
            if isinstance(self._objective, CompliantMechanismObjective):
                state = analyzer.solve_state(rho_val=rho_phys, adjoint=True)

            if enable_timing:
                t.send('位移场求解')

            # 使用物理密度和位移计算目标函数
            obj_val = self._objective.fun(density=rho_phys, state=state)
            if enable_timing:
                t.send('目标函数计算')

            # 计算目标函数相对于物理密度的灵敏度
            obj_grad_rho = self._objective.jac(density=rho_phys, state=state)
            if enable_timing:
                t.send('目标函数灵敏度分析 1')

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

            # OC 算法: 二分法求解拉格朗日乘子
            l1, l2 = 0.0, self.options.initial_lambda
            while (l2 - l1) / (l2 + l1) > bisection_tol and l2 > 1e-40:
                lmid = 0.5 * (l2 + l1)
                dv_new = self._update_density(design_variable=dv, dc=obj_grad_dv, dg=con_grad_dv, lmid=lmid)

                # 过滤后得到的物理密度
                rho_phys = self._filter.filter_design_variable(design_variable=dv_new, physical_density=rho_phys)

                # 检查约束函数值
                if self._constraint.fun(rho_phys) > 0:
                    l1 = lmid
                else:
                    l2 = lmid
            if enable_timing:
                t.send('OC 优化')

            # 设计变量变化（无穷范数）
            change = bm.max(bm.abs(dv_new - dv))

            # 更新设计变量
            dv = dv_new

            # 当前体积分数
            current_volume = self._constraint._v
            total_volume = self._constraint._v0
            volfrac = current_volume / total_volume

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

            self._log_info(
                    f"Iteration: {iter_idx + 1}, "
                    f"Objective: {obj_val:.4f}, "
                    f"Volfrac: {volfrac:.4f}, "
                    f"Change: {change:.4f}, "
                    f"Time: {iteration_time:.3f} sec"
                )
            
            scalars = {
                        'compliance': obj_val,
                        'volfrac': volfrac,
                    }
            
            fields = {}

            history.log_iteration(
                            iter_idx=iter_idx + 1,
                            change=change,
                            time_cost=iteration_time,
                            physical_density=rho_phys,
                            scalars=scalars,
                            fields=fields,
                        )
            
            change, beta_updated = self._filter.continuation_step(change)
            if beta_updated:
                    continue
            
            if enable_timing:
                t.send(None)

            # 收敛检查
            if change <= change_tol:
                msg = (f"Converged after {iter_idx + 1} iterations "
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
        dmin = self.options._design_variable_min
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

        B_e = -dc / (dg * lmid)
        clip_B = 1e-12
        B_e_clipped = bm.maximum(bm.tensor(clip_B, **kwargs), B_e)
        B_e_damped = bm.pow(B_e_clipped, eta)

        dv_new[:] = bm.maximum(
                        bm.tensor(dmin, **kwargs), 
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
        
        # 强制被动单元保持密度为 1.0
        if self._passive_mask is not None:
            dv_new[self._passive_mask] = 1.0

        return dv_new
        
