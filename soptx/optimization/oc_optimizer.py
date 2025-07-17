from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from ..regularization.filter import Filter

class OCOptimizer():
    def __init__(self, 
                objective, 
                filter: Filter):
        
        self.objective = objective
        self.filter = filter

    def optimize(self, rho: TensorLike, **kwargs) -> TensorLike:
        """运行 OC 优化算法

        Parameters:
        -----------
        rho : 初始密度场
        **kwargs : 其他参数
        """

        tensor_kwargs = bm.context(rho)
        rho_phys = bm.zeros_like(rho, **tensor_kwargs)
        rho_phys = self.filter.get_initial_density(rho, rho_phys)

        # 优化主循环
        for iter_idx in range(max_iters):

            # 使用物理密度计算约束函数值梯度
            obj_val = self.objective.fun(rho_phys)
            obj_grad = self.objective.jac(rho_phys)

            # 过滤目标函数灵敏度 (灵敏度过滤)
            obj_grad = self.filter.filter_objective_sensitivities(
                                    rho_phys=rho_phys, obj_grad=obj_grad)
            
            # 当前体积分数
            vol_frac = self.constraint.get_volume_fraction(rho_phys)

                        # 当前体积分数
            vol_frac = self.constraint.get_volume_fraction(rho_phys)
            
            # 二分法求解拉格朗日乘子
            l1, l2 = 0.0, self.options.initial_lambda
            while (l2 - l1) / (l2 + l1) > bisection_tol:
                lmid = 0.5 * (l2 + l1)
                rho_new = self._update_density(rho, obj_grad, con_grad, lmid)
                
                # 计算新的物理密度
                rho_phys = self.filter.filter_variables(rho_new, rho_phys)

                # 检查约束函数值
                if self.constraint.fun(rho_phys) > 0:
                    l1 = lmid
                else:
                    l2 = lmid
        
