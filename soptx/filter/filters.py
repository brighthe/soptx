from typing import Literal, Dict, Type, Tuple

from fealpy.mesh import Mesh
from fealpy.typing import TensorLike

from .matrix_builder import FilterMatrixBuilder 
from .filter_strategies import (
                                _FilterStrategy,
                                NoneStrategy,
                                DensityStrategy,
                                SensitivityStrategy, 
                                HeavisideDensityStrategy,
                            )

# 策略注册表
FILTER_STRATEGY_REGISTRY: Dict[str, Type[_FilterStrategy]] = \
                        {
                            'none': NoneStrategy,
                            'sensitivity': SensitivityStrategy,
                            'density': DensityStrategy,
                            'heaviside_density': HeavisideDensityStrategy,
                        }

class Filter:
    """统一的过滤方法接口类

    该类使用策略模式来动态选择和应用不同的过滤算法
    """
    def __init__(self,
                mesh: Mesh,
                rmin: float,
                filter_type: Literal['none', 'sensitivity', 'density', 'heaviside_density'],
                # Heaviside 投影相关参数
                beta: float = 1.0,
                max_beta: float = 512,
                continuation_iter: int = 50
            ) -> None:
        
        self.mesh = mesh
        self.rmin = rmin
        self.filter_type = filter_type
        
        # 1. 构建过滤矩阵
        if self.filter_type != 'none' and self.rmin > 0:
            builder = FilterMatrixBuilder(mesh, rmin)
            self._H, self._Hs = builder.build()
            self._cell_measure = self.mesh.entity_measure('cell')
            self._normalize_factor = self._H.matmul(self._cell_measure)
        else:
            self._H, self._Hs, self._cell_measure, self._normalize_factor = None, None, None, None

        # 2. 策略选择和实例化
        strategy_class = FILTER_STRATEGY_REGISTRY.get(self.filter_type)
        if strategy_class is None:
            raise ValueError(f"未知的过滤方法: '{self.filter_type}'. "
                             f"可用选项: {list(FILTER_STRATEGY_REGISTRY.keys())}")

        # 准备策略所需的参数
        strategy_params = {}
        if self.filter_type in ['sensitivity', 'density', 'heaviside_density']:
            strategy_params.update({
                'H': self._H,
                'cell_measure': self._cell_measure,
                'normalize_factor': self._normalize_factor,
            })
        if self.filter_type == 'heaviside_density':
            strategy_params.update({
                'beta': beta,
                'max_beta': max_beta,
                'continuation_iter': continuation_iter,
            })
        
        # 实例化策略
        self._strategy: _FilterStrategy = strategy_class(**strategy_params)

    # 3. 委托公共方法到具体策略
    def get_initial_density(self, x: TensorLike) -> TensorLike:
        """获取初始物理密度场"""
        return self._strategy.get_initial_density(x)

    def filter_variables(self, x: TensorLike) -> TensorLike:
        """对设计变量进行滤波得到物理变量"""
        return self._strategy.filter_variables(x)

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        """过滤目标函数的灵敏度"""
        return self._strategy.filter_objective_sensitivities(xPhys, dobj)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        """过滤约束函数的灵敏度"""
        return self._strategy.filter_constraint_sensitivities(xPhys, dcons)

    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """
        为支持 continuation 的策略 (如 Heaviside 密度过滤) 执行一步 beta continuation
            如果当前策略不支持，则不执行任何操作
        """
        return self._strategy.continuation_step(change)

    
# # 代理所有方法到内部的过滤器实例
# def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
#     """获取初始的物理密度场"""
#     if self._filter is None:
#         xPhys[:] = x
#         return xPhys
#     return self._filter.get_initial_density(x, xPhys)

# def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
#     """对设计变量进行滤波得到物理变量"""
#     if self._filter is None:
#         xPhys[:] = x
#         return xPhys
#     return self._filter.filter_variables(x, xPhys)

# def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
#     """过滤目标函数的灵敏度"""
#     if self._filter is None:
#         return dobj
#     return self._filter.filter_objective_sensitivities(xPhys, dobj)

# def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
#     """过滤约束函数的灵敏度"""
#     if self._filter is None:
#         return dcons
#     return self._filter.filter_constraint_sensitivities(xPhys, dcons)

# # Heaviside 密度过滤特有的方法
# def continuation_step(self, change: float) -> tuple[float, bool]:
#     """执行 beta continuation (仅对 Heaviside 密度过滤有效)"""
#     if hasattr(self._filter, 'continuation_step'):
#         return self._filter.continuation_step(change)
#     return change, False