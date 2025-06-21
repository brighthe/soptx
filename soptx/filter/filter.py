from abc import ABC, abstractmethod

from typing import Optional, Literal

from fealpy.mesh import Mesh
from fealpy.typing import TensorLike

from .basic_filter import (BasicFilter, 
                          SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)

class Filter:
    """统一的过滤方法接口类

    该类使用策略模式来动态选择和应用不同的滤波算法
    """
    def __init__(self,
                mesh: Mesh,
                rmin: float,
                filter_type: Literal['none', 'sensitivity', 'density', 'heaviside_denisty'],
                # Heaviside 投影相关参数
                beta: float = 1.0,
                max_beta: float = 512,
                continuation_iter: int = 50
            ) -> None:
        
        self.mesh = mesh
        self.rmin = rmin
        self.filter_type = filter_type
        
        # 1. 构建滤波矩阵
        if self.filter_type != 'none' and self.rmin > 0:
            builder = FilterMatrixBuilder(mesh, rmin)
            self._H, self._Hs = builder.build()
            self._cell_measure = self.mesh.entity_measure('cell')
            self._normalize_factor = self._H.matmul(self._cell_measure)
        else:
            # 如果不滤波，将这些设为 None
            self._H, self._Hs, self._cell_measure, self._normalize_factor = None, None, None, None

    
    # 代理所有方法到内部的过滤器实例
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
        """获取初始的物理密度场"""
        if self._filter is None:
            xPhys[:] = x
            return xPhys
        return self._filter.get_initial_density(x, xPhys)
    
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> TensorLike:
        """对设计变量进行滤波得到物理变量"""
        if self._filter is None:
            xPhys[:] = x
            return xPhys
        return self._filter.filter_variables(x, xPhys)
    
    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> TensorLike:
        """过滤目标函数的灵敏度"""
        if self._filter is None:
            return dobj
        return self._filter.filter_objective_sensitivities(xPhys, dobj)
    
    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> TensorLike:
        """过滤约束函数的灵敏度"""
        if self._filter is None:
            return dcons
        return self._filter.filter_constraint_sensitivities(xPhys, dcons)
    
    # Heaviside 特有的方法
    def continuation_step(self, change: float) -> tuple[float, bool]:
        """执行 beta continuation（仅对 Heaviside 过滤有效）"""
        if hasattr(self._filter, 'continuation_step'):
            return self._filter.continuation_step(change)
        return change, False
    
    # 属性访问
    @property
    def H(self):
        """滤波矩阵"""
        if self._filter is None:
            return None
        return self._filter.H
    
    @property
    def Hs(self):
        """滤波矩阵行和向量"""
        if self._filter is None:
            return None
        return self._filter.Hs
    
    @property
    def beta(self):
        """Heaviside beta 参数"""
        if hasattr(self._filter, 'beta'):
            return self._filter.beta
        return None
    
    @beta.setter
    def beta(self, value):
        """设置 Heaviside beta 参数"""
        if hasattr(self._filter, 'beta'):
            self._filter.beta = value


# 为了向后兼容，也可以提供一个工厂函数
def create_filter(filter_type: str, **kwargs) -> Optional[BasicFilter]:
    """创建过滤器的工厂函数（向后兼容）
    
    Parameters
    ----------
    filter_type : str
        过滤类型
    **kwargs
        传递给 Filter 类的其他参数
        
    Returns
    -------
    BasicFilter or None
        过滤器实例
    """
    filter_instance = Filter(filter_type, **kwargs)
    return filter_instance._filter