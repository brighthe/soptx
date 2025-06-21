from typing import Optional, Literal, List, Union
from fealpy.mesh import StructuredMesh
from fealpy.typing import TensorLike

from .basic_filter import (BasicFilter, 
                          SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)

class Filter:
    """统一的过滤器接口类
    
    用户只需实例化这个类，通过参数选择不同的过滤方法。
    """
    
    def __init__(self, 
                 filter_type: Literal['none', 'sensitivity', 'density', 'heaviside'],
                 mesh: StructuredMesh,
                 rmin: float,
                 # Heaviside 投影相关参数
                 beta: float = 1.0,
                 max_beta: float = 512,
                 continuation_iter: int = 50,
                 # 高级参数
                 domain: Optional[List] = None,
                 method: Literal['auto', 'uniform', 'general'] = 'auto',
                 periodic: List[bool] = [False, False, False],
                 enable_timing: bool = False):
        """
        Parameters
        ----------
        filter_type : {'none', 'sensitivity', 'density', 'heaviside'}
            过滤方法类型：
            - 'none': 不使用过滤
            - 'sensitivity': 灵敏度过滤
            - 'density': 线性密度过滤
            - 'heaviside': Heaviside投影（非线性密度过滤）
        mesh : StructuredMesh
            网格对象
        rmin : float
            过滤半径（物理距离）
        beta : float, default=1.0
            Heaviside 投影参数（仅在 filter_type='heaviside' 时有效）
        max_beta : float, default=512
            Heaviside 投影的最大 beta 值
        continuation_iter : int, default=50
            Heaviside 投影的 continuation 迭代间隔
        domain : List, optional
            计算域边界，如果不提供会自动计算
        method : {'auto', 'uniform', 'general'}, default='auto'
            过滤矩阵计算方法
        periodic : List[bool], default=[False, False, False]
            各方向周期性边界条件
        enable_timing : bool, default=False
            是否启用计时功能
        """
        self.filter_type = filter_type.lower()
        
        # 创建具体的过滤器实例
        if self.filter_type == 'none':
            self._filter = None
        elif self.filter_type == 'sensitivity':
            self._filter = SensitivityBasicFilter(
                mesh=mesh, 
                rmin=rmin, 
                domain=domain,
                method=method,
                periodic=periodic,
                enable_timing=enable_timing
            )
        elif self.filter_type == 'density':
            self._filter = DensityBasicFilter(
                mesh=mesh, 
                rmin=rmin, 
                domain=domain,
                method=method,
                periodic=periodic,
                enable_timing=enable_timing
            )
        elif self.filter_type == 'heaviside':
            self._filter = HeavisideProjectionBasicFilter(
                mesh=mesh, 
                rmin=rmin, 
                domain=domain,
                beta=beta,
                max_beta=max_beta,
                continuation_iter=continuation_iter,
                method=method,
                periodic=periodic,
                enable_timing=enable_timing
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
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