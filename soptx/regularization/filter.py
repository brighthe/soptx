from typing import Literal, Dict, Type, Tuple, Optional

from fealpy.mesh import HomogeneousMesh
from fealpy.typing import TensorLike

from ..utils.base_logged import BaseLogged

from .matrix_builder import FilterMatrixBuilder 
from .filter_strategy import (
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

class Filter(BaseLogged):
    """统一的过滤方法接口类

    该类使用策略模式来动态选择和应用不同的过滤算法
    """
    def __init__(self,
                mesh: HomogeneousMesh,
                filter_type: Literal['none', 'sensitivity', 'density', 'heaviside_density'],
                rmin: Optional[float] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self.mesh = mesh
        self.filter_type = filter_type
        self.rmin = rmin

        if self.filter_type != 'none' and (self.rmin is None or self.rmin <= 0):
            error_msg = (f"当 filter_type='{self.filter_type}' 时，必须提供有效的 rmin (> 0). "
                        f"当前 rmin={self.rmin}")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        # 1. 构建过滤矩阵
        if self.filter_type != 'none' and self.rmin > 0:
            builder = FilterMatrixBuilder(mesh, rmin)
            self._H, self._Hs = builder.build()
            self._cell_measure = self.mesh.entity_measure('cell')
        else:
            self._H, self._Hs, self._cell_measure = None, None, None

        # 2. 策略选择和实例化
        strategy_class = FILTER_STRATEGY_REGISTRY.get(self.filter_type)
        if strategy_class is None:
            error_msg = (f"未知的过滤方法: '{self.filter_type}'. "
                        f"可用选项: {list(FILTER_STRATEGY_REGISTRY.keys())}")
            self._log_error(error_msg)
            raise ValueError(error_msg)

        strategy_params = {}
        if self.filter_type in ['sensitivity', 'density', 'heaviside_density']:
            strategy_params.update({
                'H': self._H,
                'Hs': self._Hs,
                'cell_measure': self._cell_measure,
            })
        if self.filter_type == 'heaviside_density':
            strategy_params.update({
                'beta': 1.0,
                'max_beta': 512.0,
                'continuation_iter': 50,
            })
        
        # 实例化策略
        self._strategy: _FilterStrategy = strategy_class(**strategy_params)

    # 3. 委托公共方法到具体策略
    def get_initial_density(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        """获取初始物理密度场"""

        return self._strategy.get_initial_density(rho=rho, rho_Phys=rho_Phys)

    def filter_variables(self, rho: TensorLike, rho_Phys: TensorLike) -> TensorLike:
        """对设计变量进行滤波得到物理变量"""

        return self._strategy.filter_variables(rho=rho, rho_Phys=rho_Phys)

    def filter_objective_sensitivities(self, rho_Phys: TensorLike, obj_grad: TensorLike) -> TensorLike:
        """过滤目标函数的灵敏度"""

        return self._strategy.filter_objective_sensitivities(rho_Phys, obj_grad)

    def filter_constraint_sensitivities(self, rho_Phys: TensorLike, con_grad: TensorLike) -> TensorLike:
        """过滤约束函数的灵敏度"""

        return self._strategy.filter_constraint_sensitivities(rho_Phys, con_grad)

    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """
        为支持 continuation 的策略 (如 Heaviside 密度过滤) 执行一步 beta continuation
            如果当前策略不支持，则不执行任何操作
        """

        if hasattr(self._strategy, 'continuation_step'):
            return self._strategy.continuation_step(change)
        else:
            return change, False