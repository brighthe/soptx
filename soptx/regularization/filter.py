from typing import Literal, Dict, Type, Tuple, Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import Function
from fealpy.typing import TensorLike

from ..utils.base_logged import BaseLogged

from .matrix_builder import FilterMatrixBuilder 
from .filter_strategy import (
                                _FilterStrategy,
                                NoneStrategy,
                                DensityStrategy,
                                SensitivityStrategy, 
                                ProjectionStrategy,
                            )

FILTER_STRATEGY_REGISTRY: Dict[str, Type[_FilterStrategy]] = \
                                {
                                    'none': NoneStrategy,
                                    'sensitivity': SensitivityStrategy,
                                    'density': DensityStrategy,
                                    'projection': ProjectionStrategy,
                                }

class Filter(BaseLogged):
    """统一的过滤方法接口类

    该类使用策略模式来动态选择和应用不同的过滤算法
    """
    def __init__(self,
                mesh: HomogeneousMesh,
                filter_type: Literal['none', 'sensitivity', 'density', 'projection'],
                rmin: Optional[float] = None,
                density_location: Optional[str] = None,
                projection_params: Optional[Dict] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._mesh = mesh
        self._filter_type = filter_type
        self._rmin = rmin
        self._density_location = density_location
        
        # 1. 构建过滤矩阵
        if self._filter_type != 'none' and self._rmin > 0:
            builder = FilterMatrixBuilder(
                                    mesh=mesh, 
                                    rmin=rmin, 
                                    density_location=density_location,
                                )
            self._H = builder.build()
            self._cell_measure = self._mesh.entity_measure('cell')

        else:
            self._H = None

        # 1. 构建过滤矩阵
        if self._filter_type != 'none' and self._rmin is not None and self._rmin > 0:
            builder = FilterMatrixBuilder(
                                    mesh=mesh, 
                                    rmin=rmin, 
                                    density_location=density_location,
                                )
            self._H = builder.build()
            self._cell_measure = self._mesh.entity_measure('cell')

        else:
            self._H = None
            if self._filter_type != 'none':
                error_msg = (f"过滤类型 '{self._filter_type}' 需要有效的过滤半径 rmin。"
                             f"当前 rmin={self._rmin}")
                self._log_error(error_msg)
                raise ValueError(error_msg)

        # 2. 策略选择和实例化
        strategy_class = FILTER_STRATEGY_REGISTRY.get(self._filter_type)
        if strategy_class is None:
            error_msg = (f"未知的过滤方法: '{self._filter_type}'. "
                        f"可用选项: {list(FILTER_STRATEGY_REGISTRY.keys())}")
            self._log_error(error_msg)

        strategy_params = {
                            'H': self._H,
                            'mesh': self._mesh,
                            'density_location': self._density_location,
                            'enable_logging': enable_logging, 
                            'logger_name': logger_name
                        }
        
        if self._filter_type == 'projection':
            proj_defaults = {
                            'projection_type': 'exponential',
                            'beta': 1.0,
                            'beta_max': 512.0,
                            'continuation_iter': 50,
                        }
            if projection_params:
                proj_defaults.update(projection_params)
            
            strategy_params.update(proj_defaults)
        
        # 实例化策略
        self._strategy: _FilterStrategy = strategy_class(**strategy_params)

    # 3. 委托公共方法到具体策略
    def get_initial_density(self, 
                        density:  Union[TensorLike, Function], 
                    ) ->  Union[TensorLike, Function]:

        return self._strategy.get_initial_density(density=density)

    def filter_design_variable(self,
                        design_variable: Union[TensorLike, Function], 
                        physical_density: Union[TensorLike, Function]
                    ) -> Union[TensorLike, Function]:

        return self._strategy.filter_design_variable(design_variable=design_variable, physical_density=physical_density)

    def filter_objective_sensitivities(self, 
                                    design_variable: Union[TensorLike, Function], 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:

        return self._strategy.filter_objective_sensitivities(design_variable=design_variable, obj_grad_rho=obj_grad_rho)

    def filter_constraint_sensitivities(self, 
                                    design_variable: Union[TensorLike, Function], 
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:

        return self._strategy.filter_constraint_sensitivities(design_variable=design_variable, con_grad_rho=con_grad_rho)

    def continuation_step(self, change: float) -> Tuple[float, bool]:

        return self._strategy.continuation_step(change)