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
                design_mesh: HomogeneousMesh,
                filter_type: Literal['none', 'sensitivity', 'density', 'projection'],
                rmin: Optional[float] = None,
                density_location: Optional[str] = None,
                filter_exponent: int = 1, 
                disp_mesh: Optional[HomogeneousMesh] = None, 
                projection_params: Optional[Dict] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:
        """
        Parameters
        ----------
        design_mesh : HomogeneousMesh
            设计变量网格
        filter_type : {'none', 'sensitivity', 'density', 'projection'}
            过滤方法类型
        rmin : float, optional
            过滤半径 (物理长度尺度), 必须为正数.
            当 filter_type != 'none' 时必须提供.
        density_location : str, optional
            密度变量的位置, 可选 'element', 'element_multiresolution', 'node'
        filter_exponent : int, optional
            过滤权重的衰减速率指数, 默认为 1 (线性过滤).
            控制过滤权重随距离的衰减速率:
                - q=1: 线性衰减, w = max(0, 1 - d/rmin), 过滤效果较平滑
                - q>1: 加速衰减, w = (1 - d/rmin)^q, 过滤效果更集中于邻近单元
            仅对非均匀网格的通用过滤方法生效,
            2D/3D 均匀网格的专用方法暂不支持该参数.
        disp_mesh : HomogeneousMesh, optional
            位移网格, 当 density_location 为 'element_multiresolution' 时必须提供
        projection_params : dict, optional
            投影过滤的参数, 仅当 filter_type='projection' 时生效
        enable_logging : bool, optional
            是否启用日志, 默认为 True
        logger_name : str, optional
            日志记录器名称
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._design_mesh = design_mesh
        self._filter_type = filter_type

        self._rmin = rmin
        self._density_location = density_location
        self._filter_exponent = filter_exponent

        self._disp_mesh = disp_mesh

        if self._density_location == 'element_multiresolution' and self._disp_mesh is None:
            self._log_error(
                "当 density_location 为 'element_multiresolution' 时, disp_mesh 不能为 None。"
            )

        # 1. 构建过滤矩阵
        if self._filter_type != 'none' and self._rmin is not None and self._rmin > 0:
            builder = FilterMatrixBuilder(
                                    mesh=self._design_mesh, 
                                    rmin=self._rmin, 
                                    density_location=self._density_location,
                                    filter_exponent=self._filter_exponent,
                                )
            self._H = builder.build()
            self._cell_measure = self._design_mesh.entity_measure('cell')

        else:
            self._H = None
            if self._filter_type != 'none':
                error_msg = (f"过滤类型 '{self._filter_type}' 需要有效的过滤半径 rmin。"
                             f"当前 rmin={self._rmin}")
                self._log_error(error_msg)

        # 2. 策略选择和实例化
        strategy_class = FILTER_STRATEGY_REGISTRY.get(self._filter_type)
        if strategy_class is None:
            error_msg = (f"未知的过滤方法: '{self._filter_type}'. "
                        f"可用选项: {list(FILTER_STRATEGY_REGISTRY.keys())}")
            self._log_error(error_msg)

        strategy_params = {
                            'H': self._H,
                            'design_mesh': self._design_mesh,
                            'density_location': self._density_location,
                            'disp_mesh': self._disp_mesh, 
                            'enable_logging': enable_logging, 
                            'logger_name': logger_name
                        }
        
        if self._filter_type == 'projection':
            proj_defaults = {
                'projection_type'       : 'exponential',
                'beta'                  : 1.0,
                'eta'                   : 0.5,
                'beta_max'              : 512.0,
                'continuation_strategy' : 'multiplicative',
                'continuation_iter'     : 50,
                'beta_increment'        : 1.0,
                'beta_multiplier'       : 2.0,
            }
            if projection_params:
                proj_defaults.update(projection_params)
            
            strategy_params.update(proj_defaults)
        
        # 实例化策略
        self._strategy: _FilterStrategy = strategy_class(**strategy_params)

    @property
    def beta(self) -> Optional[float]:
        """动态获取当前策略的 beta 值（如果存在）"""
        # 探测底层策略对象是否具有 beta 属性
        return getattr(self._strategy, 'beta', None)

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