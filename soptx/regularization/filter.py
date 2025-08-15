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
                density_location: Optional[str] = None,
                integration_order: Optional[int] = None,
                interpolation_order: Optional[int] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._mesh = mesh
        self._filter_type = filter_type
        self._rmin = rmin
        self._density_location = density_location
        self._integration_order = integration_order
        self._interpolation_order = interpolation_order

        # 参数验证
        if self._filter_type != 'none' and (self._rmin is None or self._rmin <= 0):
            error_msg = (f"当 filter_type='{self._filter_type}' 时，必须提供有效的 rmin (> 0). "
                        f"当前 rmin={self._rmin}")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        # 验证密度位置参数
        if self._filter_type != 'none' and self._density_location is None:
            error_msg = f"当 filter_type='{self._filter_type}' 时，必须提供 density_location 参数"
            self._log_error(error_msg)
            raise ValueError(error_msg)
            
        # 验证积分/插值参数
        if (self._filter_type != 'none' and 
            self._density_location == 'gauss_integration_point' and 
            self._integration_order is None):
            error_msg = "当 density_location='gauss_integration_point' 时，必须提供 integrator_order 参数"
            self._log_error(error_msg)
            raise ValueError(error_msg)
            
        if (self._filter_type != 'none' and 
            self._density_location == 'interpolation_point' and 
            self._interpolation_order is None):
            error_msg = "当 density_location='interpolation_point' 时，必须提供 interpolation_order 参数"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        
        # 1. 构建过滤矩阵
        if self._filter_type != 'none' and self._rmin > 0:
            builder = FilterMatrixBuilder(
                mesh=mesh, 
                rmin=rmin, 
                density_location=density_location,
                integration_order=integration_order,
                interpolation_order=interpolation_order
            )
            self._H, self._Hs = builder.build()
            self._integration_weights = self._compute_integration_weights()
            self._cell_measure = self._mesh.entity_measure('cell')
        else:
            self._H, self._Hs, self._integration_weights = None, None, None

        # 2. 策略选择和实例化
        strategy_class = FILTER_STRATEGY_REGISTRY.get(self._filter_type)
        if strategy_class is None:
            error_msg = (f"未知的过滤方法: '{self._filter_type}'. "
                        f"可用选项: {list(FILTER_STRATEGY_REGISTRY.keys())}")
            self._log_error(error_msg)
            raise ValueError(error_msg)

        strategy_params = {}
        
        if self._filter_type in ['sensitivity']:
            strategy_params.update({
                'H': self._H,
                'Hs': self._Hs,
                'integration_weights': self._integration_weights,
                'density_location': self._density_location,
                'mesh': self._mesh,
                'integration_order': self._integration_order,
            })

        if self._filter_type == 'density':
            strategy_params.update({
                'H': self._H,
                'integration_weights': self._integration_weights,
                'density_location': self._density_location,
                'mesh': self._mesh,
                'integration_order': self._integration_order,
            })

        if self._filter_type == 'heaviside_density':
            strategy_params.update({
                'beta': 1.0,
                'max_beta': 512.0,
                'continuation_iter': 50,
            })
        
        # 实例化策略
        self._strategy: _FilterStrategy = strategy_class(**strategy_params)


    ###########################################################################################################
    # 属性访问器
    ###########################################################################################################

    @property
    def mesh(self) -> HomogeneousMesh:
        """获取网格对象"""
        return self._mesh
    

    ###########################################################################################################
    # 核心方法
    ###########################################################################################################

    # 3. 委托公共方法到具体策略
    def get_initial_density(self, rho: Function, rho_Phys: Function) -> Function:
        """获取初始物理密度场"""

        return self._strategy.get_initial_density(rho=rho, rho_Phys=rho_Phys)

    def filter_variables(self, rho: Function, rho_Phys: Function) -> Function:
        """对设计变量进行滤波得到物理变量"""

        return self._strategy.filter_variables(rho=rho, rho_Phys=rho_Phys)

    def filter_objective_sensitivities(self, rho_Phys: Union[TensorLike, Function], obj_grad: TensorLike) -> TensorLike:
        """过滤目标函数的灵敏度"""
        
        return self._strategy.filter_objective_sensitivities(rho_Phys=rho_Phys, obj_grad=obj_grad)

    def filter_constraint_sensitivities(self, rho_Phys: Function, con_grad: TensorLike) -> TensorLike:
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
        

    ###########################################################################################################
    # 内部方法
    ###########################################################################################################
        
    def _compute_integration_weights(self) -> TensorLike:
        """根据密度位置类型计算积分权重"""

        if self._density_location == 'element':
            
            integration_weights = self._mesh.entity_measure('cell')

            return integration_weights
            
        elif self._density_location == 'gauss_integration_point':
            
            qf = self._mesh.quadrature_formula(q=self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()
            
            J = self._mesh.jacobi_matrix(bcs)
            detJ = bm.linalg.det(J)

            integration_weights = bm.einsum('q, cq -> cq', ws, detJ)

            return integration_weights
            
        elif self._density_location == 'density_subelement_gauss_point':

            # 获取单元测度
            cell_measure = self._mesh.entity_measure('cell')  # (NC,)
            
            # 获取子单元数量（等于高斯点数量）
            qf = self._mesh.quadrature_formula(q=self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()
            NQ = ws.shape[0]
            
            # 每个子单元的测度 = 单元测度 / 子单元数量
            NC = self._mesh.number_of_cells()
            subcell_measure = cell_measure[:, None] / NQ
            integration_weights = bm.broadcast_to(subcell_measure, (NC, NQ))

            return integration_weights
            
        else:
            raise ValueError(f"不支持的密度位置类型: {self._density_location}")