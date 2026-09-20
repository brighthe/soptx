from typing import Literal, Dict, Type, Tuple, Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import Function
from fealpy.typing import TensorLike

from soptx.core import BaseLogged
from soptx.topology.constraints.exemption import (
                                apply_exemption,
                                apply_passive_solid,
                                validate_exemption_mask,
                            )

from .matrix import FilterMatrixBuilder
from .strategies import (
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

    该类使用策略模式来动态选择和应用不同的过滤算法。filter_type 的四个取值
    不是四种并列的过滤器, 而是一条两级的正则化链:

    - 'none'        : 不做正则化 (恒等映射);
    - 'sensitivity' : 灵敏度过滤, 作用在梯度空间, 密度场本身不被平滑;
    - 'density'     : 线性密度过滤, 对密度场做加权平均 (卷积矩阵 H);
    - 'projection'  : 密度过滤 **加** Heaviside 投影 (含 beta 延拓), 即在
                      'density' 之上再叠一层非线性映射, 而不是它的替代品。

    这一层次关系在实现上由 ``ProjectionStrategy(DensityStrategy)`` 的继承
    表达; 扁平枚举只是配置层的门面, 读配置时不要把后两者理解成互斥选项。
    """
    def __init__(self,
                design_mesh: HomogeneousMesh,
                filter_type: Literal['none', 'sensitivity', 'density', 'projection'],
                rmin: Optional[float] = None,
                density_location: Optional[str] = None,
                disp_mesh: Optional[HomogeneousMesh] = None, 
                filter_q: int = 1,
                projection_params: Optional[Dict] = None,
                passive_mask: Optional[TensorLike] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._design_mesh = design_mesh
        self._filter_type = filter_type

        # 实体保留 (passive solid) 掩码: 该批单元的物理密度在过滤/投影之后被
        # 覆写为 1, 相应的密度灵敏度置零。施加点必须在过滤之后 —— 只钉设计
        # 变量时, 宽过滤下保留单元的物理密度仍由邻域决定, 达不到实体保留。
        self._passive_mask = validate_exemption_mask(
            passive_mask, design_mesh.number_of_cells()
        )

        self._rmin = rmin
        self._density_location = density_location
        self._filter_q = filter_q

        self._disp_mesh = disp_mesh

        if self._density_location == 'element_multiresolution' and self._disp_mesh is None:
            self._log_error(
                "当 density_location 为 'element_multiresolution' 时, disp_mesh 不能为 None。"
            )

        # 1. 构建过滤矩阵
        if self._filter_type != 'none' and self._rmin is not None and self._rmin > 0:
            # filter_q 只作用于非结构网格所走的 KD-tree 通用路径 (权重
            # (1 - d/rmin)^q); 均匀笛卡尔网格走结构化快路径, 权重恒为线性
            # 锥形, 该参数在那条路径上不起作用。
            builder = FilterMatrixBuilder(
                                    mesh=self._design_mesh, 
                                    rmin=self._rmin, 
                                    density_location=self._density_location,
                                    q=self._filter_q,
                                    enable_logging=enable_logging,
                                    logger_name=logger_name,
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
        
        if self._filter_type == 'projection' and projection_params:
            # 投影参数的默认值只在 ProjectionStrategy 的签名里维护一份。
            # 曾经这里另有一份门面默认值, 且无条件覆盖签名默认 (例如把
            # projection_type 从 'tanh' 静默换成 'exponential'), 调用方读签名
            # 会读到与实际不符的行为, 故这里只透传调用方显式给定的键。
            strategy_params.update(projection_params)
        
        # 实例化策略
        self._strategy: _FilterStrategy = strategy_class(**strategy_params)

    @property
    def design_mesh(self) -> HomogeneousMesh:
        """设计变量所在网格。

        优化器 (如 OC) 据此向问题类索取被动单元掩码
        (``pde.get_passive_element_mask(mesh=design_mesh)``); 问题类不定义
        该方法时优化器不设掩码, 现有无被动区的工况不受影响。
        """
        return self._design_mesh

    @property
    def passive_mask(self) -> Optional[TensorLike]:
        """实体保留单元的布尔掩码; 无保留区时为 None。

        消费端 (驱动层的产物汇总、冻结评价) 需要知道哪些单元的密度不是设计
        结果而是硬约束, 否则会把垫片的体积算成优化得到的体积。
        """
        return self._passive_mask

    def _enforce_passive_solid(self,
                        physical_density: Union[TensorLike, Function],
                    ) -> Union[TensorLike, Function]:
        """把实体保留单元的物理密度覆写为满密度。"""
        if self._passive_mask is None:
            return physical_density

        if isinstance(physical_density, Function):
            physical_density[:] = apply_passive_solid(
                physical_density[:], self._passive_mask
            )
            return physical_density

        return apply_passive_solid(physical_density, self._passive_mask)

    @property
    def has_projection(self) -> bool:
        """本过滤链是否含非线性投影。

        算法层 (如 MMA 的目标函数缩放) 需要知道这件事, 但不该去比 filter_type
        字符串: 那是配置层的门面。这里按策略对象的实际类型回答。
        """
        return isinstance(self._strategy, ProjectionStrategy)

    @property
    def beta(self) -> Optional[float]:
        """动态获取当前策略的 beta 值（如果存在）"""
        # 探测底层策略对象是否具有 beta 属性
        return getattr(self._strategy, 'beta', None)

    @property
    def beta_max(self) -> Optional[float]:
        """当前策略的 beta 上限; 无投影连续化时为 None。

        停止准则要求先判定连续化已终止再判定收敛 (否则把连续化中途的停滞
        记为收敛), 该判定需要上限而不只是当前值, 故与 beta 成对暴露。
        """
        return getattr(self._strategy, 'beta_max', None)

    # 3. 委托公共方法到具体策略
    def get_initial_density(self, 
                        density:  Union[TensorLike, Function], 
                    ) ->  Union[TensorLike, Function]:

        return self._enforce_passive_solid(
            self._strategy.get_initial_density(density=density)
        )

    def filter_design_variable(self,
                        design_variable: Union[TensorLike, Function], 
                        physical_density: Union[TensorLike, Function]
                    ) -> Union[TensorLike, Function]:

        return self._enforce_passive_solid(
            self._strategy.filter_design_variable(
                design_variable=design_variable, physical_density=physical_density
            )
        )

    def filter_objective_sensitivities(self, 
                                    design_variable: Union[TensorLike, Function], 
                                    obj_grad_rho: TensorLike
                                ) -> TensorLike:

        # 保留单元的 rho_phys 是常数, d rho_phys / d z = 0, 故先把这些行清零
        # 再走链式法则; 否则梯度里会留下一份并不存在的下降方向。
        obj_grad_rho = apply_exemption(obj_grad_rho, self._passive_mask, 0.0)

        return self._strategy.filter_objective_sensitivities(design_variable=design_variable, obj_grad_rho=obj_grad_rho)

    def filter_constraint_sensitivities(self, 
                                    design_variable: Union[TensorLike, Function], 
                                    con_grad_rho: TensorLike
                                ) -> TensorLike:

        con_grad_rho = apply_exemption(con_grad_rho, self._passive_mask, 0.0)

        return self._strategy.filter_constraint_sensitivities(design_variable=design_variable, con_grad_rho=con_grad_rho)

    def continuation_step(self, change: float) -> Tuple[float, bool]:

        return self._strategy.continuation_step(change)
