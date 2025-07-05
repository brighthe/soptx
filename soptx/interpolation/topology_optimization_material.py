from typing import Optional, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike
from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace, Function

from .linear_elastic_material import LinearElasticMaterial
from .interpolation_scheme import MaterialInterpolationScheme
from ..utils.base_logged import BaseLogged

class TopologyOptimizationMaterial(BaseLogged):
    def __init__(self, 
                mesh: Mesh,
                base_material: LinearElasticMaterial,
                interpolation_scheme: MaterialInterpolationScheme,
                relative_density: float = 1,
                density_location: str = 'element',
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        1. 实例化时设置默认密度分布变体方法
        tom = TopologyOptimizationMaterial(density_location='element')
        2. 直接使用默认方法生成单元密度分布
        rrho = tom.setup_density_distribution()  # 获取单元密度分布
        3. 切换到其他密度分布方法
        tom.setup_density_distribution.set('element_gauss_integrate_point')     # 设置变体 (返回 None)
        rrho = tom.setup_density_distribution(quadrature_order=3)
        注意: 
        - setup_density_distribution.set() 只设置变体，不执行方法，返回 None
        - 需要分别调用 set() 和 setup_density_distribution() 来生成获取弹性矩阵
        - 每次 set() 后，后续的 setup_density_distribution() 调用都使用新设置的变体
        """
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.mesh = mesh
        self.base_material = base_material
        self.interpolation_scheme = interpolation_scheme

        self._relative_density = relative_density

        self._setup_function_spaces()

        self.setup_density_distribution.set(density_location)
        self._density_distribution = self.setup_density_distribution()

        self._log_info(f"Topology optimization material initialized: "
                       f"relative_density={relative_density}, "
                       f"density_location={density_location}, "
                       f"distribution_shape={self._density_distribution.shape}")
        
    def _setup_function_spaces(self):
        """设置函数空间"""
        self.element_space = LagrangeFESpace(self.mesh, p=0, ctype='D')

    @property
    def relative_density(self) -> float:
        """获取当前的相对密度值"""
        return self._relative_density
    
    @property
    def density_location(self) -> str:
        """获取当前的密度定义位置"""
        return self.setup_density_distribution.vm.get_key(self)
    
    @property
    def penalty_factor(self) -> float:
        """获取当前的惩罚因子 (代理属性)"""

    @property
    def interpolation_method(self) -> str:
        """获取当前的插值方法 (代理属性)"""
        return self.interpolation_scheme.interpolation_method
        
    @property
    def density_distribution(self) -> TensorLike:
        """获取当前的密度分布"""
        return self._density_distribution
    
    @variantmethod('element')
    def setup_density_distribution(self, **kwargs) -> Function:
        """设置单元密度分布 (NC, )"""
        NC = self.mesh.number_of_cells()
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=self.mesh.device
                            )

        density_dist = self.element_space.function(density_tensor)

        self._log_info(f"Created element density distribution: ({NC},) "
                       f"with value {self._relative_density}")

        return density_dist
    
    @setup_density_distribution.register('element_gauss_integrate_point')
    def setup_density_distribution(self, **kwargs) -> Function:
        """设置单元高斯积分点密度分布 (NC, NQ)"""
        quadrature_order = kwargs.get('quadrature_order', 3)
        qf = self.mesh.quadrature_formula(quadrature_order)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = self.mesh.number_of_cells()
        NQ = len(ws)
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=self.mesh.device
                            )
        density_dist = self.element_space.function(density_tensor)
        density_dist = density_dist(bcs)

        self._log_info(f"Created Gauss point density distribution: ({NC}, {NQ}) "
                f"with value {self._relative_density}, q={quadrature_order}")

        return density_dist
    
    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """计算插值后的弹性矩阵"""
        if self.relative_density is None:
            error_msg = "No relative density set. Please call set_relative_density() first."
            self._log_error(error_msg)
            raise ValueError(error_msg)

        D = self.interpolation_scheme.interpolate(self.base_material, self._density_distribution)

        self._log_info(f"[TopologyOptimizationMaterial] Elastic matrix computed successfully, "
                   f"shape: {D.shape}")
        
        return D

    def set_relative_density(self, relative_density: float) -> None:
        """设置相对密度值"""
        if relative_density < 0.0 or relative_density > 1.0:
            error_msg = f"Relative density must be in [0, 1] range, got {relative_density:.3f}"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        old_relative_density = self._relative_density
        self._relative_density = relative_density

        self._log_info(f"Relative density updated from {old_relative_density:.3f} "
                       f"to {self._relative_density:.3f}")

    def set_material_parameters(self, **kwargs) -> None:
        """设置基础材料参数（代理方法）"""
        self.base_material.set_material_parameters(**kwargs)
        self._log_info(f"[TopologyOptimizationMaterial] Material parameters updated via proxy method")

    def set_penalty_factor(self, penalty_factor: float) -> None:
        """设置惩罚因子 (代理方法)"""
        self.interpolation_scheme.set_penalty_factor(penalty_factor)
        self._log_info(f"[TopologyOptimizationMaterial] Penalty factor updated via proxy method")

    def get_material_info(self) -> Dict[str, Any]:
        """获取材料信息, 包括基础材料和插值方案的参数"""
        base_material_info = self.base_material.get_material_params()

        interpolation_info = self.interpolation_scheme.get_interpolation_params()

        material_info = {
            **base_material_info,
            **interpolation_info
        }

        return material_info

    def display_material_info(self) -> None:
        """显示材料信息"""
        info = self.get_material_info()

        self._log_info(f"Topology optimization material info: {info}", force_log=True)
        