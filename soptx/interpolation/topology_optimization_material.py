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
                quadrature_order: Optional[int] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        密度分布变体方法示例:
        -----------------------------------
        1. 初始化时完整设置
        topm = TopologyOptimizationMaterial(
                            mesh=mesh, base_material=ilem, interpolation_scheme=simpi,
                            relative_density=0.5,
                            density_location='element_gauss_integrate_point',
                            quadrature_order=3)

        topm.set_quadrature_order(2)  # 直接更新

        2. 使用新的统一接口
        topm2 = TopologyOptimizationMaterial(
                            mesh=mesh, base_material=ilem, interpolation_scheme=simpi,
                            density_location='element')
        
        topm2.set_density_location('element_gauss_integrate_point', quadrature_order=3)

        3. 分步设置
        topm3 = TopologyOptimizationMaterial(
                            mesh=mesh, base_material=ilem, interpolation_scheme=simpi,
                            density_location='element')

        topm3.set_quadrature_order(3)  # 先设置参数（有警告但允许）
        topm3.set_density_location('element_gauss_integrate_point')  # 切换位置，自动更新
        """
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        if density_location == 'element_gauss_integrate_point' and quadrature_order is None:
            error_msg = ("quadrature_order is required when density_location='element_gauss_integrate_point'. "
                        "Please provide quadrature_order parameter.")
            self._log_error(error_msg)
            raise ValueError(error_msg)

        self.mesh = mesh
        self._base_material = base_material
        self._interpolation_scheme = interpolation_scheme

        self._relative_density = relative_density
        self._quadrature_order = quadrature_order

        self._setup_function_spaces()

        self.setup_density_distribution.set(density_location)
        self._density_distribution = self.setup_density_distribution()

        self._log_info(f"Topology optimization material initialized: "
                       f"relative_density={relative_density}, "
                       f"density_location={density_location}, "
                       f"quadrature_order={quadrature_order}, "
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
    def quadrature_order(self) -> int:
        """获取当前的高斯积分次数"""
        return self._quadrature_order
    
    @property
    def penalty_factor(self) -> float:
        """获取当前的惩罚因子 (代理属性)"""
        return self._interpolation_scheme.penalty_factor

    @property
    def interpolation_method(self) -> str:
        """获取当前的插值方法 (代理属性)"""
        return self._interpolation_scheme.interpolation_method
        
    @property
    def density_distribution(self) -> Function:
        """获取当前的密度分布"""
        return self._density_distribution
    
    @density_distribution.setter
    def density_distribution(self, new_density_distribution: Function) -> None:
        """设置新的密度分布"""
        if new_density_distribution.shape != self._density_distribution.shape:
            error_msg = f"Shape mismatch: expected {self._density_distribution.shape}, got {new_density_distribution.shape}"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        self._density_distribution[:] = new_density_distribution

        self._log_info(f"Density distribution updated, shape: {new_density_distribution.shape}")

        return new_density_distribution

    @variantmethod('element')
    def setup_density_distribution(self, **kwargs) -> Function:
        """初始化单元密度分布 (NC, )"""
        NC = self.mesh.number_of_cells()
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=self.mesh.device
                            )

        density_dist = self.element_space.function(density_tensor)

        self._density_distribution = density_dist

        self._log_info(f"Initialized element density distribution: ({NC},) "
                       f"with value {self._relative_density}")

        return density_dist
    
    @setup_density_distribution.register('element_gauss_integrate_point')
    def setup_density_distribution(self, **kwargs) -> Function:
        """初始化单元高斯积分点密度分布 (NC, NQ)"""
        if self._quadrature_order is None:
            error_msg = ("Quadrature order not set for 'element_gauss_integrate_point' density location. "
                        "Please call set_quadrature_order() first.")
            self._log_error(error_msg)
            raise ValueError(error_msg)
    

        qf = self.mesh.quadrature_formula(self._quadrature_order)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = self.mesh.number_of_cells()
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=self.mesh.device
                            )
        density_dist = self.element_space.function(density_tensor)
        density_dist = density_dist(bcs)

        self._density_distribution = density_dist

        self._log_info(f"Initialized element Gauss point density distribution: {density_dist.shape} "
                f"with value {self._relative_density}, q={self._quadrature_order}")

        return density_dist
    
    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """计算插值后的弹性矩阵"""

        D = self._interpolation_scheme.interpolate(self._base_material, self._density_distribution)

        self._log_info(f"[TopologyOptimizationMaterial] Elastic matrix computed successfully, "
                   f"shape: {D.shape}")
        
        return D

    def set_relative_density(self, relative_density: float) -> None:
        """设置相对密度值并自动更新密度分布"""
        if relative_density < 0.0 or relative_density > 1.0:
            error_msg = f"Relative density must be in [0, 1] range, got {relative_density:.3f}"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        old_relative_density = self._relative_density

        self._relative_density = relative_density
        self._density_distribution = self.setup_density_distribution()

        self._log_info(f"Relative density updated from {old_relative_density:.3f} "
                       f"to {self._relative_density:.3f}, density distribution recalculated")
        
    def set_quadrature_order(self, quadrature_order: int) -> None:
        """设置高斯积分次数并自动更新密度分布"""
        if self.density_location != 'element_gauss_integrate_point':
            self._log_warning(f"Quadrature order only affects 'element_gauss_integrate_point' density location, "
                            f"current location is '{self.density_location}'.")
            self._quadrature_order = quadrature_order
            return
    
        old_order = self._quadrature_order
        old_shape = self._density_distribution.shape
        
        self._quadrature_order = quadrature_order
        self._density_distribution = self.setup_density_distribution()
        
        new_shape = self._density_distribution.shape
        self._log_info(f"Quadrature order updated from {old_order} to {quadrature_order}, "
                       f"density distribution shape: {old_shape} -> {new_shape}")
        
    def set_density_location(self, density_location: str, quadrature_order: Optional[int] = None) -> None:
        """设置密度分布位置并自动更新密度分布"""
        if density_location == 'element_gauss_integrate_point':
            if quadrature_order is None and self._quadrature_order is None:
                error_msg = ("quadrature_order is required when switching to 'element_gauss_integrate_point'. "
                            "Please provide quadrature_order parameter.")
                self._log_error(error_msg)
                raise ValueError(error_msg)
            
            if quadrature_order is not None:
                self._quadrature_order = quadrature_order

        old_location = self.density_location
        old_shape = self._density_distribution.shape
        
        self.setup_density_distribution.set(density_location)
        self._density_distribution = self.setup_density_distribution()
        
        new_shape = self._density_distribution.shape
        self._log_info(f"Density location changed from '{old_location}' to '{density_location}', "
                    f"density distribution shape: {old_shape} -> {new_shape}")
        
    def set_interpolation_method(self, interpolation_method: str) -> None:
        """设置材料插值方法 (代理方法)"""
        self._interpolation_scheme.set_interpolation_method(interpolation_method)
        self._log_info(f"[TopologyOptimizationMaterial] Interpolation method updated via proxy method")

    def set_material_parameters(self, **kwargs) -> None:
        """设置基础材料参数（代理方法）"""
        self.base_material.set_material_parameters(**kwargs)
        self._log_info(f"[TopologyOptimizationMaterial] Material parameters updated via proxy method")

    def set_penalty_factor(self, penalty_factor: float) -> None:
        """设置惩罚因子 (代理方法)"""
        self._interpolation_scheme.set_penalty_factor(penalty_factor)
        self._log_info(f"[TopologyOptimizationMaterial] Penalty factor updated via proxy method")

    def get_material_info(self) -> Dict[str, Any]:
        """获取材料信息, 包括基础材料和插值方案的参数"""
        base_material_info = self.base_material.get_material_params()

        interpolation_info = self._interpolation_scheme.get_interpolation_params()

        topology_info = {
                        'relative_density': self.relative_density,
                        'density_location': self.density_location,
                        'quadrature_order': self.quadrature_order,
                        'density_distribution_shape': self.density_distribution.shape
                    }
        
        material_info = {
            **base_material_info,
            **interpolation_info,
            **topology_info
        }

        return material_info

    def display_material_info(self) -> None:
        """显示材料信息"""
        info = self.get_material_info()

        self._log_info(f"Topology optimization material info: {info}", force_log=True)
        