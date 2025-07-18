from typing import Optional, Dict, Any, Literal

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, Function

from .linear_elastic_material import LinearElasticMaterial
from .interpolation_scheme import MaterialInterpolationScheme
from ..utils.base_logged import BaseLogged

class TopologyOptimizationMaterial(BaseLogged):
    def __init__(self, 
                base_material: LinearElasticMaterial,
                interpolation_scheme: MaterialInterpolationScheme,
                relative_density: float = 1,
                density_location: Literal['element', 'element_gauss_integrate_point'] = 'element',
                quadrature_order: Optional[int] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        拓扑优化材料类
        
        使用示例:
        -----------------------------------
        # 1. 创建材料对象
        topm = TopologyOptimizationMaterial(
                            base_material=ilem,
                            interpolation_scheme=simpi,
                            relative_density=0.5,
                            density_location='element'
                        )
        
        # 2. 初始化密度分布
        topm.setup_density_distribution(mesh)
        
        # 3. 网格更新时重新初始化
        mesh.uniform_refine()
        topm.setup_density_distribution(mesh)
        """
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        if density_location == 'element_gauss_integrate_point' and quadrature_order is None:
            error_msg = ("quadrature_order is required when density_location='element_gauss_integrate_point'. "
                        "Please provide quadrature_order parameter.")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        if density_location == 'element' and quadrature_order is not None:
            warning_msg = ("quadrature_order is provided but not needed when density_location='element'. ")
            self._log_warning(warning_msg, force_log=True)

        # 私有属性 (不建议外部直接访问)
        self._base_material = base_material
        self._interpolation_scheme = interpolation_scheme

        self._relative_density = relative_density
        self._density_location = density_location
        self._quadrature_order = quadrature_order

        self._density_distribution = None

        self.setup_density_distribution.set(density_location)

        self._log_info(f"Topology optimization material initialized: "
                       f"relative_density={relative_density}, "
                       f"density_location={density_location}, "
                       f"quadrature_order={quadrature_order}, ")


    #######################################################################################################################
    # 访问器
    #######################################################################################################################

    @property
    def base_material(self) -> LinearElasticMaterial:
        """获取当前的基础材料"""
        return self._base_material
    
    @property
    def interpolation_scheme(self) -> MaterialInterpolationScheme:
        """获取当前的材料插值方案"""
        return self._interpolation_scheme

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
    def density_distribution(self) -> Function:
        """获取当前的密度分布"""
        if self._density_distribution is None:
            self._log_warning("Density distribution not initialized. " \
                "Call setup_density_distribution(mesh) first.")
            
            return self._density_distribution
    
    
    #######################################################################################################################
    # 修改器
    #######################################################################################################################
    
    @density_distribution.setter
    def density_distribution(self, new_density_distribution: Function) -> None:
        """设置新的密度分布"""
        self._density_distribution = new_density_distribution

        self._log_info(f"Density distribution updated, shape: {new_density_distribution.shape}")


    #########################################################################################################################
    # 变体方法
    #########################################################################################################################

    @variantmethod('element')
    def setup_density_distribution(self, mesh: HomogeneousMesh, **kwargs) -> None:
        """初始化单元密度分布 (NC, )"""
        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=mesh.device
                                )
        
        element_space = LagrangeFESpace(mesh, p=0, ctype='D')

        density_dist = element_space.function(density_tensor)

        self.density_distribution = density_dist

        self._log_info(f"Initialized element density distribution: ({NC},) "
                       f"with value {self._relative_density}")
    
    @setup_density_distribution.register('element_gauss_integrate_point')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh, 
                                bcs, 
                                **kwargs
                            ) -> None:
        """设置单元高斯积分点密度分布 (NC, NQ)"""

        if self._quadrature_order is None:
            error_msg = ("Quadrature order not set for 'element_gauss_integrate_point' density location. "
                        "Please call set_quadrature_order() first.")
            self._log_error(error_msg)
            raise ValueError(error_msg)

        # qf = mesh.quadrature_formula(self._quadrature_order)
        # bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=mesh.device
                            )

        element_space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = element_space.function(density_tensor)
        density_dist = density_dist(bcs)

        self.density_distribution = density_dist

        self._log_info(f"Initialized element Gauss point density distribution: {density_dist.shape} "
                f"with value {self._relative_density}, q={self._quadrature_order}")
        
    def setup_element_gauss_intgrator_point_density_distribution(self, 
                                    mesh: HomogeneousMesh, **kwargs) -> None:
        """初始化单元高斯积分点密度分布 (NC, NQ)"""
        if self._quadrature_order is None:
            error_msg = ("Quadrature order not set for 'element_gauss_integrate_point' density location. "
                        "Please call set_quadrature_order() first.")
            self._log_error(error_msg)
            raise ValueError(error_msg)

        qf = mesh.quadrature_formula(self._quadrature_order)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), 
                                self._relative_density, 
                                dtype=bm.float64, 
                                device=mesh.device
                            )

        element_space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = element_space.function(density_tensor)
        density_dist = density_dist(bcs)

        self.density_distribution = density_dist

        self._log_info(f"Initialized element Gauss point density distribution: {density_dist.shape} "
                f"with value {self._relative_density}, q={self._quadrature_order}")
    
    #########################################################################################################################
    # 核心方法
    #########################################################################################################################
    
    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """计算插值后的弹性矩阵"""
        if self._density_distribution is None:
            error_msg = ("Density distribution not initialized. "
                        "Call setup_density_distribution(mesh) first.")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        if self._density_location == 'element_gauss_integrate_point':
            if self._quadrature_order is None:
                error_msg = ("Quadrature order not set for 'element_gauss_integrate_point' density location. "
                            "Please call set_quadrature_order() first.")
                self._log_error(error_msg)
                raise ValueError(error_msg)
        
        denisty_distribution = self.setup_density_distribution(mesh=)
                        
        
        D = self._interpolation_scheme.interpolate(self._base_material, self._density_distribution[:])

        self._log_info(f"[TopologyOptimizationMaterial] Elastic matrix computed successfully, "
                   f"shape: {D.shape}")
        
        return D
    
    #########################################################################################################################
    # 辅助方法
    #########################################################################################################################

    # def set_relative_density(self, relative_density: float) -> None:
    #     """设置相对密度值并自动更新密度分布"""
    #     if relative_density < 0.0 or relative_density > 1.0:
    #         error_msg = f"Relative density must be in [0, 1] range, got {relative_density:.3f}"
    #         self._log_error(error_msg)
    #         raise ValueError(error_msg)

    #     old_relative_density = self._relative_density

    #     self._relative_density = relative_density
    #     self._density_distribution = self.setup_density_distribution()

    #     self._log_info(f"Relative density updated from {old_relative_density:.3f} "
    #                    f"to {self._relative_density:.3f}, density distribution recalculated")
        
    # def set_quadrature_order(self, quadrature_order: int) -> None:
    #     """设置高斯积分次数并自动更新密度分布"""
    #     if self.density_location != 'element_gauss_integrate_point':
    #         self._log_warning(f"Quadrature order only affects 'element_gauss_integrate_point' density location, "
    #                         f"current location is '{self.density_location}'.")
    #         self._quadrature_order = quadrature_order
    #         return
    
    #     old_order = self._quadrature_order
    #     old_shape = self._density_distribution.shape
        
    #     self._quadrature_order = quadrature_order
    #     self._density_distribution = self.setup_density_distribution()
        
    #     new_shape = self._density_distribution.shape
    #     self._log_info(f"Quadrature order updated from {old_order} to {quadrature_order}, "
    #                    f"density distribution shape: {old_shape} -> {new_shape}")
        
    # def set_density_location(self, density_location: str, quadrature_order: Optional[int] = None) -> None:
    #     """设置密度分布位置并自动更新密度分布"""
    #     if density_location == 'element_gauss_integrate_point':
    #         if quadrature_order is None and self._quadrature_order is None:
    #             error_msg = ("quadrature_order is required when switching to 'element_gauss_integrate_point'. "
    #                         "Please provide quadrature_order parameter.")
    #             self._log_error(error_msg)
    #             raise ValueError(error_msg)
            
    #         if quadrature_order is not None:
    #             self._quadrature_order = quadrature_order

    #     old_location = self.density_location
    #     old_shape = self._density_distribution.shape
        
    #     self.setup_density_distribution.set(density_location)
    #     self._density_distribution = self.setup_density_distribution()
        
    #     new_shape = self._density_distribution.shape
    #     self._log_info(f"Density location changed from '{old_location}' to '{density_location}', "
    #                 f"density distribution shape: {old_shape} -> {new_shape}")

    def get_material_info(self) -> Dict[str, Any]:
        """获取材料信息, 包括基础材料和插值方案的参数"""
        base_material_info = self._base_material.get_material_params()

        interpolation_info = self._interpolation_scheme.get_interpolation_params()

        topology_info = {
                        'relative_density': self._relative_density,
                        'density_location': self.density_location,
                        'quadrature_order': self._quadrature_order,
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


    ###########################################################################################################################  
    # 内部方法
    ###########################################################################################################################

    # def _setup_function_spaces(self):
    #     """设置函数空间"""
    #     self._element_space = LagrangeFESpace(self._mesh, p=0, ctype='D')