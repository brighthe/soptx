from typing import Optional, Dict, Any, Literal, List

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod, cartesian
from fealpy.typing import TensorLike
from fealpy.functionspace import Function, LagrangeFESpace
from fealpy.mesh import HomogeneousMesh

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

class MaterialInterpolationScheme(BaseLogged):
    """材料插值方案类"""
    def __init__(self,
                density_location: Literal['element', 'element_coscos', 
                                          'gauss_integration_point', 
                                          'continuous'] = 'element',
                interpolation_method: Literal['simp', 'msimp', 'ramp'] = 'simp',
                options: Optional[dict] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        材料插值方法变体示例
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._density_location = density_location
        self._interpolation_method = interpolation_method

        self._options = options or {}
        self._set_default_options()

        self.setup_density_distribution.set(density_location)
        self.interpolate_map.set(interpolation_method)

        self._log_info(f"Material interpolation scheme initialized: "
                      f"density_location={density_location}, "
                      f"interpolation_method={interpolation_method}, ")
        
    
    #########################################################################################
    # 属性访问器
    #########################################################################################

    @property
    def density_location(self) -> Optional[str]:
        """获取当前的密度位置"""
        return self._density_location

    @property
    def interpolation_method(self) -> Optional[str]:
        """获取当前的插值方法"""
        return self._interpolation_method

    
    #########################################################################################
    # 变体方法
    #########################################################################################

    @variantmethod('element')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integrator_order: int = None,
                                **kwargs,
                            ) -> Function:
        """单元密度分布"""
        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), relative_density, dtype=bm.float64, device=mesh.device)

        space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = space.function(density_tensor)

        self._log_info(f"Element density: shape={density_dist.shape}, value={relative_density}")

        return density_dist
    
    @setup_density_distribution.register('element_coscos')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = None,
                                integrator_order: int = None,
                                **kwargs,
                            ) -> Function:
        """单元密度分布 (变体1)"""
        element_space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = element_space.interpolate(u=self._density_distribution_coscos)

        self._log_info(f"此时的密度是分片常数函数, ")

        return density_dist

    @setup_density_distribution.register('gauss_integration_point')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integrator_order: int = 3,
                                **kwargs,
                            ) -> TensorLike:
        """单元高斯点密度分布"""
        qf = mesh.quadrature_formula(integrator_order)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), relative_density, dtype=bm.float64, device=mesh.device)

        space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = space.function(density_tensor)
        density_dist = density_dist(bcs)
        density_dist = space.function(density_dist)

        self._log_info(f"Element-Gauss density: shape={density_dist.shape}, value={relative_density}, q={integrator_order}")

        return density_dist
    
    @setup_density_distribution.register('continuous')
    def setup_density_distribution(self,
                                   mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   interpolation_order: int = 1,
                                   **kwargs,
                                ) -> Function:
        "连续插值点密度分布"
        def density_func(points: TensorLike) -> TensorLike:
            NI = points.shape[0] 

            return bm.full((NI,), relative_density, dtype=bm.float64, device=points.device)
    
        space = LagrangeFESpace(mesh, p=interpolation_order, ctype='C')
        density_dist = space.interpolate(u=density_func)

        self._log_info(f"Continuous density: shape={density_dist.shape}, value={relative_density}, p={interpolation_order}")
        
        return density_dist
        

    @variantmethod('simp')
    def interpolate_map(self, 
                    material: LinearElasticMaterial, 
                    density_distribution: Function,
                ) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']

        if target_variables == ['E']:
            E0 = material.youngs_modulus
            simp_map = density_distribution[:] ** penalty_factor * E0 / E0
            
            return simp_map
    
    @interpolate_map.register('msimp')
    def interpolate_map(self, 
                    material: LinearElasticMaterial, 
                    density_distribution: Function,
                ) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""

        required_options = ['penalty_factor', 'target_variables', 'void_youngs_modulus']
        for option in required_options:
            if option not in self._options:
                error_msg = f"Missing required option '{option}' for msimp interpolation"
                self._log_error(error_msg)
                raise ValueError(error_msg)

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']
        void_youngs_modulus = self._options['void_youngs_modulus']

        if target_variables == ['E']:
            E0 = material.youngs_modulus
            Emin = void_youngs_modulus
            msimp_map = (Emin + density_distribution[:] ** penalty_factor * (E0 - Emin)) / E0

            return msimp_map

    @interpolate_map.register('simp_double')
    def interpolate_map(self) -> TensorLike:
        """双指数 SIMP 插值"""
        pass

    @interpolate_map.register('ramp')
    def interpolate_map(self) -> TensorLike:
        """RAMP 插值"""
        pass


    ###########################################################################################################
    # 核心方法
    ###########################################################################################################

    def interpolate_derivative(self,
                        material: LinearElasticMaterial, 
                        density_distribution: Function,
                    ) -> TensorLike:
        """获取当前插值方法的导数对应的系数"""

        method = self.interpolation_method
        p = self._options['penalty_factor'] 

        if method == 'simp':
            dval = p * density_distribution[:] ** (p - 1)

            return dval

        elif method == 'msimp':
            E0 = material.youngs_modulus
            Emin = self._options['void_youngs_modulus']
            dval = p * density_distribution[:] ** (p - 1) * (E0 - Emin) / E0
            
            return dval


    ###########################################################################################################
    # 内部方法
    ###########################################################################################################

    def _set_default_options(self) -> None:
        """设置默认选项"""
        defaults = {
            'penalty_factor': 3.0,
            'void_youngs_modulus': 1e-9,
            'target_variables': ['E']
        }
        
        for key, default_value in defaults.items():
            if key not in self._options:
                self._options[key] = default_value
    
    @cartesian
    def _density_distribution_coscos(self, points: TensorLike) -> TensorLike:
        """
        周期性密度函数, 基于二维余弦函数和双曲正切变换

        数学表达式：
            base = bm.cos(scale * x) * bm.cos(scale * y)
            density = 0.5 * (1 + bm.tanh(s * (base + 2 * r - 1)))
        """
        x, y = points[..., 0], points[..., 1]

        scale = 2
        base_pattern = bm.cos(scale * x) * bm.cos(scale * y)

        r = 0.5
        smoothness = 2.0 
        val = 0.5 * (1 + bm.tanh(smoothness * (base_pattern + 2 * r - 1)))

        return val