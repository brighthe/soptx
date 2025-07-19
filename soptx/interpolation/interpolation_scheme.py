from typing import Optional, Dict, Any, Literal, List

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike
from fealpy.functionspace import Function, LagrangeFESpace
from fealpy.mesh import HomogeneousMesh

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

class MaterialInterpolationScheme(BaseLogged):
    """材料插值方案类"""
    def __init__(self,
                density_location: Literal['element', 'gauss_integration_point', 'continuous'] = 'element',
                interpolation_method: Literal['simp', 'msimp', 'ramp'] = 'simp',
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        材料插值方法变体示例
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.setup_density_distribution.set(density_location)
        self.interpolate_map.set(interpolation_method)

        self._log_info(f"Material interpolation scheme initialized: "
                      f"density_location={density_location}, "
                      f"interpolation_method={interpolation_method}, ")
        
    
    #####################################################################################################################################
    # 变体方法
    #####################################################################################################################################

    @variantmethod('element')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float,
                                integrator_order: int = None,
                                **kwargs,
                            ) -> Function:
        """单元密度分布"""
        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), relative_density, dtype=bm.float64, device=mesh.device)

        element_space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = element_space.function(density_tensor)

        self._log_info(f"Element density: shape={density_dist.shape}, value={relative_density}")

        return density_dist

    @setup_density_distribution.register('gauss_integration_point')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float,
                                integrator_order: int = None,
                                **kwargs,
                            ) -> Function:
        """单元高斯点密度分布"""
        if integrator_order is None:
            error_msg = "integrator_order must be specified for 'gauss_integration_point'"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        qf = mesh.quadrature_formula(integrator_order)
        bcs, ws = qf.get_quadrature_points_and_weights()
        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), relative_density, dtype=bm.float64, device=mesh.device)

        element_space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = element_space.function(density_tensor)
        density_dist = density_dist(bcs)

        self._log_info(f"Element-Gauss density: shape={density_dist.shape}, value={relative_density}, q={integrator_order}")

        return density_dist
    
    @setup_density_distribution.register('continuous')
    def setup_density_distribution(self):
        "连续密度分布"
        pass

    @variantmethod('simp')
    def interpolate_map(self, 
                    material: LinearElasticMaterial, 
                    density_distribution: Function,
                    penalty_factor: float = 3.0,
                    target_variables: List[Literal['E', 'nu', 'lam', 'mu']] = ['E'],
                    void_youngs_modulus: Optional[float] = None,
                    **kwargs,
                ) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""

        if target_variables == ['E']:
            E0 = material.youngs_modulus
            simp_map = density_distribution[:] ** penalty_factor * E0 / E0
            
            return simp_map
    
    @interpolate_map.register('msimp')
    def interpolate_map(self, 
                    material: LinearElasticMaterial, 
                    density_distribution: Function,
                    penalty_factor: float = 3.0,
                    target_variables: List[Literal['E', 'nu', 'lam', 'mu']] = ['E'],
                    void_youngs_modulus: Optional[float] = None,
                    **kwargs,
                ) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""
        if void_youngs_modulus is None:
            error_msg = "void_youngs_modulus must be specified for 'msimp'"
            self._log_error(error_msg)
            raise ValueError(error_msg)

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

    #####################################################################################################################################
    # 变体方法
    #####################################################################################################################################

    def interpolate_derivative(self,
                        base_material: LinearElasticMaterial, 
                        density_distribution: TensorLike,
                    ) -> TensorLike:
        """获取当前插值方法的导数对应的系数"""

        if not bm.is_tensor(density_distribution):
            error_msg = f"density_distribution must be TensorLike, got {type(density_distribution)}"
            self._log_error(error_msg)
            raise TypeError(error_msg)
        
        method = self.interpolation_method
        p = self._penalty_factor

        if method == 'simp':
            return p * density_distribution ** (p - 1)

        elif method == 'modified_simp':
            E0 = base_material.youngs_modulus
            Emin = self._void_youngs_modulus
            return p * density_distribution * (p - 1) * (E0 - Emin) / E0







    @property
    def penalty_factor(self) -> Optional[float]:
        """获取当前的惩罚因子"""
        return self._penalty_factor
    
    @property
    def void_youngs_modulus(self) -> float:
        """获取当前的空心材料杨氏模量"""
        return self._void_youngs_modulus
    
    @property
    def interpolation_method(self) -> Optional[str]:
        """获取当前的插值方法"""
        return self.interpolate.vm.get_key(self)
        
    def set_penalty_factor(self, penalty_factor: float) -> None:
        """设置惩罚因子"""
        if penalty_factor <= 0:
            error_msg = "Penalty factor must be positive"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        old_penalty_factor = self._penalty_factor
        self._penalty_factor = penalty_factor

        self._log_info(f"[MaterialInterpolationScheme] Penalty factor updated from "
                       f"{old_penalty_factor} to {penalty_factor}")

    def set_void_youngs_modulus(self, void_youngs_modulus: float) -> None:
        """设置空心材料的杨氏模量"""
        if void_youngs_modulus < 0:
            error_msg = "Void Young's modulus must be non-negative"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        old_void_youngs_modulus = self._void_youngs_modulus
        self._void_youngs_modulus = void_youngs_modulus

        self._log_info(f"[MaterialInterpolationScheme] Void Young's modulus updated from "
                       f"{old_void_youngs_modulus} to {void_youngs_modulus}")
        
    def set_interpolation_method(self, interpolation_method: str) -> None:
        """设置材料插值方法"""
        supported_methods = ['simp', 'modified_simp', 'simp_double', 'ramp']
        if interpolation_method not in supported_methods:
            error_msg = f"Unsupported interpolation method '{interpolation_method}'. " \
                    f"Supported methods: {supported_methods}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        old_method = self.interpolation_method
        self.interpolate.set(interpolation_method)
        
        self._log_info(f"[MaterialInterpolationScheme] Interpolation method updated from "
                    f"'{old_method}' to '{interpolation_method}'")

    def get_interpolation_params(self) -> Dict[str, Any]:
        """获取当前的插值参数"""
        params = {
            'interpolation_method': self.interpolation_method,
            'penalty_factor': self.penalty_factor,
        }

        return params

    def display_interpolation_params(self) -> None:
        """显示当前的插值参数"""
        params = self.get_interpolation_params()
        self._log_info(f"Interpolation parameters: {params}", force_log=True)



        

   

        # if not bm.is_tensor(density_distribution):
        #     error_msg = f"density_distribution must be TensorLike, got {type(density_distribution)}"
        #     self._log_error(error_msg)
        #     raise TypeError(error_msg)
        
        # D0 = base_material.elastic_matrix() # (1, 1, :, :)

        # simp_scaled = density_distribution ** self._penalty_factor

        # if len(simp_scaled.shape) == 1:
        #     NC = simp_scaled.shape[0]
        #     D = bm.einsum('c, ijkl -> cjkl', simp_scaled, D0)
        #     self._log_info(f"SIMP interpolation completed for {NC} elements "
        #               f"with p = {self._penalty_factor}")
        
        # elif len(simp_scaled.shape) == 2:
        #     NC, NQ = simp_scaled.shape
        #     D = bm.einsum('cq, ijkl -> cqkl', simp_scaled, D0)
        #     self._log_info(f"SIMP interpolation completed for {NC} elements "
        #               f"with {NQ} quadrature points, p = {self._penalty_factor}")
        # else:
        #     error_msg = f"Unsupported density_distribution shape: {density_distribution.shape}. " \
        #                 f"Expected (NC,) or (NC, NQ)."
        #     self._log_error(error_msg)
        #     raise ValueError(error_msg)
        


        # if not bm.is_tensor(density_distribution):
        #     error_msg = f"density_distribution must be TensorLike, got {type(density_distribution)}"
        #     self._log_error(error_msg)
        #     raise TypeError(error_msg)

        # E0 = base_material.youngs_modulus
        # Emin = self._void_youngs_modulus
        # D0 = base_material.elastic_matrix() # (1, 1, :, :)

        # msimp_scaled = (Emin + density_distribution ** self._penalty_factor * (E0 - Emin)) / E0

        # if len(msimp_scaled.shape) == 1:
        #     NC = msimp_scaled.shape[0]
        #     D = bm.einsum('c, ijkl -> cjkl', msimp_scaled, D0)
        #     self._log_info(f"Modified SIMP interpolation completed for {NC} elements "
        #               f"with p = {self._penalty_factor}, Emin = {Emin}")

        # elif len(msimp_scaled.shape) == 2:
        #     NC, NQ = msimp_scaled.shape
        #     D = bm.einsum('cq, ijkl -> cqkl', msimp_scaled, D0)
        #     self._log_info(f"Modified SIMP interpolation completed for {NC} elements "
        #               f"with {NQ} quadrature points, p = {self._penalty_factor}, Emin = {Emin}")
        # else:
        #     error_msg = f"Unsupported density_distribution shape: {density_distribution.shape}. " \
        #                  f"Expected (NC,) or (NC, NQ)."
        #     self._log_error(error_msg)
        #     raise ValueError(error_msg)

        # return D
    

