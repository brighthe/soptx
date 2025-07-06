from typing import Optional, Union, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

class MaterialInterpolationScheme(BaseLogged):
    """材料插值方案类"""
    def __init__(self,
                penalty_factor: float = 3.0,
                void_youngs_modulus: float = 1e-12,
                interpolation_method: str = 'simp',
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        材料插值方法变体示例
        -----------------
        1. 实例化时设置默认插值方法
        mis = MaterialInterpolationScheme(interpolation_method='simp')
        
        2. 直接使用默认方法生成弹性矩阵
        D0 = mis.interpolate(base_material=base_material, density_distribution=density_distribution)

        3. 切换到其他插值方法 (推荐方式)
        mis.set_interpolation_method('modified_simp')  # 友好的接口
        D1 = mis.interpolate(base_material=base_material, density_distribution=density_distribution)
        
        4. 或者使用变体方法 (底层方式)
        mis.interpolate.set('modified_simp')  # 设置变体 (返回 None)
        D2 = mis.interpolate(base_material=base_material, density_distribution=density_distribution)

        注意: 
        -----
        - 推荐使用 set_interpolation_method() 方法切换插值方式，更直观和安全
        - interpolate.set() 是底层接口，只设置变体不执行方法，返回 None
        - 每次切换方法后，后续的 interpolate() 调用都使用新设置的插值方式

        Parameters:
        -----------
        interpolation_method
        - 'simp': 标准 SIMP 插值 E(ρ) = ρ^p * E0
        - 'modified_simp': 修正 SIMP 插值 E(ρ) = Emin + ρ^p * (E0 - Emin)
        - 'simp_double': 双指数 SIMP 插值
        - 'ramp': RAMP 插值
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._penalty_factor = penalty_factor
        self._void_youngs_modulus = void_youngs_modulus

        self.interpolate.set(interpolation_method)

        self._log_info(f"Material interpolation scheme initialized: "
                      f"type={interpolation_method}, p={penalty_factor}")

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

    @variantmethod('simp')
    def interpolate(self, 
                    base_material: LinearElasticMaterial, 
                    density_distribution: TensorLike
                ) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""

        if not bm.is_tensor(density_distribution):
            error_msg = f"density_distribution must be TensorLike, got {type(density_distribution)}"
            self._log_error(error_msg)
            raise TypeError(error_msg)
        
        D0 = base_material.elastic_matrix() # (1, 1, :, :)

        simp_scaled = density_distribution ** self._penalty_factor

        if len(simp_scaled.shape) == 1:
            NC = simp_scaled.shape[0]
            D = bm.einsum('c, ijkl -> cjkl', simp_scaled, D0)
            self._log_info(f"SIMP interpolation completed for {NC} elements "
                      f"with p = {self._penalty_factor}")
        
        elif len(simp_scaled.shape) == 2:
            NC, NQ = simp_scaled.shape
            D = bm.einsum('cq, ijkl -> cqkl', simp_scaled, D0)
            self._log_info(f"SIMP interpolation completed for {NC} elements "
                      f"with {NQ} quadrature points, p = {self._penalty_factor}")
        else:
            error_msg = f"Unsupported density_distribution shape: {density_distribution.shape}. " \
                        f"Expected (NC,) or (NC, NQ)."
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        return D
    
    @interpolate.register('modified_simp')
    def interpolate(self, 
                    base_material: LinearElasticMaterial, 
                    density_distribution: TensorLike
                ) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""

        if not bm.is_tensor(density_distribution):
            error_msg = f"density_distribution must be TensorLike, got {type(density_distribution)}"
            self._log_error(error_msg)
            raise TypeError(error_msg)

        E0 = base_material.youngs_modulus
        Emin = self._void_youngs_modulus
        D0 = base_material.elastic_matrix() # (1, 1, :, :)

        msimp_scaled = (Emin + density_distribution ** self._penalty_factor * (E0 - Emin)) / E0

        if len(msimp_scaled.shape) == 1:
            NC = msimp_scaled.shape[0]
            D = bm.einsum('c, ijkl -> cjkl', msimp_scaled, D0)
            self._log_info(f"Modified SIMP interpolation completed for {NC} elements "
                      f"with p = {self._penalty_factor}, Emin = {Emin}")

        elif len(msimp_scaled.shape) == 2:
            NC, NQ = msimp_scaled.shape
            D = bm.einsum('cq, ijkl -> cqkl', msimp_scaled, D0)
            self._log_info(f"Modified SIMP interpolation completed for {NC} elements "
                      f"with {NQ} quadrature points, p = {self._penalty_factor}, Emin = {Emin}")
        else:
            error_msg = f"Unsupported density_distribution shape: {density_distribution.shape}. " \
                         f"Expected (NC,) or (NC, NQ)."
            self._log_error(error_msg)
            raise ValueError(error_msg)

        return D
    
    @interpolate.register('simp_double')
    def interpolate(self, base_material: LinearElasticMaterial, density_distribution: TensorLike) -> TensorLike:
        """双指数 SIMP 插值"""
        pass

    @interpolate.register('ramp')
    def interpolate(self, base_material: LinearElasticMaterial, density_distribution: TensorLike) -> TensorLike:
        """RAMP 插值"""
        pass
