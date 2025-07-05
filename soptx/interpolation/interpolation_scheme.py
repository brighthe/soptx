from typing import Optional, Union

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
        1. 实例化时设置默认插值变体方法
        mis = MaterialInterpolationScheme(interpolation_method='simp')
        2. 直接使用默认方法生成材料插值类
        D0 = mis.interpolate(base_material=base_material, relative_density=relative_density)
        3. 切换到其他插值方法
        mis.interpolate.set('modified_simp')     # 设置变体 (返回 None)
        D1 = mis.interpolate(base_material=base_material, relative_density=relative_density)
        注意: 
        - interpolate.set() 只设置变体，不执行方法，返回 None
        - 需要分别调用 set() 和 interpolate() 来生成获取弹性矩阵
        - 每次 set() 后，后续的 interpolate() 调用都使用新设置的变体

        Parameters
        ----------
        interpolation_method : 
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

    @variantmethod('simp')
    def interpolate(self, base_material: LinearElasticMaterial, 
                    relative_density: Union[float, TensorLike]) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""
        if bm.any(relative_density < 0.0) or bm.any(relative_density > 1.0):
            min_val = bm.min(relative_density)
            max_val = bm.max(relative_density)
            error_msg = f"Relative density values must be in [0,1] range, got [{min_val:.6f}, {max_val:.6f}]"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        D0 = base_material.elastic_matrix() # (1, 1, :, :)
        simp_scaled = relative_density ** self._penalty_factor
        if simp_scaled.shape[0] == 1:
        D = bm.einsum('b, cqkl -> cqkl', simp_scaled, D0)

        self._log_info(f"SIMP interpolation completed for {relative_density.shape[0]} elements "
                      f"with p = {self._penalty_factor}")
        
        return D
    
    @interpolate.register('modified_simp')
    def interpolate(self, base_material: LinearElasticMaterial, relative_density: TensorLike) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""
        if bm.any(relative_density < 0.0) or bm.any(relative_density > 1.0):
            min_val = bm.min(relative_density)
            max_val = bm.max(relative_density)
            error_msg = f"Relative density values must be in [0,1] range, got [{min_val:.6f}, {max_val:.6f}]"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        E0 = base_material.youngs_modulus
        Emin = self._void_youngs_modulus
        D0 = base_material.elastic_matrix()

        msimp_scaled = (Emin + relative_density ** self._penalty_factor * (E0 - Emin)) / E0
        D = bm.einsum('b, ijkl -> bjkl', msimp_scaled, D0)

        self._log_info(f"Modified SIMP interpolation completed for {relative_density.shape[0]} elements "
                      f"with p = {self._penalty_factor}, Emin = {self._void_youngs_modulus}")
        return D
    
    @interpolate.register('simp_double')
    def interpolate(self, base_material: LinearElasticMaterial, relative_density: TensorLike) -> TensorLike:
        """双指数 SIMP 插值"""
        pass

    @interpolate.register('ramp')
    def interpolate(self, base_material: LinearElasticMaterial, relative_density: TensorLike) -> TensorLike:
        """RAMP 插值"""
        pass
