from abc import ABC, abstractmethod
from typing import Optional
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

class MaterialInterpolationScheme(BaseLogged, ABC):
    """材料插值方案的抽象基类"""
    def __init__(self, 
                enable_logging: bool = True,
                logger_name: Optional[str] = None):
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._relative_density = None
        self._penalty_factor = None

    @property
    def relative_density(self) -> Optional[TensorLike]:
        """获取当前的相对密度"""
        return self._relative_density

    @property
    def penalty_factor(self) -> Optional[float]:
        """获取当前的惩罚因子"""
        return self._penalty_factor

    def set_relative_density(self, relative_density: TensorLike) -> None:
        """设置相对密度"""
        if bm.any(relative_density < 0.0) or bm.any(relative_density > 1.0):
            min_val = bm.min(relative_density)
            max_val = bm.max(relative_density)
            error_msg = f"Relative density values must be in [0,1] range, got [{min_val:.6f}, {max_val:.6f}]"
            self._log_error(error_msg)
            raise ValueError(error_msg)
    
        self._relative_density = relative_density

        self._log_info(f"[MaterialInterpolationScheme] Relative density updated, "
                       f"shape: {relative_density.shape}, "
                       f"values: {relative_density}")
        
    def set_penalty_factor(self, penalty_factor: float) -> None:
        """设置惩罚因子"""
        if penalty_factor <= 0:
            error_msg = "Penalty factor must be positive"
            self._log_error(error_msg)
            raise ValueError(error_msg)

        old_penalty_factor = self._penalty_factor
        self._penalty_factor = penalty_factor
        self._log_info(f"[MaterialInterpolationScheme] Penalty factor updated from {old_penalty_factor} to {penalty_factor}")

    @abstractmethod
    def interpolate(self, base_material: LinearElasticMaterial, relative_density: Optional[TensorLike] = None) -> TensorLike:
        """根据相对密度插值材料属性"""
        pass

class SIMPInterpolationSingle(MaterialInterpolationScheme):
    """SIMP 插值方案 (单指数杨氏模量)"""
    def __init__(self, 
                penalty_factor: float = 3.0,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._penalty_factor = penalty_factor

        self._log_info(f"SIMP interpolation initialized with penalty factor p = {penalty_factor}")

    def interpolate(self, base_material: LinearElasticMaterial, relative_density: Optional[TensorLike] = None) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""
        if relative_density is not None:
            self._relative_density = relative_density
        elif self._relative_density is None:
            raise ValueError("No relative density provided and none stored internally")

        rho = self._relative_density

        D0 = base_material.elastic_matrix()
        simp_scaled = rho ** self.penalty_factor
        D = bm.einsum('b, ijkl -> bjkl', simp_scaled, D0)

        self._log_info(f"SIMP interpolation completed for {rho.shape[0]} elements with p = {self.penalty_factor}, "
                    f"simp_scaled values: {simp_scaled}")

        return D
    
class ModifiedSIMPInterpolationSingle(MaterialInterpolationScheme):
    """修正的 SIMP 插值方案 (单指数杨氏模量)"""
    def __init__(self, 
                penalty_factor: float = 3.0, 
                void_youngs_modulus: float = 1e-12,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._penalty_factor = penalty_factor
        self.Emin = void_youngs_modulus

        self._log_info(f"Modified SIMP interpolation initialized with p={penalty_factor}, Emin={void_youngs_modulus}")

    def interpolate(self, base_material: LinearElasticMaterial, relative_density: Optional[TensorLike] = None) -> TensorLike:
        """ 修改的 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""
        if relative_density is not None:
            self._relative_density = relative_density
        elif self._relative_density is None:
            raise ValueError("No relative density provided and none stored internally")

        rho = self._relative_density

        E0 = base_material.youngs_modulus
        D0 = base_material.elastic_matrix()
        msimp_scaled = (self.Emin + rho ** self.penalty_factor * (E0 - self.Emin)) / E0
        D = bm.einsum('b, ijkl -> bjkl', msimp_scaled, D0)

        self._log_info(f"Modified SIMP interpolation completed for {rho.shape[0]} elements with p = {self.penalty_factor}, "
                    f"msimp_scaled values: {rho}")

        return D

class SIMPInterpolationDouble(MaterialInterpolationScheme):
    pass

class RAMPInterpolation(MaterialInterpolationScheme):
    pass
