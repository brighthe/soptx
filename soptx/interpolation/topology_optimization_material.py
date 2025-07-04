from typing import Optional, Dict, Any

from fealpy.typing import TensorLike

from .linear_elastic_material import LinearElasticMaterial
from .interpolation_scheme import MaterialInterpolationScheme
from ..utils.base_logged import BaseLogged

class TopologyOptimizationMaterial(BaseLogged):
    def __init__(self, 
                base_material: LinearElasticMaterial,
                interpolation_scheme: MaterialInterpolationScheme,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.base_material = base_material
        self.interpolation_scheme = interpolation_scheme

    @property
    def relative_density(self) -> Optional[TensorLike]:
        """获取当前的相对密度"""
        return self.interpolation_scheme.relative_density
    
    @property
    def penalty_factor(self) -> float:
        """获取当前的惩罚因子"""
        return self.interpolation_scheme.penalty_factor
    
    def set_material_parameters(self, **kwargs) -> None:
        """设置基础材料参数（代理方法）"""
        self.base_material.set_material_parameters(**kwargs)
        self._log_info(f"[TopologyOptimizationMaterial] Material parameters updated via proxy method")
    
    def set_relative_density(self, relative_density: TensorLike) -> None:
        """设置相对密度分布 (代理方法)"""
        self.interpolation_scheme.set_relative_density(relative_density)
        self._log_info(f"[TopologyOptimizationMaterial] Relative density updated via proxy method")

    def set_penalty_factor(self, penalty_factor: float) -> None:
        """设置惩罚因子 (代理方法)"""
        self.interpolation_scheme.set_penalty_factor(penalty_factor)
        self._log_info(f"[TopologyOptimizationMaterial] Penalty factor updated via proxy method")

    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """计算插值后的弹性矩阵"""
        if self.relative_density is None:
            error_msg = "No relative density set. Please call set_relative_density() first."
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        D = self.interpolation_scheme.interpolate(self.base_material)

        self._log_info(f"[TopologyOptimizationMaterial] Elastic matrix computed successfully, "
                   f"shape: {D.shape}")
        
        return D
        
    def get_material_info(self) -> Dict[str, Any]:
        """获取材料信息, 包括基础材料和插值方案的参数"""
        base_material_info = {
            'base_material_type': type(self.base_material).__name__,
            **self.base_material.get_material_params(),
        }

        interp_info = {
            'interpolation_scheme': type(self.interpolation_scheme).__name__,
            'penalty_factor': self.penalty_factor,
            'relative_density': self.relative_density
        }

        material_info = {
            **base_material_info,
            **interp_info

        }
        self._log_info(f"[TopologyOptimizationMaterial] Material info retrieved via get_material_info method")

        return material_info

    def display_material_info(self) -> None:
        """显示材料信息"""
        info = self.get_material_info()

        self._log_info(f"Topology optimization material info: {info}", force_log=True)
        