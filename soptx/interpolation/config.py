# soptx/config/topopt_config.py
from typing import Literal, List
from dataclasses import dataclass, field

__all__ = ['InterpolationConfig', 'DensityBasedConfig']

@dataclass
class InterpolationConfig:
    """材料插值方法配置类"""
    method: Literal['simp', 'msimp', 'ramp'] = 'simp'
    penalty_factor: float = 3.0
    target_variables: List[Literal['E', 'nu', 'lam', 'mu']] = field(default_factory=lambda: ['E'])
    void_youngs_modulus: float = 1e-12 

@dataclass
class DensityBasedConfig:
    """基于密度的拓扑优化方法配置类
    
    # 示例 1: 自定义 SIMP 配置 (插值杨氏模量)
    simp_config = DensityBasedConfig(
                        density_location='element',
                        initial_density=0.5,
                        interpolation=InterpolationConfig(method='simp',
                                                        penalty_factor=3.0,
                                                        target_variables=['E'],
                                                        void_youngs_modulus=1e-9),
                    )

    analyzer = LagrangeFEMAnalyzer(
                        mesh=mesh,
                        pde=pde,
                        material=material,
                        topopt_algorithm='density_based',
                        topopt_config=simp_config
                    )
    """
    density_location: Literal['element', 'gauss_integration_point', 'continuous'] = 'element'
    initial_density: float = 0.5
    interpolation: InterpolationConfig = None
    
    def __post_init__(self):
        """后初始化处理，设置默认插值配置"""
        if self.interpolation is None:
            self.interpolation = InterpolationConfig()

@dataclass
class LevelSetConfig:
    pass