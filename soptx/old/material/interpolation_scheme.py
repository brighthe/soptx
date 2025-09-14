from abc import ABC, abstractmethod
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

class MaterialInterpolation(ABC):
    """材料插值模型的抽象基类"""
    
    def __init__(self, name: str):
        """初始化插值模型并设置标识名"""
        self.name = name

    @abstractmethod
    def calculate_property(self, 
            field_variable: TensorLike, 
            max_property: float, 
            min_property: float, 
            **kwargs
        ) -> TensorLike:
        """
        计算插值后的材料属性
        
        Parameters:
        - field_variable: 场变量(密度或水平集函数等)
        - max_property: 实体材料的属性值
        - min_property: 空洞材料的属性值
        - **kwargs: 其他特定于插值方案的参数
        """
        pass

    @abstractmethod
    def calculate_property_derivative(self, 
            field_variable: TensorLike, 
            max_property: float, 
            min_property: float, 
            **kwargs
        ) -> TensorLike:
        """计算材料属性对场变量的导数"""
        pass

class SIMPInterpolation(MaterialInterpolation):
    """Solid Isotropic Material with Penalization (SIMP) interpolation model."""

    def __init__(self, penalty_factor: float = 3.0):
        """Initialize SIMP interpolation."""
        if penalty_factor <= 0:
            raise ValueError("Penalty factor must be positive")
        
        super().__init__(name="SIMP")

        self.penalty_factor = penalty_factor

    def calculate_property(self, 
            rho: TensorLike, 
            P0: float, 
            Pmin: float, 
            penalty_factor: float
        ) -> TensorLike:
        """Calculate interpolated property using SIMP model."""

        if Pmin is None:
            P = rho ** penalty_factor * P0
        else:
            P = Pmin + rho ** penalty_factor * (P0 - Pmin)
        return P

    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penalty_factor: float
                                    ) -> TensorLike:
        """Calculate derivative of interpolated property using SIMP model."""

        if Pmin is None:
            dP = penalty_factor * rho ** (penalty_factor - 1) * P0
            return dP
        else:
            dP = penalty_factor * rho ** (penalty_factor - 1) * (P0 - Pmin)
            return dP


class RAMPInterpolation(MaterialInterpolation):
    """Rational Approximation of Material Properties (RAMP) interpolation model."""
    
    def __init__(self, penalty_factor: float = 3.0):
        """Initialize RAMP interpolation."""
        if penalty_factor <= 0:
            raise ValueError("Penalty factor must be positive")
        
        super().__init__(name="RAMP")

        self.penalty_factor = penalty_factor

    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, 
                        Pmin: float, 
                        penalty_factor: float
                        ) -> TensorLike:
        """Calculate interpolated property using 'RAMP' model."""

        if Pmin is None:
            P = rho * (1 + penalty_factor * (1 - rho)) ** (-1) * P0
        else:
            P = Pmin + (P0 - Pmin) * rho * (1 + penalty_factor * (1 - rho)) ** (-1)
        return P

    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penalty_factor: float
                                    ) -> TensorLike:
        """Calculate derivative of interpolated property using 'RAMP' model."""
        
        if Pmin is None:
            return P0*(1+penalty_factor) * (1+penalty_factor*(1-rho))**(-2)
        else:
            return (P0-Pmin) * (1+penalty_factor) * (1+penalty_factor*(1-rho))**(-2)
        
class LevelSetAreaRatioInterpolation(MaterialInterpolation):
    """使用面积比例的水平集插值方法"""
    
    def __init__(self):
        """初始化水平集面积比例插值模型"""

        super().__init__(name="AreaRatio")

    def calculate_property(self, 
            phi: TensorLike, 
            max_property: float, 
            min_property: float, 
        ) -> TensorLike:
        """根据单元节点的水平集值计算单元材料属性"""

        NC = phi.shape[0]
        E = bm.zeros(NC, dtype=phi.dtype, device=phi.device)

        s, t = bm.meshgrid(bm.arange(-1, 1.1, 0.1), bm.arange(-1, 1.1, 0.1))
        s = s.flatten()
        t = t.flatten()

        inside_mask = bm.min(phi, axis=1) > 0
        outside_mask = bm.max(phi, axis=1) < 0
        boundary_mask = ~(inside_mask | outside_mask)

        E = bm.where(inside_mask, max_property, E)
        E = bm.where(outside_mask, min_property, E)
        
        if bm.any(boundary_mask):
            boundary_indices = bm.where(boundary_mask)[0]

            for idx in boundary_indices:
                node_phi = phi[idx]

                # 使用双线性形函数插值计算采样点的 phi 值
                tmp_phi = ((1 - s)*(1 - t)/4 * node_phi[0] + 
                        (1 + s)*(1 - t)/4 * node_phi[1] + 
                        (1 + s)*(1 + t)/4 * node_phi[2] + 
                        (1 - s)*(1 + t)/4 * node_phi[3])
            
                # 计算结构内部的面积比例
                area_ratio = bm.sum(tmp_phi >= 0) / len(s)
            
                # 按面积比例混合属性值
                E = bm.set_at(E, idx, \
                            area_ratio * max_property + (1 - area_ratio) * min_property)
        return E
        
    def calculate_property_derivative(self,
            phi: TensorLike, 
            max_property: float, 
            min_property: float, 
        ) -> TensorLike:
        pass
