from abc import ABC, abstractmethod
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

# 抽象基类
class MaterialInterpolation(ABC):
    """Abstract base class for material interpolation models."""
    
    def __init__(self, name: str):
        """
        初始化插值模型并设置标识名.
        
        Args:
            name : 插值模型的唯一标识符
        """
        self.name = name

    @abstractmethod
    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, 
                        Pmin: float, 
                        penal: float
                        ) -> TensorLike:
        """
        计算插值后的材料属性场.
        
        Args:
            rho : Material density field
            P0 : Property value for solid material
            Pmin : Property value for void material 
            penal : Penalization factor
            
        Returns:
            TensorLike: Interpolated property field
        """
        pass

    @abstractmethod
    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penal: float
                                    ) -> TensorLike:
        """
        计算插值后材料属性的导数
                
        Args:
            rho : Material density field
            P0 : Property value for solid material
            Pmin : Property value for void material
            penal : Penalization factor
            
        Returns:
            TensorLike: Derivative of interpolated property field
        """
        pass


# 具体实现类
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
            return P0 * (1 + penalty_factor) * (1 + penalty_factor * (1 - rho)) ** (-2)
        else:
            return (P0 - Pmin) * (1 + penalty_factor) * (1 + penalty_factor * (1 - rho)) ** (-2)