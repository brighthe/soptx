from abc import ABC, abstractmethod
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .linear_elastic_material import IsotropicLinearElasticMaterial

class MaterialInterpolationScheme(ABC):
    @abstractmethod
    def interpolate(self, base_material: IsotropicLinearElasticMaterial, relative_density: TensorLike) -> TensorLike:
        pass

class SIMPInterpolationSingle(MaterialInterpolationScheme):
    def __init__(self, penalty_factor: float = 3.0):
        self.p = penalty_factor

    def interpolate(self, base_material: IsotropicLinearElasticMaterial, relative_density: TensorLike) -> TensorLike:
        rho = relative_density
        D0 = base_material.elastic_matrix()
        simp_scaled = rho ** self.p
        D = bm.einsum('b, ijkl -> bjkl', simp_scaled, D0)

        return D
    
class ModifiedSIMPInterpolationSingle(MaterialInterpolationScheme):
    def __init__(self, penalty_factor: float = 3.0, void_youngs_modulus: float = 1e-12):
        self.p = penalty_factor
        self.Emin = void_youngs_modulus

    def interpolate(self, base_material: IsotropicLinearElasticMaterial, relative_density: TensorLike) -> TensorLike:
        rho = relative_density
        E0 = base_material.youngs_modulus
        D0 = base_material.elastic_matrix()
        msimp_scaled = (self.Emin + rho ** self.p * (E0 - self.Emin)) / E0
        D = bm.einsum('b, ijkl -> bjkl', msimp_scaled, D0)

        return D

class SIMPInterpolationDouble(MaterialInterpolationScheme):
    pass

class RAMPInterpolation(MaterialInterpolationScheme):
    pass
