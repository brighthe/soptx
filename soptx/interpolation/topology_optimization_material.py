from fealpy.typing import TensorLike

from .linear_elastic_material import LinearElasticMaterial
from .interpolation_scheme import MaterialInterpolationScheme

class TopologyOptimizationMaterial:
    def __init__(self, 
                base_material: LinearElasticMaterial,
                interpolation_scheme: MaterialInterpolationScheme,
                relative_density: TensorLike
            ) -> None:
        """
        The generic, stateful context class that uses a specific interpolation strategy.
        This class is passed to the integrator.
        """
        self.base_material = base_material
        self.interpolation_scheme = interpolation_scheme
        self.relative_density = relative_density

    def elastic_matrix(self) -> TensorLike:
        D = self.interpolation_scheme.interpolate(self.base_material, self.relative_density)
        
        return D