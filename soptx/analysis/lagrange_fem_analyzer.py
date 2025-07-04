from fealpy.fem import BilinearForm

from .integrators.linear_elastic_integrator import LinearElasticIntegrator
from ..interpolation.linear_elastic_material import IsotropicLinearElasticMaterial

class LagrangeFEMAnalyzer:
    def __init__(self):
        pass

    def set_pde(self):
        pass

    def set_material_parameters(self):
        pass
        
    def linear_system(self):
        pass
    
    def apply_bc(self):
        pass

    def solve(self):
        pass

