from fealpy.fem import BilinearForm

from .integrators.linear_elastic_integrator import LinearElasticIntegrator
from ..interpolation.interpolation_scheme import LinearElasticMaterial

class LagrangeFEMAnalyzer:
    def __init__(self):
        pass

    def linear_sysytem():
        LEM = LinearElasticMaterial()
        LEI = LinearElasticIntegrator(material=LEM, q=5)
        bform = BilinearForm()
        bform.add_integrator(LEI)

        A = bform.assembly(format='csr')

