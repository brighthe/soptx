from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import save_optimization_history, plot_optimization_history
from soptx.optimization.tools import OptimizationHistory


class DensityTopOptPVersionTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)


    @variantmethod('test_mbb_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60, 0, 10]
        T = -2.0
        E, nu = 1.0, 0.3

        nx, ny = 60, 10
        # nx, ny = 120, 20
        # nx, ny = 240, 40
        # nx, ny = 480, 80
        mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_aligned_tri'
        # mesh_type = 'uniform_crisscross_tri'

        space_degree = 8
        integration_order = space_degree + 1

        volume_fraction = 0.6
        penalty_factor = 3.0

        # density_location = 'element_multi_resolution'
        density_location = 'element_multi_resolution2'
        multi_resolution = 8

        relative_density = volume_fraction

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500

        filter_type = 'density' # 'none', 'sensitivity', 'density'

        domain_length = domain[1] - domain[0]
        # rmin = (nx * 0.04) * (domain_length / nx)
        rmin = 1.2            
        # rmin = 0.75
        # rmin = 0.5
        # rmin = 0.25

        from soptx.model.mbb_beam_2d import MBBBeam2dData
        pde = MBBBeam2dData(
                            domain=domain,
                            T=T, E=E, nu=nu,
                            enable_logging=False
                        )
        
        pde.init_mesh.set(mesh_type)
        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

        # fe_mesh.to_vtk('fe_mesh.vtu')

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        opt_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=relative_density,
                                                multi_resolution=multi_resolution
                                            )
        

if __name__ == "__main__":
    test = DensityTopOptPVersionTest(enable_logging=True)

    test.run.set('test_mbb_2d')
    test.run()