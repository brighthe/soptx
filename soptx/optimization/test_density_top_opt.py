from typing import Optional, Union
from pathlib import Path

from fealpy.decorator import variantmethod

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import save_optimization_history, plot_optimization_history


class DensityTopOptTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    def set_pde(self, pde):
        self.pde = pde

    def set_init_mesh(self, meshtype: str, **kwargs):
        self.pde.init_mesh.set(meshtype)
        self.mesh = self.pde.init_mesh(**kwargs)

    def set_material(self, material):
        self.material = material

    def set_space_degree(self, space_degree: int):
        self.space_degree = space_degree

    def set_integrator_order(self, integrator_order: int):
        self.integrator_order = integrator_order

    def set_assembly_method(self, method: str):
        self.assembly_method = method

    def set_solve_method(self, method: str):
        self.solve_method = method

    def set_volume_fraction(self, volume_fraction: float):
        self.volume_fraction = volume_fraction

    def set_relative_density(self, relative_density: float):
        self.relative_density = relative_density

    @variantmethod('OC')
    def run(self):
        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, 60, 0, 20],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        domain_length = pde.domain[1] - pde.domain[0]
        domain_height = pde.domain[3] - pde.domain[2]
        pde.init_mesh.set('uniform_quad')
        fe_mesh = pde.init_mesh(nx=60, ny=20)

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        opt_mesh = pde.init_mesh(nx=60, ny=20)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location='element',
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-12,
                                        'target_variables': ['E']
                                    },
                                )

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=1.0,
                                                integrator_order=self.integrator_order,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=1,
                                    integrator_order=self.integrator_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=self.volume_fraction)

        from soptx.regularization.filter import Filter
        rmin = 0.04 * domain_length
        filter_regularization = Filter(
                                    mesh=opt_mesh,
                                    filter_type='sensitivity',
                                    rmin=rmin
                                )


        from soptx.optimization.oc_optimizer import OCOptimizer
        oc_optimizer = OCOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': 200,
                                'tolerance': 1e-2,
                            }
                        )
        # 设置高级参数 (可选)
        oc_optimizer.options.set_advanced_options(
                                    move_limit=0.2,
                                    damping_coef=0.5,
                                    initial_lambda=1e9,
                                    bisection_tol=1e-3
                                )
        rho_opt, history = oc_optimizer.optimize(density_distribution=rho)

        # 保存结果
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/density_top_opt_results")
        save_path.mkdir(parents=True, exist_ok=True)
        save_optimization_history(opt_mesh, history, str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    
if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)
    
    p = 1
    q = p+3
    test.set_space_degree(p)
    test.set_integrator_order(q)
    test.set_assembly_method('standard')
    test.set_solve_method('mumps')
    test.set_volume_fraction(0.5)
    test.set_relative_density(0.5)

    test.run.set('OC')

    rho_opt, history = test.run()