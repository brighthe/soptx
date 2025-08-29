from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import save_optimization_history, plot_optimization_history
from soptx.optimization.tools import OptimizationHistory

class DensityTopOptTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)


    @variantmethod('test_mbb_2d')
    def run(self, parameter_type: str = 'mbb_2d') -> Union[TensorLike, OptimizationHistory]:

        if parameter_type == 'mbb_2d':
            domain = [0, 60, 0, 10]
            T = -2.0
            E, nu = 1.0, 0.3

            # nx, ny = 60, 10
            nx, ny = 120, 20
            # nx, ny = 240, 40
            # nx, ny = 480, 80
            # nx, ny = 300, 50
            mesh_type = 'uniform_quad'
            # mesh_type = 'uniform_aligned_tri'
            # mesh_type = 'uniform_crisscross_tri'

            space_degree = 4
            integration_order = space_degree + 3

            volume_fraction = 0.6
            penalty_factor = 3.0

            # 'lagrange_interpolation_point', 'element'
            density_location = 'element'
            relative_density = volume_fraction

            optimizer_algorithm = 'mma'  # 'oc', 'mma'
            max_iterations = 500

            filter_type = 'density' # 'none', 'sensitivity', 'density'

            # rmin = 1.5
            # rmin = 1.25
            # rmin = 1.0
            rmin = 0.75
            # rmin = 0.5
            # rmin = 0.25

            from soptx.model.mbb_beam_2d import MBBBeam2dData
            pde = MBBBeam2dData(
                                domain=domain,
                                T=T, E=E, nu=nu,
                                enable_logging=False
                            )
        
        pde.init_mesh.set(mesh_type)
        analysis_mesh = pde.init_mesh(nx=nx, ny=ny)

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
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=analysis_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        
        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)

        from soptx.regularization.filter import Filter

        filter_regularization = Filter(
                                    mesh=opt_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    integration_order=integration_order,
                                    interpolation_order=1,
                                )

        if optimizer_algorithm == 'mma': 

            from soptx.optimization.mma_optimizer import MMAOptimizer
            optimizer = MMAOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': max_iterations,
                                'tolerance': 1e-2,
                            }
                        )
            design_variables_num = rho.shape[0]
            constraints_num = 1
            optimizer.options.set_advanced_options(
                                    m=constraints_num,
                                    n=design_variables_num,
                                    xmin=bm.zeros((design_variables_num, 1)),
                                    xmax=bm.ones((design_variables_num, 1)),
                                    a0=1,
                                    a=bm.zeros((constraints_num, 1)),
                                    c=1e4 * bm.ones((constraints_num, 1)),
                                    d=bm.zeros((constraints_num, 1)),
                                )
        
        elif optimizer_algorithm == 'oc':

            from soptx.optimization.oc_optimizer import OCOptimizer
            optimizer = OCOptimizer(
                                objective=compliance_objective,
                                constraint=volume_constraint,
                                filter=filter_regularization,
                                options={
                                    'max_iterations': max_iterations,
                                    'tolerance': 1e-2,
                                }
                            )
            optimizer.options.set_advanced_options(
                                        move_limit=0.2,
                                        damping_coef=0.5,
                                        initial_lambda=1e9,
                                        bisection_tol=1e-3
                                    )

        self._log_info(f"开始密度拓扑优化, "
                       f"模型名称={pde.__class__.__name__}, "
                       f"网格类型={mesh_type},  " 
                       f"密度类型={density_location}, " 
                       f"密度网格尺寸={opt_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={analysis_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        rho_opt, history = optimizer.optimize(density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_p")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=opt_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
        

if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_mbb_2d')
    test.run()