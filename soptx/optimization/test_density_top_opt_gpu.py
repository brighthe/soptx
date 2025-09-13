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
            domain = [0, 60.0, 0, 10.0]
            T = -1.0
            E, nu = 1.0, 0.3

            nx, ny = 3, 2
            # nx, ny = 60, 10
            # nx, ny = 120, 20
            # nx, ny = 240, 40
            # nx, ny = 480, 80
            # nx, ny = 300, 50
            mesh_type = 'uniform_quad'
            # mesh_type = 'uniform_aligned_tri'
            # mesh_type = 'uniform_crisscross_tri'

            space_degree = 1
            integration_order = space_degree + 4

            volume_fraction = 0.6
            penalty_factor = 3.0

            # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
            density_location = 'element_multiresolution'
            sub_density_element = 4
            relative_density = volume_fraction

            # 'standard', 'voigt', 'voigt_multiresolution'
            assembly_method = 'voigt_multiresolution'

            optimizer_algorithm = 'mma'  # 'oc', 'mma'
            max_iterations = 30
            tolerance = 1e-3

            filter_type = 'density' # 'none', 'sensitivity', 'density'

            # rmin = 1.2
            # rmin = 1.25
            # rmin = 1.0
            rmin = 0.625
            # rmin = 0.5
            # rmin = 0.3125
            # rmin = 0.25
            # rmin = 0.15625

            from soptx.model.mbb_beam_2d import MBBBeam2dData
            pde = MBBBeam2dData(
                                domain=domain,
                                T=T, E=E, nu=nu,
                                enable_logging=False
                            )
            
        elif parameter_type == 'half_mbb_2d':
            domain = [0, 60.0, 0, 10.0]
            T = -1.0
            E, nu = 1.0, 0.3

            nx, ny = 60, 20
            # nx, ny = 90, 30
            # nx, ny = 120, 40
            # nx, ny = 240, 80
            mesh_type = 'uniform_quad'
            # mesh_type = 'uniform_aligned_tri'
            # mesh_type = 'uniform_crisscross_tri'

            space_degree = 1
            integration_order = space_degree + 3

            volume_fraction = 0.5
            penalty_factor = 3.0

            density_location = 'element'
            relative_density = 0.5

            # 'voigt', 'voigt_multi_resolution'
            assembly_method = 'voigt'

            optimizer_algorithm = 'mma'  # 'oc', 'mma'
            max_iterations = 500

            filter_type = 'density' # 'none', 'sensitivity', 'density'

            rmin = 2.4

            from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
            pde = HalfMBBBeam2dData(
                                domain=domain,
                                T=T, E=E, nu=nu,
                                enable_logging=False
                            )

        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

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


        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                ) 
        elif density_location in ['element_multiresolution']:
            import math
            sub_x, sub_y = int(math.sqrt(sub_density_element)), int(math.sqrt(sub_density_element))
            pde.init_mesh.set(mesh_type)
            design_variable_mesh = pde.init_mesh(nx=nx*sub_x, ny=ny*sub_y)
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    sub_density_element=sub_density_element,
                                                )
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    integration_order=integration_order,
                                                )
        elif density_location in ['node_multiresolution']:
            import math
            sub_x, sub_y = int(math.sqrt(sub_density_element)), int(math.sqrt(sub_density_element))
            pde.init_mesh.set(mesh_type)
            design_variable_mesh = pde.init_mesh(nx=nx*sub_x, ny=ny*sub_y)
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    sub_density_element=sub_density_element,
                                                    integration_order=integration_order,
                                                )
            
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )

        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()

        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)


        if optimizer_algorithm == 'mma': 

            from soptx.optimization.mma_optimizer import MMAOptimizer
            optimizer = MMAOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': max_iterations,
                                'tolerance': tolerance,
                                'use_penalty_continuation': True,
                            }
                        )
            design_variables_num = d.shape[0]
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
                       f"体积约束={volume_fraction}, "
                       f"网格类型={mesh_type},  " 
                       f"密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, "
                       f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, 收敛容差={tolerance}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_mtop2")
        save_path.mkdir(parents=True, exist_ok=True)

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


    @run.register('test_cantilever_3d')
    def run(self, parameter_type: str = 'cantilever_3d') -> Union[TensorLike, OptimizationHistory]:

        if parameter_type == 'cantilever_3d':
            domain = [0, 60, 0, 20, 0, 4]
            T = -1.0
            E, nu = 1.0, 0.3

            nx, ny, nz = 60, 20, 4
            mesh_type = 'uniform_hex'

            space_degree = 1
            integration_order = space_degree + 3

            volume_fraction = 0.3
            penalty_factor = 1.0

            # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
            density_location = 'element'
            sub_density_element = 4
            relative_density = volume_fraction

            # 'voigt', 'voigt_multi_resolution'
            assembly_method = 'voigt'

            optimizer_algorithm = 'mma'  # 'oc', 'mma'
            max_iterations = 100

            filter_type = 'density' # 'none', 'sensitivity', 'density'

            rmin = 1.5


            from soptx.model.cantilever_3d import CantileverBeam3dData
            pde = CantileverBeam3dData(
                                domain=domain,
                                T=T, E=E, nu=nu,
                                enable_logging=False
                            )

        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

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


        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                ) 
        elif density_location in ['element_multiresolution']:
            sub_x, sub_y = int(bm.sqrt(sub_density_element)), int(bm.sqrt(sub_density_element))
            pde.init_mesh.set(mesh_type)
            design_variable_mesh = pde.init_mesh(nx=nx*sub_x, ny=ny*sub_y)
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    sub_density_element=sub_density_element,
                                                )
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    integration_order=integration_order,
                                                )
        elif density_location in ['node_multiresolution']:
            sub_x, sub_y = int(bm.sqrt(sub_density_element)), int(bm.sqrt(sub_density_element))
            pde.init_mesh.set(mesh_type)
            design_variable_mesh = pde.init_mesh(nx=nx*sub_x, ny=ny*sub_y)
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    sub_density_element=sub_density_element,
                                                    integration_order=integration_order,
                                                )
            
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        # H = filter_regularization._H

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        # K = lagrange_fem_analyzer.assemble_stiff_matrix(rho_val=rho, sub_density_element=sub_density_element)    

        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()

        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)


        if optimizer_algorithm == 'mma': 

            from soptx.optimization.mma_optimizer import MMAOptimizer
            optimizer = MMAOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': max_iterations,
                                'tolerance': 1e-2,
                                # 'use_penalty_continuation': True,
                            }
                        )
            design_variables_num = d.shape[0]
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
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_p")
        save_path.mkdir(parents=True, exist_ok=True)

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


    @run.register('test_stop_mtop')
    def run(self) -> None:
            
        domain = [0, 60.0, 0, 10.0]
        T = -1.0
        E, nu = 1.0, 0.3

        # nx, ny = 3, 2
        # nx, ny = 60, 10
        nx, ny = 120, 20
        # nx, ny = 240, 40
        # nx, ny = 480, 80
        # nx, ny = 300, 50
        mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_aligned_tri'
        # mesh_type = 'uniform_crisscross_tri'

        space_degree = 1
        integration_order = space_degree + 1

        volume_fraction = 0.6
        penalty_factor = 3.0

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element'
        sub_density_element = 4
        relative_density = volume_fraction

        # 'standard', 'voigt', 'voigt_multiresolution'
        assembly_method = 'voigt'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 50
        tolerance = 1e-3

        filter_type = 'density' # 'none', 'sensitivity', 'density'

        # rmin = 1.2
        # rmin = 1.25
        # rmin = 1.0
        rmin = 0.625
        # rmin = 0.5
        # rmin = 0.3125
        # rmin = 0.25
        # rmin = 0.15625

        from soptx.model.mbb_beam_2d import MBBBeam2dData
        pde = MBBBeam2dData(
                            domain=domain,
                            T=T, E=E, nu=nu,
                            enable_logging=False
                        )
            
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme_stop = MaterialInterpolationScheme(
                                        density_location='element',
                                        interpolation_method='msimp',
                                        options={
                                            'penalty_factor': penalty_factor,
                                            'void_youngs_modulus': 1e-9,
                                            'target_variables': ['E']
                                        },
                                    )
        design_variable_mesh_stop = displacement_mesh
        d_stop, rho_stop = interpolation_scheme_stop.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh_stop,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            ) 

        interpolation_scheme_mtop = MaterialInterpolationScheme(
                                density_location='element_multiresolution',
                                interpolation_method='msimp',
                                options={
                                    'penalty_factor': penalty_factor,
                                    'void_youngs_modulus': 1e-9,
                                    'target_variables': ['E']
                                },
                            )
        sub_density_element = 4
        import math
        sub_x, sub_y = int(math.sqrt(sub_density_element)), int(math.sqrt(sub_density_element))
        pde.init_mesh.set(mesh_type)
        design_variable_mesh_mtop = pde.init_mesh(nx=nx*sub_x, ny=ny*sub_y)
        d_mtop, rho_mtop = interpolation_scheme_mtop.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh_mtop,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                                sub_density_element=sub_density_element,
                                            )

            
        from soptx.regularization.filter import Filter
        filter_regularization_stop = Filter(
                                    mesh=design_variable_mesh_stop,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location='element',
                                )
        filter_regularization_mtop = Filter(
                            mesh=design_variable_mesh_mtop,
                            filter_type=filter_type,
                            rmin=rmin,
                            density_location='element_multiresolution',
                        )
        

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer_stop = LagrangeFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme_stop,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='voigt',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        lagrange_fem_analyzer_mtop = LagrangeFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme_mtop,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='voigt_multiresolution',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        K_stop, KE_stop = lagrange_fem_analyzer_stop.assemble_stiff_matrix(rho_val=rho_stop)
        K_mtop, KE_mtop = lagrange_fem_analyzer_mtop.assemble_stiff_matrix(rho_val=rho_mtop)
        error_K = bm.sum(bm.abs(K_stop.toarray() - K_mtop.toarray()))
        error_KE = bm.sum(bm.abs(KE_stop[0] - KE_mtop[0]))
        uh_stop = lagrange_fem_analyzer_stop.solve_displacement(rho_val=rho_stop)
        uh_mtop = lagrange_fem_analyzer_mtop.solve_displacement(rho_val=rho_mtop)
        diff_K_stop = lagrange_fem_analyzer_stop.get_stiffness_matrix_derivative(rho_val=rho_stop)
        diff_K_mtop = lagrange_fem_analyzer_mtop.get_stiffness_matrix_derivative(rho_val=rho_mtop)
        
        error_uh = bm.linalg.norm(uh_stop[:] - uh_mtop[:])
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective_stop = ComplianceObjective(analyzer=lagrange_fem_analyzer_stop)
        compliance_objective_mtop = ComplianceObjective(analyzer=lagrange_fem_analyzer_mtop)
        c_stop = compliance_objective_stop.fun(density=rho_stop)
        c_mtop = compliance_objective_mtop.fun(density=rho_mtop)
        dc_stop = compliance_objective_stop.jac(density=rho_stop, diff_mode='manual')
        dc_mtop = compliance_objective_mtop.jac(density=rho_mtop, diff_mode='manual')
        dc_mtop_sum = bm.sum(dc_mtop, axis=1)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint_stop = VolumeConstraint(analyzer=lagrange_fem_analyzer_stop, volume_fraction=volume_fraction)
        volume_constraint_mtop = VolumeConstraint(analyzer=lagrange_fem_analyzer_mtop, volume_fraction=volume_fraction)
        v_stop = volume_constraint_stop.fun(density=rho_stop)
        v_mtop = volume_constraint_mtop.fun(density=rho_mtop)
        dv_stop = volume_constraint_stop.jac(density=rho_stop)
        dv_mtop = volume_constraint_mtop.jac(density=rho_mtop)

        from soptx.optimization.mma_optimizer import MMAOptimizer
        optimizer_stop = MMAOptimizer(
                        objective=compliance_objective_stop,
                        constraint=volume_constraint_stop,
                        filter=filter_regularization_stop,
                        options={
                            'max_iterations': max_iterations,
                            'tolerance': tolerance,
                            'use_penalty_continuation': True,
                        }
                    )
        design_variables_num = d_stop.shape[0]
        constraints_num = 1
        optimizer_stop.options.set_advanced_options(
                                m=constraints_num,
                                n=design_variables_num,
                                xmin=bm.zeros((design_variables_num, 1)),
                                xmax=bm.ones((design_variables_num, 1)),
                                a0=1,
                                a=bm.zeros((constraints_num, 1)),
                                c=1e4 * bm.ones((constraints_num, 1)),
                                d=bm.zeros((constraints_num, 1)),
                            )
        
        optimizer_mtop = MMAOptimizer(
                        objective=compliance_objective_mtop,
                        constraint=volume_constraint_mtop,
                        filter=filter_regularization_mtop,
                        options={
                            'max_iterations': max_iterations,
                            'tolerance': tolerance,
                            'use_penalty_continuation': True,
                        }   
                    )
        design_variables_num = d_mtop.shape[0]
        constraints_num = 1
        optimizer_mtop.options.set_advanced_options(
                                m=constraints_num,
                                n=design_variables_num,
                                xmin=bm.zeros((design_variables_num, 1)),
                                xmax=bm.ones((design_variables_num, 1)),
                                a0=1,
                                a=bm.zeros((constraints_num, 1)),
                                c=1e4 * bm.ones((constraints_num, 1)),
                                d=bm.zeros((constraints_num, 1)),
                            )

        self._log_info(f"开始密度拓扑优化, "
                       f"模型名称={pde.__class__.__name__}, "
                       f"体积约束={volume_fraction}, "
                       f"网格类型={mesh_type},  " 
                       f"密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, "
                       f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, 收敛容差={tolerance}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_480p2")
        save_path.mkdir(parents=True, exist_ok=True)

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_mbb_2d')
    # test.run.set('test_cantilever_3d')
    # test.run.set('test_stop_mtop')
    test.run()