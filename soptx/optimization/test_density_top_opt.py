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

    def set_pde(self, pde):
        self.pde = pde

    def set_init_mesh(self, meshtype: str, **kwargs):
        self.pde.init_mesh.set(meshtype)
        self.mesh = self.pde.init_mesh(**kwargs)

    def set_material(self, material):
        self.material = material

    def set_space_degree(self, space_degree: int):
        self.space_degree = space_degree

    def set_integrator_order(self, integration_order: int):
        self.integration_order = integration_order

    def set_assembly_method(self, method: str):
        self.assembly_method = method

    def set_solve_method(self, method: str):
        self.solve_method = method

    def set_volume_fraction(self, volume_fraction: float):
        self.volume_fraction = volume_fraction

    def set_relative_density(self, relative_density: float):
        self.relative_density = relative_density

    @variantmethod('test_nodal_variable')
    def run(self, parameter_type: str = 'element') -> Union[TensorLike, OptimizationHistory]:

        if parameter_type == "element":
            domain = [0, 30, 0, 10]
            T = -1.0
            E, nu = 1.0, 0.3

            # nx, ny = 30, 10
            nx, ny = 60, 20
            # nx, ny = 90, 30
            # mesh_type = 'uniform_quad'
            # mesh_type = 'uniform_aligned_tri'
            mesh_type = 'uniform_crisscross_tri'

            space_degree = 2
            integration_order = space_degree + 1

            # 'lagrange_interpolation_point', 'berstein_interpolation_point',
            density_location = 'lagrange_interpolation_point'
            density_interpolation_order = 1
            relative_density = 0.5

            volume_fraction = 0.5
            penalty_factor = 3.0

            optimizer_algorithm = 'mma'  # 'mma', 'mma'
            max_iterations = 500

            filter_type = 'none' # 'none', 'sensitivity', 'density'

            domain_length = domain[1] - domain[0]
            rmin = 1.5 * (domain_length / nx)

            from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
            pde = HalfMBBBeam2dData(
                                domain=domain,
                                T=T, E=E, nu=nu,
                                enable_logging=False
                            )

        elif parameter_type == "nodal":
            domain = [0, 4, 0, 2]

            T = -1.0
            E, nu = 1000.0, 0.3

            # 'uniform_tri', 'uniform_quad', 'uniform_hex'
            nx, ny = 120, 60
            # mesh_type = 'uniform_quad'
            mesh_type = 'uniform_tri'

            space_degree = 2
            integration_order = space_degree + 1
            
            # 'lagrange_interpolation_point', 'berstein_interpolation_point', shepard_interpolation_point, 'element'
            density_location = 'berstein_interpolation_point'  
            density_interpolation_order = 2
            relative_density = 0.5

            volume_fraction = 0.5
            penalty_factor = 3.0

            optimizer_algorithm = 'mma'  # 'oc', 'mma'
            max_iterations = 300

            filter_type = 'none' # 'none', 'sensitivity', 'density'

            domain_length = pde.domain[1] - pde.domain[0]
            rmin = 1.5 * (domain_length / nx)

            from soptx.model.cantilever_2d import CantileverBeamMiddle2dData
            pde = CantileverBeamMiddle2dData(
                                domain=domain,
                                T=T, E=E, nu=nu,
                                enable_logging=False
                            )

        pde.init_mesh.set(mesh_type)
        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

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
                                                interpolation_order=density_interpolation_order,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
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

            # 设置高级参数 (可选)
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
            # 设置高级参数 (可选)
            optimizer.options.set_advanced_options(
                                        move_limit=0.2,
                                        damping_coef=0.5,
                                        initial_lambda=1e9,
                                        bisection_tol=1e-3
                                    )

        self._log_info(f"开始密度拓扑优化, "
                       f"网格类型={mesh_type}, 密度类型={density_location}, " 
                       f"密度网格尺寸={opt_mesh.number_of_cells()}, 密度插值次数={density_interpolation_order}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={fe_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={fe_dofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/nodal_variable_{density_location}")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=opt_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
        
    
    @run.register('test_gauss_variable')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 30, 0, 10]
        T = -1.0
        E, nu = 1.0, 0.3

        # nx, ny = 30, 10
        nx, ny = 60, 20
        # nx, ny = 90, 30
        mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_aligned_tri'
        # mesh_type = 'uniform_crisscross_tri'

        space_degree = 2
        integration_order = space_degree + 1

        # 'gauss_integration_point', 'element'
        density_location = 'gauss_integration_point'
        relative_density = 0.5
        denisty_integration_order = integration_order

        volume_fraction = 0.5
        penalty_factor = 3.0

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500

        filter_type = 'none' # 'none', 'sensitivity', 'density'

        domain_length = domain[1] - domain[0]
        rmin = 1.5 * (domain_length / nx)


        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
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
                                                integration_order=denisty_integration_order,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        fe_sspace = lagrange_fem_analyzer.scalar_space
        fe_sldofs = fe_sspace.number_of_local_dofs()
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_tdofs = fe_tspace.number_of_global_dofs()
        
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
                       f"位移网格尺寸={fe_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={fe_tdofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        rho_opt, history = optimizer.optimize(density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_821")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=opt_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history

    @run.register('test_element_variable')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 30, 0, 10]
        T = -1.0
        E, nu = 1.0, 0.3

        # nx, ny = 30, 10
        # nx, ny = 60, 20
        nx, ny = 90, 30
        # mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_aligned_tri'
        mesh_type = 'uniform_crisscross_tri'

        space_degree = 5
        integration_order = space_degree + 1

        # 'lagrange_interpolation_point', 'element'
        density_location = 'element'
        relative_density = 0.5

        volume_fraction = 0.5
        penalty_factor = 3.0

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500

        filter_type = 'none' # 'none', 'sensitivity', 'density'

        domain_length = domain[1] - domain[0]
        rmin = 1.5 * (domain_length / nx)


        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
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
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        fe_sspace = lagrange_fem_analyzer.scalar_space
        fe_sldofs = fe_sspace.number_of_local_dofs()
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_tdofs = fe_tspace.number_of_global_dofs()
        
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
                       f"位移网格尺寸={fe_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={fe_tdofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        rho_opt, history = optimizer.optimize(density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_821")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=opt_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history

    @run.register('test_density_location')
    def run(self) -> Union[TensorLike, OptimizationHistory]:

        # 参数设置
        nx, ny = 120, 60
        density_location = 'element'  # 'density_subelement_gauss_point', 'gauss_integration_point', 'element'
        space_degree = 2
        integration_order = space_degree + 1

        volume_fraction = 0.5
        penalty_factor = 3.0
        filter_type = 'density' # 'none', 'density'
        
        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )

        pde.init_mesh.set('uniform_quad')

        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

        domain_length = pde.domain[1] - pde.domain[0]
        rmin = (0.04 * nx) / (domain_length / nx)
        # rmin = 1.2

        # 设置基础材料
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
                                                relative_density=self.relative_density,
                                                integration_order=integration_order,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
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
        self._log_info(f"开始密度拓扑优化, 密度类型={density_location}, 密度场自由度={rho.shape}, " 
                       f"网格尺寸={nx}*{ny}, 位移有限元空间阶数={space_degree}, 物理场自由度={fe_dofs}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = oc_optimizer.optimize(density_distribution=rho)

        if density_location == 'density_subelement_gauss_point':
            # 直接绘制高斯点密度
            # from soptx.interpolation.tools import plot_gauss_integration_point_density
            # plot_gauss_integration_point_density(mesh=opt_mesh, rho_gip=rho_opt)

            opt_mesh = pde.init_mesh(nx=nx*integration_order, ny=ny*integration_order)

        # 保存结果
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/density_type_{density_location}")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(opt_mesh, 
                                history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    

    @run.register('test_volume_constraint')
    def run(self) -> None:
        density_location = 'density_subelement_gauss_point'  # 'lagrange_interpolation_point', 'gauss_integration_point', 'element'
        volume_fraction = 0.5

        nx, ny = 30, 10

        space_degree = 2
        integration_order = space_degree + 1

        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )

        pde.init_mesh.set('uniform_quad')
        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

        pde.init_mesh.set('uniform_quad')
        opt_mesh = pde.init_mesh(nx=nx, ny=ny)

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
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )
        
        rho = interpolation_scheme.setup_density_distribution(
                                        mesh=opt_mesh,
                                        relative_density=volume_fraction,
                                    )
    
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        
        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)
        test = volume_constraint._compute_volume(physical_density=rho)
        test1 = volume_constraint._manual_differentiation(physical_density=rho)

        print("---------------")


    @run.register('test_regularization')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        
        density_location = 'density_subelement_gauss_point'  # 'lagrange_interpolation_point', 'gauss_integration_point', 'element'
        volume_fraction = 0.5

        nx, ny = 30, 10
        filter_type = 'density'

        space_degree = 2
        integration_order = space_degree + 1

        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        
        pde.init_mesh.set('uniform_quad')
        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

        pde.init_mesh.set('uniform_quad')
        opt_mesh = pde.init_mesh(nx=nx, ny=ny)

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
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=volume_fraction,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)

        domain_length = pde.domain[1] - pde.domain[0]
        domain_height = pde.domain[3] - pde.domain[2]
        rmin = (0.04 * nx) / (domain_length / nx)

        from soptx.regularization.matrix_builder import FilterMatrixBuilder
        filter_matrix = FilterMatrixBuilder(
                                    mesh=opt_mesh,
                                    rmin=rmin,
                                    density_location=density_location,
                                    integration_order=integration_order,
                                    interpolation_order=1,
                                )
        
        hx, hy = domain_length / nx, domain_height / ny
        H, Hs = filter_matrix._compute_weighted_matrix_general(rmin=rmin, domain=pde.domain)


        from soptx.regularization.filter import Filter
        Filter = Filter(
                        mesh=opt_mesh,
                        filter_type=filter_type,
                        rmin=rmin,
                        density_location=density_location,
                        integration_order=integration_order,
                        interpolation_order=1,
                    )
        iw = Filter._compute_integration_weights()
        print("---------------------")
    

    @run.register('test_matlab_code')
    def run(self, paramter_type: int = 0) -> Union[TensorLike, OptimizationHistory]:
        """ 测试与 matlab 代码是否一致 """

        if paramter_type == 0:
            domain = [0, 60, 0, 20, 0, 4]
            nx, ny, nz = 60, 20, 4
            mesh_type = 'uniform_hex'

            density_location = 'element'  
            space_degree = 1
            integration_order = space_degree + 1

            relative_density = 0.3
            volume_fraction = 0.3
            penalty_factor = 3.0
    
            optimizer_algorithm = 'mma'  # 'oc', 'mma'
    
            filter_type = 'sensitivity' # 'none', 'density', 'sensitivity'
            
            # 设置 pde
            from soptx.model.cantilever_3d import CantileverBeam3dData
            pde = CantileverBeam3dData(
                                domain=domain,
                                T=-1.0, E=1.0, nu=0.3,
                                enable_logging=False
                            )

            pde.init_mesh.set(mesh_type)

            fe_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)
            opt_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)

            rmin = 1.5
        
        elif paramter_type == 1:
            domain = [0, 60, 0, 20]
            nx, ny = 60, 20
            density_location = 'element'  
            space_degree = 1
            integration_order = space_degree + 1

            relative_density = 0.5
            volume_fraction = 0.5
            penalty_factor = 3.0
            filter_type = 'sensitivity' # 'none', 'density', 'sensitivity'
            optimizer_algorithm = 'mma'  # 'oc', 'mma'
            
            # 设置 pde
            from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
            pde = HalfMBBBeam2dData(
                                domain=domain,
                                T=-1.0, E=1.0, nu=0.3,
                                enable_logging=False
                            )

            pde.init_mesh.set('uniform_quad')

            fe_mesh = pde.init_mesh(nx=nx, ny=ny)
            opt_mesh = pde.init_mesh(nx=nx, ny=ny)

            domain_length = pde.domain[1] - pde.domain[0]
            rmin = (0.04 * nx) / (domain_length / nx)

        # 设置基础材料
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

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=relative_density,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                        mesh=fe_mesh,
                                        pde=pde,
                                        material=material,
                                        interpolation_scheme=interpolation_scheme,
                                        space_degree=space_degree,
                                        integration_order=integration_order,
                                        assembly_method='standard',
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                    )
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, 
                                            volume_fraction=volume_fraction)

        from soptx.regularization.filter import Filter
        
        filter_regularization = Filter(
                                    mesh=opt_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    integration_order=integration_order,
                                    interpolation_order=1,
                                )
        
        if optimizer_algorithm == 'oc':
            from soptx.optimization.oc_optimizer import OCOptimizer
            optimizer = OCOptimizer(
                                objective=compliance_objective,
                                constraint=volume_constraint,
                                filter=filter_regularization,
                                options={
                                    'max_iterations': 200,
                                    'tolerance': 1e-2,
                                }
                            )
            # 设置高级参数 (可选)
            optimizer.options.set_advanced_options(
                                        move_limit=0.2,
                                        damping_coef=0.5,
                                        initial_lambda=1e9,
                                        bisection_tol=1e-3
                                    )
        elif optimizer_algorithm == 'mma':
            from soptx.optimization.mma_optimizer import MMAOptimizer
            optimizer = MMAOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': 200,
                                'tolerance': 1e-2,
                            }
                        )
            # 设置高级参数 (可选)
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
        
        self._log_info(f"开始密度拓扑优化, "
                       f"网格类型={mesh_type}, 密度类型={density_location}, " 
                       f"密度网格尺寸={opt_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={fe_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={fe_dofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(density_distribution=rho)

        # 保存结果
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/matlab_{density_location}")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(opt_mesh, 
                                history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history


if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)
    
    p = 2
    q = p+1
    test.set_space_degree(p)
    test.set_integrator_order(q)
    test.set_assembly_method('standard')
    test.set_solve_method('mumps')
    test.set_volume_fraction(0.5)
    test.set_relative_density(0.5)

    # test.run.set('test_nodal_variable')
    # rho_opt, history = test.run()

    # test.run.set('test_element_variable')
    # rho_opt, history = test.run()
    
    test.run.set('test_gauss_variable')
    rho_opt, history = test.run()


    # test.run.set('test_matlab_code')
    # rho, history = test.run()
    
    # test.run.set('test_volume_constraint')
    # test.run()

    # test.run.set('test_regularization')
    # rho, history = test.run()


