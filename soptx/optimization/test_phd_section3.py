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

    @variantmethod('test_subsec_3_6_2_linear_elastic_2d')
    def run(self):
        # 三角函数真解 + 齐次 Dirichlet + 非齐次 Neumann
        lam, mu = 1.0, 0.5
        plane_type = 'plane_stress' # 'plane_stress' or 'plane_strain'

        mesh_type = 'uniform_quad'
        from soptx.model.linear_elastic_2d import TriMixHomoDirNHomoNeu2d
        pde = TriMixHomoDirNHomoNeu2d(domain=[0, 1, 0, 1], lam=lam, mu=mu, plane_type=plane_type)
        pde.init_mesh.set(mesh_type)
        nx, ny = 2, 2
        mesh = pde.init_mesh(nx=nx, ny=ny)
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu,
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        space_degree = 3
        integration_order = space_degree + 1

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                    f"空间次数={space_degree}, 积分阶数={integration_order}")

        maxit = 5
        errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{\\Omega, 0}$', 
                     '$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{\\Omega, 1}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer

        for i in range(maxit):
            N = 2**(i+1)

            lfa = LagrangeFEMAnalyzer(
                                    mesh=mesh,
                                    pde=pde, 
                                    material=material, 
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm=None,
                                    interpolation_scheme=None
                                )
                    
            uh = lfa.solve_displacement(rho_val=None)

            NDof[i] = lfa.tensor_space.number_of_global_dofs()

            e_l2 = mesh.error(uh, pde.disp_solution)
            e_h1 = mesh.error(uh.grad_value, pde.grad_disp_solution)

            h[i] = 1/N
            errorMatrix[0, i] = e_l2
            errorMatrix[1, i] = e_h1

            if i < maxit - 1:
                mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        print("order_h1:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))

        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return uh
        
    
    @run.register('test_mbb_2d_subsection_3_6_2')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        # 固定参数
        domain = [0, 60.0, 0, 20.0]
        P = -1.0
        E, nu = 1.0, 0.3

        # 测试参数
        # nx, ny = 60, 20
        nx, ny = 90, 30
        # nx, ny = 150, 50
        mesh_type = 'uniform_quad'

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        volume_fraction = 0.5
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = 0.5

        # 'standard', 'voigt'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500
        tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 2.4

        from soptx.model.mbb_beam_2d import HalfMBBBeam2d
        pde = HalfMBBBeam2d(
                            domain=domain,
                            P=P, E=E, nu=nu,
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
                                                
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
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
                       f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, 收敛容差={tolerance}, 惩罚因子连续化={use_penalty_continuation}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/section3_6_2_half_mbb")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    

    @run.register('test_subsec_3_6_3_half_mbb_right_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 20.0]
        P = -1.0
        E, nu = 1.0, 0.3

        nx, ny = 60, 20
        # nx, ny = 90, 30
        # nx, ny = 150, 50

        mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_aligned_tri'
        # mesh_type = 'uniform_crisscross_tri'

        # # 三角形网格(P1) + 单元密度
        # space_degree = 1
        # integration_order = space_degree  

        # # 三角形网格(Pk) + 单元密度
        # space_degree = 4
        # integration_order = space_degree*2 - 2 

        # # 三角形网格(P1) + 节点密度
        # space_degree = 1
        # integration_order = space_degree*2 + 2

        # 四边形网格 + 节点密度
        space_degree = 1
        integration_order = space_degree + 3

        # integration_order = space_degree + 1 # 单元密度 + 四边形网格
        # integration_order = space_degree + 2 # 节点密度 + 四边形网格
        # integration_order = space_degree  # 单元密度 + 三角形网格
        # integration_order = space_degree**2 + 2  # 单元密度 + 三角形网格

        volume_fraction = 0.5
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'node'
        relative_density = 0.5

        # 'standard', 'voigt'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 2.4

        from soptx.model.mbb_beam_2d import HalfMBBBeamRight2d
        pde = HalfMBBBeamRight2d(
                            domain=domain,
                            P=P, E=E, nu=nu,
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
                                                
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
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
                                'change_tolerance': change_tolerance,
                                'use_penalty_continuation': use_penalty_continuation,
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
                                    'change_tolerance': change_tolerance,
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
            f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 积分次数={integration_order}, 位移场自由度={analysis_tgdofs}, \n"
            f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
            f"设计变量变化收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec_3_6_3_half_mbb_right_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))
        return rho_opt, history


    @run.register('test_subsec_3_6_4_bearing_device_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        # domain = [0, 1200.0, 0, 400.0]
        t = -1.8e-2
        E, nu = 1, 0.5
        domain = [0, 120, 0, 40]
        # t = -1.8
        # E, nu = 100, 0.5

        plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'
        from soptx.model.bearing_device_2d import BearingDevice2d
        pde = BearingDevice2d(
                            domain=domain,
                            t=t, E=E, nu=nu, 
                            plane_type=plane_type,
                            enable_logging=False
                        )

        nx, ny = 120, 40
        # nx, ny = 60, 40
        mesh_type = 'uniform_quad'

        space_degree = 1
        # integration_order = space_degree + 1 # 单元密度 + 四边形网格
        integration_order = space_degree + 2 # 节点密度 + 四边形网格


        volume_fraction = 0.35
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'node'
        relative_density = volume_fraction

        # 'standard', 'voigt'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 2.5


        # from soptx.model.bearing_device_2d import HalfBearingDeviceLeft2d
        # pde = HalfBearingDeviceLeft2d(
        #                     domain=domain,
        #                     t=t, E=E, nu=nu, 
        #                     plane_type=plane_type,
        #                     enable_logging=False
        #                 )

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
                                                
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
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
                                'change_tolerance': change_tolerance,
                                'use_penalty_continuation': use_penalty_continuation,
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
                                    'change_tolerance': change_tolerance,
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
            f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 积分次数={integration_order}, 位移场自由度={analysis_tgdofs}, \n"
            f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
            f"设计变量变化收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec_3_6_4_bearing_device_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history

    @run.register('test_subsec_3_6_6_disp_inverter_upper_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 40.0, 0, 20.0]
        E, nu = 1.0, 0.3
        plane_type = 'plane_stress' 

        # 测试参数
        # nx, ny = 40, 20
        # nx, ny = 60, 30
        nx, ny = 100, 50
        mesh_type = 'uniform_quad'

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 2000
        tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 1.2

        from soptx.model.displacement_inverter_2d import DisplacementInverterUpper2d
        pde = DisplacementInverterUpper2d(
                        domain=domain,
                        f_in=1.0, f_out=-1.0,
                        k_in=0.1, k_out=0.1,
                        E=E, nu=nu,
                        plane_type=plane_type,
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
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
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

        from soptx.optimization.compliant_mechanism_objective import CompliantMechanismObjective
        compliant_mechanism_objective = CompliantMechanismObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)

        if optimizer_algorithm == 'mma': 
            from soptx.optimization.mma_optimizer import MMAOptimizer
            optimizer = MMAOptimizer(
                            objective=compliant_mechanism_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': max_iterations,
                                'tolerance': tolerance,
                                'use_penalty_continuation': use_penalty_continuation,
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
                                objective=compliant_mechanism_objective,
                                constraint=volume_constraint,
                                filter=filter_regularization,
                                options={
                                    'max_iterations': max_iterations,
                                    'tolerance': 1e-2,
                                }
                            )
            # 柔顺机构设计参数
            move_limit = 0.1
            damping_coef = 0.3
            initial_lambda = 1e5
            bisection_tol = 1e-4
            optimizer.options.set_advanced_options(
                                        move_limit=move_limit,
                                        damping_coef=damping_coef,
                                        initial_lambda=initial_lambda,
                                        bisection_tol=bisection_tol
                                    )

        self._log_info(f"开始密度拓扑优化, "
                       f"模型名称={pde.__class__.__name__}, "
                       f"体积约束={volume_fraction}, "
                       f"网格类型={mesh_type},  " 
                       f"密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, \n"
                       f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, 收敛容差={tolerance}, 惩罚因子连续化={use_penalty_continuation}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec_3_6_6_disp_inverter_upper_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history


    @run.register('test_cantilever_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        #* 矩形悬臂梁 (rectangle) */
        # domain = [0, 100.0, 0, 60.0]
        # nx, ny = 100, 60
        # volume_fraction = 0.4
        #* 方形悬臂梁 (square) */
        # domain = [0, 40.0, 0, 40.0]
        # nx, ny = 40, 40
        # volume_fraction = 0.35
        # from soptx.model.cantilever_2d import CantileverCorner2d
        # pde = CantileverCorner2d(
        #                     domain=domain,
        #                     p=-1.0, E=1.0, nu=0.3,
        #                     enable_logging=False
        #                 )
        #* 右端中点载荷悬臂梁 */
        # domain = [0, 120.0, 0, 60.0]
        # nx, ny = 120, 60
        domain = [0, 150.0, 0, 50.0]
        nx, ny = 150, 50
        volume_fraction = 0.5
        from soptx.model.cantilever_2d import CantileverRightMiddle2d
        pde = CantileverRightMiddle2d(
                            domain=domain,
                            p=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )

        mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_aligned_tri'
        # mesh_type = 'uniform_crisscross_tri'

        space_degree = 1
        integration_order = space_degree + 1

        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'node'
        relative_density = volume_fraction

        # 'standard', 'voigt'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 1000
        tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'none' # 'none', 'sensitivity', 'density'
        rmin = 6

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
                                                
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
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
                                'use_penalty_continuation': use_penalty_continuation,
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
                       f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                       f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
                       f"网格类型={mesh_type}, 密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, \n"
                       f"体积约束分数={volume_fraction}, "
                       f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, 收敛容差={tolerance}, 惩罚因子连续化={use_penalty_continuation}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_cantilever_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


    @run.register('test_cantilever_3d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 20.0, 0, 4.0]
        p = -1.0
        E, nu = 1.0, 0.3

        nx, ny, nz = 60, 20, 4
        mesh_type = 'uniform_hex'

        space_degree = 1
        integration_order = space_degree + 4 

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500
        tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 1.5

        from soptx.model.cantilever_3d import CantileverBeam3d
        pde = CantileverBeam3d(
                            domain=domain,
                            p=p, E=E, nu=nu,
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
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
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
                                'use_penalty_continuation': use_penalty_continuation,
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
                       f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                       f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
                       f"网格类型={mesh_type}, 密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, \n"
                       f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, 收敛容差={tolerance}, 惩罚因子连续化={use_penalty_continuation}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_cantilever_3d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history

if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec_3_6_4_bearing_device_2d')
    rho_opt, history = test.run()