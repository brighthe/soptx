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

    @variantmethod('test_subsec4_2_2')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 30.0, 0, 10.0]
        E, nu = 1.0, 0.3
        P = -1.0
        plane_type = 'plane_stress' 

        nx, ny = 30, 10
        mesh_type = 'uniform_quad'

        space_degree = 2
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        volume_fraction = 0.6
        penalty_factor = 3.0

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element'
        sub_density_element = 16

        relative_density = volume_fraction

        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500
        change_tolerance = 1e-3
        use_penalty_continuation = True

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.2
        # rmin = 0.3

        from soptx.model.mbb_beam_2d import HalfMBBBeamRight2d
        pde = HalfMBBBeamRight2d(
                            domain=domain,
                            P=P, E=E, nu=nu,
                            plane_type=plane_type
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
            
        self._log_info(f"开始密度拓扑优化, "
        f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
        f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
        f"体积约束={volume_fraction}, "
        f"网格类型={mesh_type},  " 
        f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, \n" 
        f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, \n"
        f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, "
        f"收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
        f"过滤类型={filter_type}, 过滤半径={rmin}, ")
            
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec4_")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    

    @run.register('test_subsec4_6_2')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 10.0]
        E, nu = 1.0, 0.3
        P = -2.0
        plane_type = 'plane_stress' 

        # nx, ny = 60, 10
        # nx, ny = 120, 20
        # nx, ny = 240, 40
        nx, ny = 480, 80
        mesh_type = 'uniform_quad'

        space_degree = 8
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        volume_fraction = 0.6
        penalty_factor = 3.0

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element_multiresolution'
        sub_density_element = 64

        relative_density = volume_fraction

        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 1000
        change_tolerance = 1e-3
        use_penalty_continuation = True

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        # rmin = 1.2
        # rmin = 1.0
        # rmin = 0.75
        # rmin = 0.5
        rmin = 0.25

        from soptx.model.mbb_beam_2d import MBBBeam2d
        pde = MBBBeam2d(
                        domain=domain,
                        P=P, E=E, nu=nu,
                        plane_type=plane_type
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
            
        self._log_info(f"开始密度拓扑优化, "
        f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
        f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
        f"体积约束={volume_fraction}, "
        f"网格类型={mesh_type},  " 
        f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, \n" 
        f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, \n"
        f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, "
        f"收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
        f"过滤类型={filter_type}, 过滤半径={rmin}, ")
            

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec4_6_2")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    
    @run.register('test_subsec4_6_3_mma')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        #* MBB 
        # domain = [0, 60.0, 0, 10.0]
        # E, nu = 1.0, 0.3
        # P = -2.0
        # plane_type = 'plane_stress' 

        # nx, ny = 60, 10
        # mesh_type = 'uniform_quad'

        # from soptx.model.mbb_beam_2d import MBBBeam2d
        # pde = MBBBeam2d(
        #                 domain=domain,
        #                 P=P, E=E, nu=nu,
        #                 plane_type=plane_type
        #             )
        # volume_fraction = 0.6
        
        #* 对称 MBB
        domain = [0, 60.0, 0, 20.0]
        P = -1.0
        E, nu = 1.0, 0.3
        plane_type = 'plane_stress' 

        nx, ny = 60, 20
        mesh_type = 'uniform_quad'

        from soptx.model.mbb_beam_2d import HalfMBBBeamRight2d
        pde = HalfMBBBeamRight2d(
                            domain=domain,
                            P=P, E=E, nu=nu,
                            plane_type=plane_type,
                        )
        
        volume_fraction = 0.5
        
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格
        
        penalty_factor = 3.0

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element'
        sub_density_element = 16

        relative_density = volume_fraction

        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 1000
        change_tolerance = 1e-3
        use_penalty_continuation = True

        filter_type = 'projection' # 'none', 'sensitivity', 'density', 'projection'
        rmin = 1.8
        projection_config = {
                        'projection_type': 'exponential',  # 或 'exponential'
                        'beta': 1.0,                # 初始 beta
                        'beta_max': 512.0,          # 最大 beta 
                        'continuation_iter': 50,    # 每 50 步尝试更新
                        'eta': 0.5                  # 投影阈值
                    }

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
            
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    projection_params=projection_config,
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
            
        self._log_info(f"开始密度拓扑优化, "
        f"模型名称={pde.__class__.__name__}, \n"
        f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
        f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
        f"网格类型={mesh_type}, 空间阶数={space_degree}, \n" 
        f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
        f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移场自由度={analysis_tgdofs}, \n"
        f"体积分数约束={volume_fraction}, \n"
        f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, "
        f"收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
        f"过滤类型={filter_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec4_6_2")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


    @run.register('test_subsec4_6_3_oc')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 20.0]
        P = -1.0
        E, nu = 1.0, 0.3
        plane_type = 'plane_stress' 

        nx, ny = 60, 20
        mesh_type = 'uniform_quad'

        from soptx.model.mbb_beam_2d import HalfMBBBeamRight2d
        pde = HalfMBBBeamRight2d(
                            domain=domain,
                            P=P, E=E, nu=nu,
                            plane_type=plane_type,
                        )
        
        volume_fraction = 0.5
        
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格
        
        penalty_factor = 3.0

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element'
        sub_density_element = 16

        relative_density = volume_fraction

        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'projection' # 'none', 'sensitivity', 'density', 'projection'
        rmin = 1.8
        projection_config = {
                        'projection_type': 'exponential',  # 或 'exponential'
                        'beta': 1.0,                # 初始 beta
                        'beta_max': 512.0,          # 最大 beta 
                        'continuation_iter': 50,    # 每 50 步尝试更新
                        'eta': 0.5                  # 投影阈值
                    }

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
            
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    projection_params=projection_config,
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
                                'change_tolerance': change_tolerance,
                            }
                        )
        optimizer.options.set_advanced_options(
                                    move_limit=0.2,
                                    damping_coef=0.5,
                                    initial_lambda=1e9,
                                    bisection_tol=1e-3,
                                    design_variable_min=1e-9
                                )
            
        self._log_info(f"开始密度拓扑优化, "
        f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
        f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
        f"体积约束={volume_fraction}, "
        f"网格类型={mesh_type},  " 
        f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, \n" 
        f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, \n"
        f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, "
        f"收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
        f"过滤类型={filter_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec4_6_2")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    
if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec4_6_3_mma')
    rho_opt, history = test.run()