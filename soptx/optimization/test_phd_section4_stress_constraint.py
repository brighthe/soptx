from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import (save_optimization_history, plot_optimization_history, 
                                    save_history_data,)
from soptx.optimization.tools import OptimizationHistory

class DensityTopOptTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = True, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test_subsec4_6_5_L_bracket_compliance')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        # 归一化尺寸
        # domain = [0, 1.0, 0, 1.0]
        # hole_domain = [0.4, 1.0, 0.4, 1.0]
        # rmin = 0.05
        # P = -2.0
        # 物理尺寸
        domain = [0, 200.0, 0, 200.0]            
        hole_domain = [80.0, 200.0, 80.0, 200.0]
        rmin = 10.0 
        P = -400.0

        # domain = [0, 1.0, 0, 1.0]
        # hole_domain = [0.4, 1.0, 0.4, 1.0]
        # P = -2.0
        E, nu = 7e4, 0.25
        plane_type = 'plane_stress' 

        nx, ny = 100, 100
        mesh_type = 'quad_threshold'

        from soptx.model.l_bracket_beam_lfem import LBracketBeam2d
        pde = LBracketBeam2d(
                            domain=domain,
                            hole_domain=hole_domain,
                            P=P, E=E, nu=nu,
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

        density_location = 'element'
        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                    },
                                )
        
        relative_density = 0.5
        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                )
        elif density_location in ['element_multiresolution']:
            sub_density_element = 4
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

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格
        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'fast'
        solve_method = 'mumps'

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        analyzer = LagrangeFEMAnalyzer(
                                disp_mesh=displacement_mesh,
                                pde=pde,
                                material=material,
                                interpolation_scheme=interpolation_scheme,
                                space_degree=space_degree,
                                integration_order=integration_order,
                                assembly_method=assembly_method,
                                solve_method=solve_method,
                                topopt_algorithm='density_based',
                            )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        objective = ComplianceObjective(analyzer=analyzer)

        volfrac = 0.31
        from soptx.optimization.volume_constraint import VolumeConstraint
        constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volfrac)

        filter_type = 'projection' 
        projection_config = {
                'continuation_strategy': 'multiplicative', # 指数型延续策略
                'projection_type': 'tanh',
                'beta': 1.0,           # 初始值 
                'beta_max': 512.0,     # 上限值
                'continuation_iter': 50, # 每 50 次迭代更新一次
                'beta_multiplier': 2.0  # 每次更新数值翻倍 (2.0) 
            }

        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    projection_params=projection_config,
                                )
        
        from soptx.optimization.mma_optimizer import MMAOptions
        options = MMAOptions(
                    # 循环控制
                    max_iterations=500,
                    change_tolerance=1e-2,
                    # MMA 渐近线控制
                    move_limit=0.2,
                    asymp_init=0.5,
                    asymp_incr=1.2,
                    asymp_decr=0.7,
                    # SIMP 连续化
                    use_penalty_continuation=False,
                )
        
        from soptx.optimization.mma_optimizer import MMAOptimizer
        optimizer = MMAOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter_regularization,
                        options=options,
                    )
        
        self._log_info(f"开始密度拓扑优化, 模型名称={pde.__class__.__name__} \n"
            f"目标函数={objective.__class__.__name__}, 约束函数={constraint.__class__.__name__}, 体积分数={constraint._volume_fraction} \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
            f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
            f"网格类型={mesh_type}, 空间阶数={space_degree}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho, is_store_stress=True)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec4_6_5_L_bracket_compliance")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, problem_type='compliance', save_path=str(save_path))

        return rho_opt, history
    
    
    @run.register('test_subsec4_6_5_L_bracket_stress')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        bm.set_backend('numpy') # numpy, pytorch
        # bm.set_default_device('cpu') # cpu, cuda
        device = 'cpu' # cpu, cuda
        # 归一化尺寸
        # domain = [0, 1.0, 0, 1.0]
        # hole_domain = [0.4, 1.0, 0.4, 1.0]
        # rmin = 0.05
        # P = -2.0
        # 物理尺寸
        domain = [0, 200.0, 0, 200.0]            
        hole_domain = [80.0, 200.0, 80.0, 200.0]
        rmin = 10.0 # 7.5, 10.0
        P = -400.0

        E, nu = 7e4, 0.25
        plane_type = 'plane_stress' 

        nx, ny = 50, 50
        # nx, ny = 100, 100
        # nx, ny = 200, 200
        mesh_type = 'quad_threshold'

        from soptx.model.l_bracket_beam_lfem import LBracketBeam2d
        pde = LBracketBeam2d(
                            domain=domain,
                            hole_domain=hole_domain,
                            P=P, E=E, nu=nu,
                            plane_type=plane_type,
                        )
        
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny, device=device)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            device=device,
                                        )

        density_location = 'element_multiresolution' # element, element_multiresolution
        interpolation_method = 'msimp'
        penalty_factor = 3.5
        void_youngs_modulus = 1e-9
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                    },
                                )
        
        # ! 这个地方就得取 0.5, 不能取 1.0
        relative_density = 0.5
        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                )
            assembly_method = 'fast'
        elif density_location in ['element_multiresolution']:
            sub_density_element = 4
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
            # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
            assembly_method = 'voigt_multiresolution'
            
        space_degree = 2
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        solve_method = 'mumps' # 'scipy', 'mumps'
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        analyzer = LagrangeFEMAnalyzer(
                                disp_mesh=displacement_mesh,
                                pde=pde,
                                material=material,
                                interpolation_scheme=interpolation_scheme,
                                space_degree=space_degree,
                                integration_order=integration_order,
                                assembly_method=assembly_method,
                                solve_method=solve_method,
                                topopt_algorithm='density_based',
                            )
                
        from soptx.optimization.volume_objective import VolumeObjective
        objective = VolumeObjective(analyzer=analyzer)

        stress_limit = 100.0
        from soptx.optimization.vanish_stress_constraint import VanishingStressConstraint
        constraint = VanishingStressConstraint(analyzer=analyzer, stress_limit=stress_limit)

        from soptx.optimization.al_mma_optimizer import ALMMMAOptions
        options = ALMMMAOptions(
                    # ALM 外层控制
                    max_al_iterations=150,
                    # max_al_iterations=50,
                    mma_iters_per_al=5,
                    change_tolerance=0.002,
                    stress_tolerance=0.003,
                    # 增广拉格朗日罚参数
                    mu_0=10.0,
                    mu_max=10000.0,
                    alpha=1.1,
                    lambda_0_init_val=0.0,
                    # MMA 渐近线控制
                    move_limit=0.15,
                    asymp_init=0.2,
                    asymp_incr=1.2,
                    asymp_decr=0.7,
                    osc=0.2,
                    # SIMP 连续化
                    use_penalty_continuation=True,
                )
        
        from soptx.optimization.augmented_lagrangian_objective import AugmentedLagrangianObjective
        augmented_lagrangian_objective = AugmentedLagrangianObjective(
                                            volume_objective=objective,
                                            stress_constraint=constraint,
                                            options=options,
                                        )

        filter_type = 'projection'
        # filter_type = 'projection' # 'none', 'sensitivity', 'density', 'projection'
        projection_config = {
                'continuation_strategy': 'additive',
                'projection_type': 'tanh',
                'beta': 1.0, 'beta_max': 10.0,
                'continuation_iter': 5, 'beta_increment': 1.0
            }
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    disp_mesh=displacement_mesh,
                                    projection_params=projection_config,
                                )

        from soptx.optimization.al_mma_optimizer import ALMMMAOptimizer
        optimizer = ALMMMAOptimizer(
                        al_objective=augmented_lagrangian_objective,
                        filter=filter_regularization,
                        options=options,
                        enable_logging=True,
                    )
        
        analysis_tspace = analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, "
            f"模型名称={pde.__class__.__name__} \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type} \n"
            f"杨氏模量={pde.E}, 泊松比={pde.nu} \n"
            f"网格类型={mesh_type}, 空间阶数={space_degree} \n"
            f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
            f"位移网格尺寸={displacement_mesh.number_of_cells()},  位移场自由度={analysis_tgdofs}, \n" 
            f"初始构型={relative_density}, 密度分布={density_location} \n"
            f"过滤类型={filter_regularization._filter_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        # # ===================== 后处理 =====================
        # from soptx.optimization.stress_post import StressPostProcessor

        # post = StressPostProcessor(
        #             analyzer=analyzer,
        #             stress_limit=100.0,         # 对应 fem.SLim
        #             solid_threshold=0.5,        # 对应 MATLAB: V > 0.5
        #             constraint_tolerance=0.01,  # 对应 MATLAB: tolerance = 0.01
        #         )
        # results = post.check_stress_constraints(rho_phys=rho_opt)
        # post.print_summary(results)
        # post.plot_density_and_stress(results)
        # post.plot_yield_surface(results)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec4_6_5_L_bracket_stress")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, problem_type='stress', save_path=str(save_path))

        return rho_opt, history
    

    @run.register('test_subsec4_6_5_cantilever_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 80, 0, 40]
        rmin = 3.5
        P = -400.0

        E, nu = 7e4, 0.25
        plane_type = 'plane_stress' 

        nx, ny = 80, 40
        mesh_type = 'uniform_quad'

        from soptx.model.cantilever_2d_lfem import Cantilever2d
        pde = Cantilever2d(
                    domain=domain,
                    P=P, 
                    E=E, nu=nu,
                    load_width=6,
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

        density_location = 'element' # element, element_multiresolution
        interpolation_method = 'msimp'
        penalty_factor = 3.5
        void_youngs_modulus = 1e-9
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                    },
                                )
        
        # ! 这个地方就得取 0.5, 不能取 1.0
        relative_density = 0.5
        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                )
            assembly_method = 'fast'
        elif density_location in ['element_multiresolution']:
            sub_density_element = 4
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
            # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
            assembly_method = 'voigt_multiresolution'
            
        space_degree = 3
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        solve_method = 'mumps'
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        analyzer = LagrangeFEMAnalyzer(
                                disp_mesh=displacement_mesh,
                                pde=pde,
                                material=material,
                                interpolation_scheme=interpolation_scheme,
                                space_degree=space_degree,
                                integration_order=integration_order,
                                assembly_method=assembly_method,
                                solve_method=solve_method,
                                topopt_algorithm='density_based',
                            )
                
        from soptx.optimization.volume_objective import VolumeObjective
        objective = VolumeObjective(analyzer=analyzer)

        stress_limit = 180.0
        from soptx.optimization.vanish_stress_constraint import VanishingStressConstraint
        constraint = VanishingStressConstraint(analyzer=analyzer, stress_limit=stress_limit)

        from soptx.optimization.al_mma_optimizer import ALMMMAOptions
        options = ALMMMAOptions(
                    # ALM 外层控制
                    max_al_iterations=150,
                    mma_iters_per_al=5,
                    change_tolerance=0.002,
                    stress_tolerance=0.003,
                    # 增广拉格朗日罚参数
                    mu_0=10.0,
                    mu_max=10000.0,
                    alpha=1.1,
                    lambda_0_init_val=0.0,
                    # MMA 渐近线控制
                    move_limit=0.15,
                    asymp_init=0.2,
                    asymp_incr=1.2,
                    asymp_decr=0.7,
                    osc=0.2,
                    # SIMP 连续化
                    use_penalty_continuation=False,
                )
        
        from soptx.optimization.augmented_lagrangian_objective import AugmentedLagrangianObjective
        augmented_lagrangian_objective = AugmentedLagrangianObjective(
                                            volume_objective=objective,
                                            stress_constraint=constraint,
                                            options=options,
                                        )

        filter_type = 'projection' # 'none', 'sensitivity', 'density', 'projection'
        projection_config = {
                'continuation_strategy': 'additive',
                'projection_type': 'tanh',
                'beta': 1.0, 'beta_max': 10.0,
                'continuation_iter': 5, 'beta_increment': 1.0
            }
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                    projection_params=projection_config,
                                )

        from soptx.optimization.al_mma_optimizer import ALMMMAOptimizer
        optimizer = ALMMMAOptimizer(
                        al_objective=augmented_lagrangian_objective,
                        filter=filter_regularization,
                        options=options,
                        enable_logging=True,
                    )
        
        self._log_info(f"开始密度拓扑优化, "
            f"模型名称={pde.__class__.__name__} \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type} \n"
            f"杨氏模量={pde.E}, 泊松比={pde.nu} \n"
            f"网格类型={mesh_type}, 空间阶数={space_degree} \n" 
            f"初始构型={relative_density}, 密度分布={density_location} \n"
            f"过滤类型={filter_regularization._filter_type}, 投影类型={filter_regularization._strategy.projection_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        # ===================== 后处理 =====================
        from soptx.optimization.stress_post import StressPostProcessor

        post = StressPostProcessor(
                    analyzer=analyzer,
                    stress_limit=stress_limit,         # 对应 fem.SLim
                    solid_threshold=0.5,        # 对应 MATLAB: V > 0.5
                    constraint_tolerance=0.01,  # 对应 MATLAB: tolerance = 0.01
                )
        results = post.check_stress_constraints(rho_phys=rho_opt)
        post.print_summary(results)
        post.plot_density_and_stress(results)
        post.plot_yield_surface(results)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec4_6_5_cantilever_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, problem_type='stress', save_path=str(save_path))

        return rho_opt, history
    
if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec4_6_5_L_bracket_stress')
    rho_opt, history = test.run()