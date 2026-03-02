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

    @variantmethod('test_subsec5_6_4_cantilever_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 80, 0, 40]
        P = -400.0

        E, nu = 7e4, 0.25
        plane_type = 'plane_stress' 

        nx, ny = 80, 40
        mesh_type = 'uniform_crisscross_tri'

        from soptx.model.cantilever_2d_hzmfem import CantileverMiddle2d
        pde = CantileverMiddle2d(
                    domain=domain,
                    P=P, 
                    E=E, nu=nu,
                    plane_type=plane_type,
                    load_width=6.0,  
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
        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            )

        space_degree = 3
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格
        use_relaxation = True
        solve_method = 'mumps'
        from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
        analyzer = HuZhangMFEMAnalyzer(
                                    disp_mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    use_relaxation=use_relaxation,
                                    solve_method=solve_method,
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
        stress_space = analyzer.huzhang_space
        stress_dofs = stress_space.number_of_global_dofs()

        disp_space = analyzer.tensor_space
        disp_dofs = disp_space.number_of_global_dofs()

        from soptx.optimization.volume_objective import VolumeObjective
        objective = VolumeObjective(analyzer=analyzer)

        stress_limit = 180.0
        from soptx.optimization.apparent_stress_constaint import ApparentStressConstraint
        constraint = ApparentStressConstraint(analyzer=analyzer, stress_limit=stress_limit)

        from soptx.optimization.al_mma_optimizer import ALMMMAOptions
        use_penalty_continuation = True
        max_al_iterations = 150
        max_iters_per_al = 5
        change_tolerance = 0.002
        mu_0 = 10.0
        mu_max = 10000.0
        options = ALMMMAOptions(
                    # ALM 外层控制
                    max_al_iterations=max_al_iterations,
                    mma_iters_per_al=max_iters_per_al,
                    change_tolerance=change_tolerance,
                    stress_tolerance=0.003,
                    # 增广拉格朗日罚参数
                    mu_0=mu_0,
                    mu_max=mu_max,
                    alpha=1.1,
                    lambda_0_init_val=0.0,
                    # MMA 渐近线控制
                    move_limit=0.15,
                    asymp_init=0.2,
                    asymp_incr=1.2,
                    asymp_decr=0.7,
                    osc=0.2,
                    # SIMP 连续化
                    use_penalty_continuation=use_penalty_continuation,
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
        rmin = 6.0
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
        
        self._log_info(f"开始密度拓扑优化, \n"
            f"模型名称={pde.__class__.__name__} \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 杨氏模量={pde.E}, 泊松比={pde.nu} \n"
            f"网格类型={mesh_type}, 密度类型={density_location}, "
            f"网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape[0]} \n"
            f"应力空间阶数={analyzer.huzhang_space.p}, 应力场自由度={stress_dofs} \n"
            f"位移空间阶数={analyzer.tensor_space.p}, 位移场自由度={disp_dofs} \n"
            f"分析算法={analyzer.__class__.__name__}, 是否角点松弛={use_relaxation} \n" 
            f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_al_iterations*max_iters_per_al}, "
            f"收敛容限={change_tolerance} \n" 
            f"惩罚因子={penalty_factor}, 惩罚因子延续={use_penalty_continuation}, 空材料杨氏模量={void_youngs_modulus} \n"
            f"应力约束={stress_limit}, 增广拉格朗日罚参数 mu_0={mu_0}, mu_max = {mu_max} \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin} ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_4_cantilever_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path), problem_type='stress')

        return rho_opt, history
    

    @run.register('test_subsec5_6_4_L_bracket_stress')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 1.0, 0, 1.0]
        hole_domain = [0.4, 1.0, 0.4, 1.0]
        P = -2.0
        E, nu = 7e4, 0.25
        plane_type = 'plane_stress' 

        # nx, ny = 10, 10
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
        
        relative_density = 0.5
        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                                )
            

        space_degree = 3
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格
        solve_method = 'mumps'

        from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
        analyzer = HuZhangMFEMAnalyzer(
                                    disp_mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    use_relaxation=True,
                                    solve_method=solve_method,
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
                
        from soptx.optimization.volume_objective import VolumeObjective
        objective = VolumeObjective(analyzer=analyzer)

        stress_limit = 100.0
        from soptx.optimization.stress_constraint import StressConstraint
        constraint = StressConstraint(analyzer=analyzer, stress_limit=stress_limit)

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
        rmin = 0.05
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
            f"模型名称={pde.__class__.__name__}, \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
            f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
            f"网格类型={mesh_type}, 空间阶数={space_degree}, \n" 
            f"过滤类型={filter_regularization._filter_type}, 投影类型={filter_regularization._strategy.projection_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        # ===================== 后处理 =====================
        from soptx.optimization.stress_post import StressPostProcessor

        post = StressPostProcessor(
                    analyzer=analyzer,
                    stress_limit=100.0,         # 对应 fem.SLim
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
        save_path = Path(f"{base_dir}/test_subsec5_6_4_L_bracket_stress")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, problem_type='stress', save_path=str(save_path))

        return rho_opt, history
                         
    
if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec5_6_4_cantilever_2d')
    rho_opt, history = test.run()