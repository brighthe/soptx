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

    @variantmethod('test_subsec5_6_4')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 20.0]
        E, nu = 71000, 0.33
        P = -150
        plane_type = 'plane_stress' 

        volume_fraction = 0.5

        relative_density = volume_fraction - 0.1

        stress_limit = 350.0
        p_norm_factor = 8.0
        n_clusters = 10
        recluster_freq = 1

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        nx, ny = 60, 20
        mesh_type = 'uniform_crisscross_tri'

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 2

        solve_method = 'mumps'

        from soptx.model.mbb_beam_2d_hzmfem import HalfMBBBeamRight2d
        pde = HalfMBBBeamRight2d(
                            domain=domain,
                            P=P, E=E, nu=nu,
                            plane_type=plane_type,
                        )
        
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element'
        sub_density_element = 4
        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9

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
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                        'target_variables': ['E'],
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

        space_degree = 3
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格

        use_relaxation = True

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

        from soptx.optimization.compliance_objective import ComplianceObjective
        state_variable='sigma'
        compliance_objective = ComplianceObjective(analyzer=analyzer, state_variable=state_variable)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)

        from soptx.optimization.stress_constraint import StressConstraint
        stress_constraint = StressConstraint(analyzer=analyzer, 
                                            stress_limit=stress_limit,
                                            p_norm_factor=p_norm_factor,
                                            n_clusters=n_clusters,
                                            recluster_freq=recluster_freq,
                                        )

        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        
        # constraint = [volume_constraint, stress_constraint]
        constraint = [volume_constraint]
        from soptx.optimization.mma_optimizer import MMAOptimizer
        optimizer = MMAOptimizer(
                        objective=compliance_objective,
                        constraint=constraint,
                        filter=filter_regularization,
                        options={
                            'max_iterations': max_iterations,
                            'change_tolerance': change_tolerance,
                            'use_penalty_continuation': use_penalty_continuation,
                        }
                    )
        optimizer.options.set_advanced_options(
                                a0=1,
                                asymp_init=0.5,
                                asymp_incr=1.2,
                                asymp_decr=0.7,
                                move_limit=0.2,
                                albefa=0.1,
                                raa0=1e-5,
                                epsilon_min=1e-7,
                            )
        
        disp_space = analyzer.tensor_space
        disp_tgdofs = disp_space.number_of_global_dofs()
        stress_space = analyzer.huzhang_space
        stress_tgdofs = stress_space.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, "
            f"模型名称={pde.__class__.__name__}, \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
            f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
            f"网格类型={mesh_type} \n" 
            f"离散算法={analyzer.__class__.__name__}, 状态变量={state_variable}, \n"
            f"位移空间={disp_space.__class__.__name__}, 位移空间次数={disp_space.p}, 位移场自由度={disp_tgdofs}, \n"
            f"应力空间={stress_space.__class__.__name__}, 应力空间次数={stress_space.p}, 应力场自由度={stress_tgdofs}, \n"
            f"约束类型={[type(c).__name__ for c in optimizer._constraints]}, 体积分数上限={volume_fraction}, 应力上限={stress_limit}, \n"
            f"优化算法={optimizer_algorithm}, 最大迭代次数={max_iterations}, 初始构型={relative_density}, "
            f"收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho, is_store_stress=True)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec4_6_5_mbb_beam")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

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

    test.run.set('test_subsec5_6_4_L_bracket_stress')
    rho_opt, history = test.run()