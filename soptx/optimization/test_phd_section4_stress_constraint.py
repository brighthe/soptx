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

    @variantmethod('test_subsec4_6_5_half_mbb_beam_compliance')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 20.0]
        E, nu = 1.0, 0.33
        P = -1
        plane_type = 'plane_stress' 

        volume_fraction = 0.5

        stress_limit = 350.0
        p_norm_factor = 8.0
        n_clusters = 10
        recluster_freq = 1

        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        nx, ny = 60, 20
        mesh_type = 'uniform_quad'

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 2

        solve_method = 'mumps'

        from soptx.model.mbb_beam_2d_lfem import HalfMBBBeamRight2d
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
        # stress_interpolation_method = 'power_law'
        # stress_penalty_factor = 0.5

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
                                    # stress_interpolation_method=stress_interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                        'target_variables': ['E'],
                                        # 'stress_penalty_factor': stress_penalty_factor,
                                    },
                                )

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'fast'

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
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
        
        relative_density = 1.0
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
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_objective import VolumeObjective
        # objective = VolumeObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=volume_fraction)

        # from soptx.optimization.stress_constraint import StressConstraint
        # stress_constraint = StressConstraint(analyzer=lagrange_fem_analyzer, 
        #                                     stress_limit=stress_limit,
        #                                     p_norm_factor=p_norm_factor,
        #                                     n_clusters=n_clusters,
        #                                     recluster_freq=recluster_freq,
        #                                 )

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
                        objective=objective,
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
                                raa0=1e-9, 
                                epsilon_min=1e-7,
                            )
        
        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, "
            f"模型名称={pde.__class__.__name__}, \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type} \n"
            f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
            f"分析算法={lagrange_fem_analyzer.__class__.__name__}, 网格类型={displacement_mesh.__class__.__name__}, "
            f"空间阶数={space_degree}, \n" 
            f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
            f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移场自由度={analysis_tgdofs} \n"
            f"目标函数={objective.__class__.__name__} \n"
            f"约束类型={[type(c).__name__ for c in optimizer._constraints]}, 体积分数上限={volume_fraction} \n"
            f"优化算法={optimizer.__class__.__name__} , 初始构型={relative_density}, 最大迭代次数={max_iterations}, "
            f"收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")

        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho, is_store_stress=True)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec4_6_5")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    
    
    @run.register('test_subsec4_6_5_L_bracket')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 1.0, 0, 1.0]
        hole_domain = [0.4, 1.0, 0.4, 1.0]
        P = -2.0
        E, nu = 7e4, 0.3
        plane_type = 'plane_stress' 

        nx, ny = 10, 10
        mesh_type = 'quad_threshold'
        # mesh_type = 'tri_threshold'

        from soptx.model.l_bracket_beam_lfem import LBracketBeam2d
        pde = LBracketBeam2d(
                            domain=domain,
                            hole_domain=hole_domain,
                            P=P, E=E, nu=nu,
                            plane_type=plane_type,
                        )
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)
        node = displacement_mesh.entity('node')
        right_edge = node[bm.abs(node[:, 0] - 1.0) < 1e-8]
        right_edge_in_range = right_edge[right_edge[:, 1] <= 0.4]
        print(f"右边缘节点数: {len(right_edge_in_range)}")
        print(right_edge_in_range)


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
                                        'target_variables': ['E'],
                                    },
                                )
        
        relative_density = 1.0
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
        
        uh = analyzer.solve_state(rho_val=rho, adjoint=False)


        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.gca()
        displacement_mesh.add_plot(axes)
        plt.show()      
        print("----------------")      
    
if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec4_6_5_L_bracket')
    rho_opt, history = test.run()