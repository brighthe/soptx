from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import (save_optimization_history, plot_optimization_history,
                                      save_history_data, load_history_data, plot_optimization_history_comparison)
from soptx.optimization.tools import OptimizationHistory


class DensityTopOptHuZhangTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test_linear_elastic_huzhang')
    def run(self) -> None:
        #* 算例 - 混合边界条件 - (非齐次位移 + 非齐次应力)
        # from soptx.model.linear_elastic_2d_hzmfem import HZmfemGeneralShearMix 
        # lam, mu = 1.0, 0.5
        # pde = HZmfemGeneralShearMix(lam=lam, mu=mu)

        #* 算例 - 混合边界条件 - (齐次位移 + 非齐次应力))
        from soptx.model.linear_elastic_2d_hzmfem import HDispNHStressMixedBdcc 
        lam, mu = 1.0, 0.5
        pde = HDispNHStressMixedBdcc(lam=lam, mu=mu)

        #* 第一类网格
        # pde.init_mesh.set('union_crisscross')
        # displacement_mesh = pde.init_mesh()
        # node = displacement_mesh.entity('node')
        # displacement_mesh.meshdata['corner'] = node[:-1]

        #* 第二类网格
        pde.init_mesh.set('uniform_crisscross_tri')
        nx, ny = 2, 2
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.add_subplot(111)
        # displacement_mesh.add_plot(axes)
        # displacement_mesh.find_node(axes, showindex=True)
        # displacement_mesh.find_edge(axes, showindex=True)
        # displacement_mesh.find_cell(axes, showindex=True)
        # plt.show()

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu,
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        space_degree = 2
        integration_order = space_degree*2 + 2
        use_relaxation = True # True, False
        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, \n"
                    f"网格类型={displacement_mesh.__class__.__name__}, 空间次数={space_degree}, 积分阶数={integration_order}, \n"
                    f"是否使用松弛={use_relaxation}")

        maxit = 5
        errorType = [
                    '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega, 0}$',
                    '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega, 0}$',
                    '$|| \\boldsymbol{\\div\\sigma} - \\boldsymbol{\\div\\sigma}_h||_{\\Omega, 0}$',
                    '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega, H(div)}$'
                    ]
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            N = 2**(i+1) 
            huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                                    disp_mesh=displacement_mesh,
                                                    pde=pde,
                                                    material=material,
                                                    interpolation_scheme=None,
                                                    space_degree=space_degree,
                                                    integration_order=integration_order,
                                                    use_relaxation=use_relaxation,
                                                    solve_method='scipy', # 'scipy', 'mumps'
                                                    topopt_algorithm=None,
                                                )
            
            uh_dof = huzhang_mfem_analyzer._tensor_space.number_of_global_dofs()
            sigma_dof = huzhang_mfem_analyzer._huzhang_space.number_of_global_dofs()
            NDof[i] = uh_dof + sigma_dof

            state = huzhang_mfem_analyzer.solve_state(rho_val=None)
            sigmah, uh = state['stress'], state['displacement']

            e_uh_l2 = displacement_mesh.error(u=uh, 
                                    v=pde.displacement_solution,
                                    q=integration_order) # 位移 L2 范数误差
            e_sigmah_l2 = displacement_mesh.error(u=sigmah, 
                                            v=pde.stress_solution, 
                                            q=integration_order) # 应力 L2 范数误差
            e_div_sigmah_l2 = displacement_mesh.error(u=sigmah.div_value, 
                                                v=pde.div_stress_solution, 
                                                q=integration_order) # 应力散度 L2 范数误差
            e_sigmah_hdiv = bm.sqrt(e_sigmah_l2**2 + e_div_sigmah_l2**2) # 应力 H(div) 范数误差

            h[i] = 1/N
            errorMatrix[0, i] = e_uh_l2
            errorMatrix[1, i] = e_sigmah_l2
            errorMatrix[2, i] = e_div_sigmah_l2
            errorMatrix[3, i] = e_sigmah_hdiv

            if i < maxit - 1:
                displacement_mesh.uniform_refine()
        
        import numpy as np
        with np.printoptions(formatter={'float': '{:.3e}'.format}):
            print(f"errorMatrix: {errorType}\n", errorMatrix)
        print("NDof:", NDof)
        print("order_uh_l2:\n", bm.round(bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]), 2))
        print("order_sigmah_l2:\n", bm.round(bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]), 2))
        print("order_div_sigmah_l2:\n", bm.round(bm.log2(errorMatrix[2, :-1] / errorMatrix[2, 1:]), 2))
        print("order_sigmah_hdiv:\n", bm.round(bm.log2(errorMatrix[3, :-1] / errorMatrix[3, 1:]), 2))
        # 转换为积分点应力
        stress_at_quad = huzhang_mfem_analyzer.extract_stress_at_quadrature_points(
                                                        stress_dof=sigmah, 
                                                        integration_order=integration_order
                                                    )  # (NC, NQ, NS)
                
        # 计算 von Mises 应力
        von_mises = material.calculate_von_mises_stress(stress_vector=stress_at_quad)

        von_mises_max = bm.max(von_mises, axis=1)
        displacement_mesh.celldata['von_mises'] = von_mises_max
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test")
        save_path.mkdir(parents=True, exist_ok=True)
        displacement_mesh.to_vtk(f"{save_path}/von_mises.vtu")

        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table

        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()
        print('------------------')


    @run.register('test_subsec5_6_2_lfem')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        #* 悬臂梁结构 cantilever_2d
        P = -1
        E, nu = 1, 0.3
        domain = [0, 80, 0, 50]
        plane_type = 'plane_stress' # plane_strain, plane_stress

        from soptx.model.cantilever_2d_lfem import CantileverCorner2d
        pde = CantileverCorner2d(
                    domain=domain,
                    P=P, 
                    E=E, nu=nu,
                    plane_type=plane_type,
                )
        nx, ny = 80, 50
        mesh_type = 'uniform_crisscross_tri'

        volume_fraction = 0.4

        space_degree = 1
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格

        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        
        # 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'
        solve_method = 'mumps'

        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 4.5

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
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                        'target_variables': ['E']
                                    },
                                )
        
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        analyzer = LagrangeFEMAnalyzer(
                                    disp_mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method=solve_method,
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
            
        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            )
        
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)

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
        
        fe_tspace = analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, \n"
                f"模型名称={pde.__class__.__name__} \n"
                f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 杨氏模量={pde.E}, 泊松比={pde.nu} \n"
                f"网格类型={mesh_type}, 密度类型={density_location}, 空间阶数={space_degree}, 积分次数={integration_order} \n" 
                f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, \n"
                f"分析算法={analyzer.__class__.__name__} \n" 
                f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移场自由度={fe_dofs} \n"
                f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
                f"收敛容限={change_tolerance}, 惩罚因子延续={use_penalty_continuation} \n"
                f"体积分数约束={volume_fraction}, 惩罚因子={penalty_factor}, 空材料杨氏模量={void_youngs_modulus} \n" 
                f"过滤类型={filter_type}, 过滤半径={rmin} ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_2_lfem")
        save_path.mkdir(parents=True, exist_ok=True)

        # save_history_data(history=history, save_path=str(save_path/'json'), label='k1')

        
        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    

    @run.register('test_subsec5_6_2_hzmfem')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        #* 悬臂梁结构 cantilever_2d (右下角受载)
        P = -1
        E, nu = 1, 0.3
        plane_type = 'plane_stress' # plane_strain, plane_stress
        mesh_type = 'uniform_crisscross_tri'

        # domain = [0, 80, 0, 40]
        # rmin = 2.0
        # from soptx.model.cantilever_2d_hzmfem import CantileverMiddle2d
        # pde = CantileverMiddle2d(
        #             domain=domain,
        #             P=P, 
        #             E=E, nu=nu,
        #             plane_type=plane_type,
        #             load_width=None,
        #         )
        # nx, ny = 80, 40
        # volume_fraction = 0.3

        # domain = [0, 80, 0, 50]
        # rmin = 4.5
        # from soptx.model.cantilever_2d_hzmfem import Cantilever2dCorner
        # pde = Cantilever2dCorner(
        #             domain=domain,
        #             P=P, 
        #             E=E, nu=nu,
        #             plane_type=plane_type,
        #             load_width=None,
        #         )
        # nx, ny = 80, 50
        # volume_fraction = 0.4

        domain = [0, 60, 0, 30]
        rmin = 2.4
        from soptx.model.simple_bridge_2d_hzfem import SimpleBridge2d
        pde = SimpleBridge2d(
                    domain=domain,
                    P=P, 
                    E=E, nu=nu,
                    plane_type=plane_type,
                    load_width=None,
                )
        nx, ny = 60, 30
        volume_fraction = 0.3

        space_degree = 3
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格

        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        
        # 'element'
        density_location = 'element'
        relative_density = volume_fraction

        solve_method = 'mumps'

        use_relaxation = True # True, False

        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'density' # 'none', 'sensitivity', 'density'

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
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                        'target_variables': ['E']
                                    },
                                )
        
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
            
        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            )
        
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        state_variable='sigma'
        compliance_objective = ComplianceObjective(analyzer=analyzer, state_variable=state_variable)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)

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
        
        self._log_info(f"开始密度拓扑优化, \n"
            f"模型名称={pde.__class__.__name__} \n"
            f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 杨氏模量={pde.E}, 泊松比={pde.nu} \n"
            f"网格类型={mesh_type}, 密度类型={density_location}, "
            f"网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape[0]} \n"
            f"应力空间阶数={analyzer.huzhang_space.p}, 应力场自由度={stress_dofs} \n"
            f"位移空间阶数={analyzer.tensor_space.p}, 位移场自由度={disp_dofs} \n"
            f"分析算法={analyzer.__class__.__name__}, 是否角点松弛={use_relaxation} \n" 
            f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
            f"收敛容限={change_tolerance}, 惩罚因子延续={use_penalty_continuation} \n"
            f"体积分数约束={volume_fraction}, 惩罚因子={penalty_factor}, 空材料杨氏模量={void_youngs_modulus} \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin} ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_2_hzmfem")
        save_path.mkdir(parents=True, exist_ok=True)

        # save_history_data(history=history, save_path=str(save_path/'json'), label='k1')
        
        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    

    @run.register('test_subsec5_6_3_lfem')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        #* 夹持板结构 clamped_beam_2d
        # p1, p2 = -2.0, -2.0
        # E, nu = 1, 0.4999 # 0.4999, 0.3
        # domain = [0, 80, 0, 40]
        # plane_type = 'plane_strain' # plane_strain, plane_stress

        # from soptx.model.clamped_beam_2d_lfem import ClampedBeam2d
        # pde = ClampedBeam2d(
        #             domain=domain,
        #             p1=p1, p2=p2,
        #             E=E, nu=nu,
        #             support_height_ratio=0.5,
        #             plane_type=plane_type,
        #         )
        # nx, ny = 80, 40
        # mesh_type = 'uniform_crisscross_tri'
        # # mesh_type = 'uniform_quad'

        # volume_fraction = 0.3

        #* 轴承装置结构 bearing_device_2d
        t = -8e-2
        E, nu = 1, 0.3 # 0.3, 0.4999
        domain = [0, 120, 0, 40]
        plane_type = 'plane_strain' # plane_strain, plane_stress

        from soptx.model.bearing_device_2d_lfem import BearingDevice2d
        pde = BearingDevice2d(
                            domain=domain,
                            t=t, E=E, nu=nu, 
                            plane_type=plane_type,
                            enable_logging=False
                        )

        nx, ny = 120, 40
        mesh_type = 'uniform_crisscross_tri' 

        volume_fraction = 0.35

        space_degree = 1
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格

        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        target_variables = ['E']
        
        # 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'
        solve_method = 'mumps'

        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.5

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
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                        'target_variables': target_variables
                                    },
                                )
        
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        analyzer = LagrangeFEMAnalyzer(
                                    disp_mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method=solve_method,
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
            
        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            )
        
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)

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
        
        fe_tspace = analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, \n"
                f"模型名称={pde.__class__.__name__} \n"
                f"平面类型={material.plane_type}, 外载荷类型={pde.load_type}, 杨氏模量={pde.E}, 泊松比={pde.nu} \n"
                f"网格类型={mesh_type}, 密度类型={density_location}, 空间阶数={space_degree} \n" 
                f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, \n"
                f"分析算法={analyzer.__class__.__name__} \n" 
                f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移场自由度={fe_dofs} \n"
                f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
                f"收敛容限={change_tolerance}, 惩罚因子延续={use_penalty_continuation} \n"
                f"体积分数约束={volume_fraction}, 惩罚因子={penalty_factor}, 空材料杨氏模量={void_youngs_modulus} \n" 
                f"过滤类型={filter_type}, 过滤半径={rmin} ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_3_lfem")
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history

    
    @run.register('test_subsec5_6_3_hzmfem')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        #* 夹持板结构 clamped_beam_2d
        # p1, p2 = -2.0, -2.0
        # E, nu = 1, 0.5
        # domain = [0, 80, 0, 40]
        # plane_type = 'plane_strain' # plane_stress, plane_strain

        # from soptx.model.clamped_beam_2d_hzmfem import ClampedBeam2d
        # pde = ClampedBeam2d(
        #             domain=domain,
        #             p1=p1, p2=p2,
        #             E=E, nu=nu,
        #             support_height_ratio=0.5,
        #             plane_type=plane_type,
        #         )
        # nx, ny = 80, 40
        # mesh_type = 'uniform_crisscross_tri'

        # volume_fraction = 0.3

        #* 轴承装置结构 bearing_device_2d
        t = -8e-2
        E, nu = 1, 0.3
        domain = [0, 120, 0, 40]
        plane_type = 'plane_strain' # plane_strain, plane_stress

        from soptx.model.bearing_device_2d_hzmfem import BearingDevice2d
        pde = BearingDevice2d(
                            domain=domain,
                            t=t, E=E, nu=nu, 
                            plane_type=plane_type,
                            enable_logging=False
                        )

        nx, ny = 120, 40
        mesh_type = 'uniform_crisscross_tri' 

        volume_fraction = 0.35

        space_degree = 1
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格

        # 'element'
        density_location = 'element'
        relative_density = volume_fraction

        solve_method = 'mumps' # 'scipy', 'mumps'

        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.5

        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        target_variables = ['E', 'nu']
        nu_penalty_factor = 1.0
        void_poisson_ratio = 0.3

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method=interpolation_method,
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': void_youngs_modulus,
                                        'target_variables': target_variables,
                                        'nu_penalty_factor': nu_penalty_factor,
                                        'void_poisson_ratio': void_poisson_ratio,
                                    },
                                )
        
        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            )
        
        use_relaxation = True # True, False
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
            
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        state_variable='sigma'
        compliance_objective = ComplianceObjective(analyzer=analyzer, state_variable=state_variable)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)

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
        
        fe_tspace = analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, \n"
                f"模型名称={pde.__class__.__name__} \n"
                f"平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 杨氏模量={pde.E}, 泊松比={pde.nu} \n"
                f"网格类型={mesh_type}, 密度类型={density_location}, 空间阶数={space_degree} \n" 
                f"密度空间阶数={analyzer.huzhang_space.p}, "
                f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape[0]} \n"
                f"位移空间阶数={analyzer.tensor_space.p}, "
                f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移场自由度={fe_dofs} \n"
                f"分析算法={analyzer.__class__.__name__}, 是否角点松弛={use_relaxation}, 状态变量={state_variable} \n" 
                f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
                f"收敛容限={change_tolerance}, 惩罚因子延续={use_penalty_continuation} \n"
                f"体积分数约束={volume_fraction}, 惩罚因子={penalty_factor}, 空材料杨氏模量={void_youngs_modulus} \n" 
                f"过滤类型={filter_type}, 过滤半径={rmin} ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho, is_store_stress=False)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_3_hzmfem")
        save_path.mkdir(parents=True, exist_ok=True)

        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        
        # save_optimization_history(mesh=design_variable_mesh, 
        #                         history=history, 
        #                         density_location=density_location,
        #                         save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history


if __name__ == "__main__":
    test = DensityTopOptHuZhangTest(enable_logging=True)

    # test_subsec5_6_3_hzmfem, test_linear_elastic_huzhang, test_subsec5_6_3_lfem, test_subsec5_6_2_lfem, test_subsec5_6_2_hzmfem
    test.run.set('test_subsec5_6_2_hzmfem') 
    rho_opt, history = test.run()