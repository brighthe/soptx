from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import save_optimization_history, plot_optimization_history
from soptx.optimization.tools import OptimizationHistory


class DensityTopOptHuZhangTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test_linear_elastic_huzhang')
    def run(self) -> None:
        #* 算例 - 纯位移边界条件 - 零剪切应力
        # from soptx.model.linear_elastic_2d_hzmfem import HZmfemZeroShearDirichlet
        # lam, mu = 1.0, 0.5
        # plane_type = 'plane_strain'
        # pde = HZmfemZeroShearDirichlet(lam=lam, mu=mu, plane_type=plane_type)

        #* 算例 - 纯位移边界条件 
        # from soptx.model.linear_elastic_2d_hzmfem import HZmfemGeneralShearDirichlet
        # lam, mu = 1.0, 0.5
        # pde = HZmfemGeneralShearDirichlet(lam=lam, mu=mu)

        #* 算例 - 混合边界条件 - 零剪切应力
        # from soptx.model.linear_elastic_2d_hzmfem import HZmfemZeroShearMix
        # lam, mu = 1.0, 0.5
        # pde = HZmfemZeroShearMix(lam=lam, mu=mu)

        #* 算例 - 混合边界条件 - 一般剪切应力
        from soptx.model.linear_elastic_2d_hzmfem import HZmfemGeneralShearMix
        lam, mu = 1.0, 0.5
        pde = HZmfemGeneralShearMix(lam=lam, mu=mu)

        #* 第一类网格
        pde.init_mesh.set('union_crisscross')
        analysis_mesh = pde.init_mesh()
        node = analysis_mesh.entity('node')
        analysis_mesh.meshdata['corner'] = node[:-1]

        #* 第二类网格
        # pde.init_mesh.set('uniform_crisscross_tri')
        # nx, ny = 2, 2
        # analysis_mesh = pde.init_mesh(nx=nx, ny=ny)
        # node = analysis_mesh.entity('node')
        # analysis_mesh.meshdata['corner'] = pde.mark_corners(node)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.add_subplot(111)
        # analysis_mesh.add_plot(axes)
        # analysis_mesh.find_node(axes, showindex=True)
        # analysis_mesh.find_edge(axes, showindex=True)
        # analysis_mesh.find_cell(axes, showindex=True)
        # plt.show()

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu,
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        space_degree = 3
        integration_order = space_degree*2 + 2
        use_relaxation = False
        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                       f"网格类型={analysis_mesh.__class__.__name__}, 空间次数={space_degree}, 积分阶数={integration_order}, "
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
                                                    mesh=analysis_mesh,
                                                    pde=pde,
                                                    material=material,
                                                    space_degree=space_degree,
                                                    integration_order=integration_order,
                                                    use_relaxation=use_relaxation,
                                                    solve_method='mumps',
                                                    topopt_algorithm=None,
                                                    interpolation_scheme=None,
                                                )
            
            uh_dof = huzhang_mfem_analyzer._tensor_space.number_of_global_dofs()
            sigma_dof = huzhang_mfem_analyzer._huzhang_space.number_of_global_dofs()
            NDof[i] = uh_dof + sigma_dof

            sigmah, uh = huzhang_mfem_analyzer.solve_displacement(rho_val=None)

            e_uh_l2 = analysis_mesh.error(u=uh, 
                                    v=pde.displacement_solution,
                                    q=integration_order) # 位移 L2 范数误差
            e_sigmah_l2 = analysis_mesh.error(u=sigmah, 
                                            v=pde.stress_solution, 
                                            q=integration_order) # 应力 L2 范数误差
            e_div_sigmah_l2 = analysis_mesh.error(u=sigmah.div_value, 
                                                v=pde.div_stress_solution, 
                                                q=integration_order) # 应力散度 L2 范数误差
            e_sigmah_hdiv = bm.sqrt(e_sigmah_l2**2 + e_div_sigmah_l2**2) # 应力 H(div) 范数误差

            h[i] = 1/N
            errorMatrix[0, i] = e_uh_l2
            errorMatrix[1, i] = e_sigmah_l2
            errorMatrix[2, i] = e_div_sigmah_l2
            errorMatrix[3, i] = e_sigmah_hdiv

            if i < maxit - 1:
                analysis_mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)   
        print("NDof:", NDof)
        print("order_uh_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        print("order_sigmah_l2:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
        print("order_div_sigmah_l2:\n", bm.log2(errorMatrix[2, :-1] / errorMatrix[2, 1:]))
        print("order_sigmah_hdiv:\n", bm.log2(errorMatrix[3, :-1] / errorMatrix[3, 1:]))

        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table

        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()
        print('------------------')



    @run.register('test_bridge_2d')
    def run(self, analysis_method: str = 'lfem') -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 80, 0, 40]

        E = 1.0
        nu = 0.35   
        plane_type = 'plane_strain'  # 'plane_stress' or 'plane_strain'
        
        from soptx.model.bridge_2d import Bridge2dDoubleLoadData
        p1 = -2.0
        p2 = -2.0
        pde = Bridge2dDoubleLoadData(
                            domain=domain,
                            p1=p1, p2=p2,
                            E=E, nu=nu,
                            support_height_ratio=0.5,
                            plane_type=plane_type,
                        )

        volume_fraction = 0.35
        penalty_factor = 3.0
        
        # 'node', 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', , 'voigt', 
        assembly_method = 'voigt'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500
        tolerance = 1e-2

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.25

        # 'uniform_tri', 'uniform_quad', 'uniform_hex'
        nx, ny = 80, 40
        mesh_type = 'uniform_quad'

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        pde.init_mesh.set(mesh_type)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

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
        
        if analysis_method == 'lfem':
            space_degree = 1
            # 张量网格
            integration_order = space_degree + 1
            from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
            lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=space_degree,
                                        integration_order=integration_order,
                                        assembly_method=assembly_method,
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                        interpolation_scheme=interpolation_scheme,
                                    )
            
            analyzer = lagrange_fem_analyzer

        elif analysis_method == 'huzhangfem':
            # TODO 支持低阶 1 <=p <= d
            huzhang_space_degree = 3
            integration_order = huzhang_space_degree**2 + 2
            from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
            huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=huzhang_space_degree,
                                        integration_order=integration_order,
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                        interpolation_scheme=interpolation_scheme,
                                    )
            
            analyzer = huzhang_mfem_analyzer

            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                )

            fe_tspace = lagrange_fem_analyzer.tensor_space
            fe_dofs = fe_tspace.number_of_global_dofs()
            
            from soptx.optimization.compliance_objective import ComplianceObjective
            compliance_objective = ComplianceObjective(analyzer=analyzer)

            from soptx.optimization.volume_constraint import VolumeConstraint
            volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)

            from soptx.regularization.filter import Filter
            filter_regularization = Filter(
                                        mesh=design_variable_mesh,
                                        filter_type=filter_type,
                                        rmin=rmin,
                                        density_location=density_location,
                                        integration_order=integration_order,
                                    )

            from soptx.optimization.mma_optimizer import MMAOptimizer
            optimizer = MMAOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': max_iterations,
                                'tolerance': tolerance,
                                'use_penalty_continuation': False,
                            }
                        )

            # 设置高级参数 (可选)
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
                       f"分析数值方法={analyzer.__class__.__name__}, "
                       f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                       f"杨氏模量={pde.E}, 泊松比={pde.nu}, "
                       f"离散方法={analysis_method}, "
                       f"网格类型={mesh_type}, 密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={fe_dofs}, "
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_bridge_2d")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    
    @run.register('test_subsec5_6_2_bearing_device_2d')
    def run(self, analysis_method: str = 'lfem') -> Union[TensorLike, OptimizationHistory]:
        t = -1.8e-2
        E, nu = 1, 0.5
        domain = [0, 60, 0, 40]
        plane_type = 'plane_stress'

        from soptx.model.bearing_device_2d import HalfBearingDeviceLeft2d
        pde = HalfBearingDeviceLeft2d(
                            domain=domain,
                            t=t, E=E, nu=nu, 
                            plane_type=plane_type,
                            enable_logging=False
                        )

        nx, ny = 60, 40
        mesh_type = 'uniform_crisscross_tri' 

        # t = -1.8e-2
        # E, nu = 1, 0.5
        # domain = [0, 120, 0, 40]
        # plane_type = 'plane_stress'
        
        # from soptx.model.bearing_device_2d import BearingDevice2d
        # pde = BearingDevice2d(
        #                     domain=domain,
        #                     t=t, E=E, nu=nu, 
        #                     plane_type=plane_type,
        #                     enable_logging=False
        #                 )

        # nx, ny = 120, 40
        # mesh_type = 'uniform_crisscross_tri'

        volume_fraction = 0.35
        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        
        # 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', , 'voigt', 
        assembly_method = 'standard'

        optimizer_algorithm = 'mma'
        max_iterations = 500
        change_tolerance = 1e-2
        use_penalty_continuation = True

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 2.5

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
        
        if analysis_method == 'lfem':
            space_degree = 1
            integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格
            from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
            lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=space_degree,
                                        integration_order=integration_order,
                                        assembly_method=assembly_method,
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                        interpolation_scheme=interpolation_scheme,
                                    )
            
            analyzer = lagrange_fem_analyzer

        elif analysis_method == 'huzhangfem':
            # TODO 支持低阶 1 <=p <= d
            huzhang_space_degree = 3
            integration_order = huzhang_space_degree**2 + 2
            from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
            huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=huzhang_space_degree,
                                        integration_order=integration_order,
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                        interpolation_scheme=interpolation_scheme,
                                    )
            
            analyzer = huzhang_mfem_analyzer

        design_variable_mesh = displacement_mesh
        d, rho = interpolation_scheme.setup_density_distribution(
                                                design_variable_mesh=design_variable_mesh,
                                                displacement_mesh=displacement_mesh,
                                                relative_density=relative_density,
                                            )
        
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=volume_fraction)



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
        
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
        self._log_info(f"开始密度拓扑优化, "
                f"分析数值方法={analyzer.__class__.__name__}, "
                f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, \n"
                f"杨氏模量={pde.E}, 泊松比={pde.nu}, "
                f"离散方法={analysis_method}, "
                f"网格类型={mesh_type}, 密度类型={density_location}, " 
                f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={fe_dofs}, \n"
                f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, "
                f"收敛容限={change_tolerance}, 惩罚因子延续={use_penalty_continuation}, \n" 
                f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_2_")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    
if __name__ == "__main__":
    test = DensityTopOptHuZhangTest(enable_logging=True)

    test.run.set('test_linear_elastic_huzhang')

    rho_opt, history = test.run()