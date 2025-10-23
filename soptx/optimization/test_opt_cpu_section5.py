from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import save_optimization_history, plot_optimization_history
from soptx.optimization.tools import OptimizationHistory


class DensityTopOptHuZhangTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test_bridge_2d')
    def run(self, analysis_method) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 80, 0, 40]

        E = 1.0
        nu = 0.35   # 可压缩
        # nu = 0.5    # 不可压缩
        # nu = 0.4999 # 近不可压缩
        plane_type = 'plane_strain'  # 'plane_stress' or 'plane_strain'

        # from soptx.model.bridge_2d import Bridge2dSingleLoadData
        # pde = Bridge2dSingleLoadData(
        #                     domain=domain,
        #                     T=T, E=E, nu=nu,
        #                     support_height_ratio=0.5,
        #                     plane_type=plane_type,
        #                 )
        
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
        else:
            raise ValueError(f"不支持的密度位置类型: {density_location}")

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

        if optimizer_algorithm == 'mma': 

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


    @run.register('test_clamped_beam_2d')
    def run(self, analysis_method) -> Union[TensorLike, OptimizationHistory]:
        E = 30.0
        nu = 0.4   # 可压缩
        # nu = 0.5    # 不可压缩
        # nu = 0.49    # 近不可压缩
        # nu = 0.4999 # 近不可压缩
        plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'
        
        from soptx.model.clamped_beam_2d import HalfClampedBeam2D
        pde = HalfClampedBeam2D(
                            domain=[0, 80, 0, 20],
                            p=-1.5,
                            E=E, nu=nu,
                            plane_type=plane_type,
                        )
        
        volume_fraction = 0.4
        penalty_factor = 3.0
        
        # 'node', 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', , 'voigt', 
        assembly_method = 'voigt'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500
        tolerance = 1e-2
        state_variable = 'sigma'  # 'u', 'sigma'

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.25

        nx, ny = 80, 20
        # mesh_type = 'uniform_quad'
        mesh_type = 'uniform_crisscross_tri'

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
            integration_order = space_degree + 4
            from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
            analyzer = LagrangeFEMAnalyzer(
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
            space = analyzer.tensor_space
            TGDOF_uh = space.number_of_global_dofs()
            self._log_info(f"分析阶段参数, "
                f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
                f"离散方法={analysis_method}, 网格类型={mesh_type}, "
                f"状态变量={state_variable}, "
                f"位移空间={space.__class__.__name__}, 次数={space.p}, 总自由度={TGDOF_uh}")
            
        elif analysis_method == 'hzmfem':
            huzhang_space_degree = 1
            integration_order = huzhang_space_degree + 4
            from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
            analyzer = HuZhangMFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=huzhang_space_degree,
                                        integration_order=integration_order,
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                        interpolation_scheme=interpolation_scheme,
                                    )
            space_sigmah = analyzer.huzhang_space
            space_uh = analyzer.tensor_space
            TGDOF_uh = space_uh.number_of_global_dofs()
            TGDOF_sigmah = space_sigmah.number_of_global_dofs()
            self._log_info(f"分析阶段参数, "
                           f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                           f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
                           f"离散方法={analysis_method}, 网格类型={mesh_type}, "
                           f"状态变量={state_variable}, "
                           f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 总自由度={TGDOF_uh}, "
                           f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, 总自由度={TGDOF_sigmah}")
        
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
        else:
            raise ValueError(f"不支持的密度位置类型: {density_location}")

        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=analyzer, state_variable=state_variable)

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

        if optimizer_algorithm == 'mma': 
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
            
        self._log_info(f"优化阶段参数, "
                       f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_clamped_beam_2d")
        save_path.mkdir(parents=True, exist_ok=True)

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


    @run.register('test_bearing_device_2d')
    def run(self, analysis_method, model: str = 'half') -> Union[TensorLike, OptimizationHistory]:
        E = 100.0
        nu = 0.4   # 可压缩
        # nu = 0.5    # 不可压缩
        # nu = 0.4999 # 近不可压缩
        plane_type = 'plane_strain'  # 'plane_stress' or 'plane_strain'
        
        from soptx.model.bearing_device_2d import HalfBearingDevice2D, BearingDevice2D
        if model == 'half':
            domain = [0, 60, 0, 40]
            pde = HalfBearingDevice2D(
                                domain=domain,
                                t=-1.8,
                                E=E, nu=nu,
                                plane_type=plane_type,
                            )
            nx, ny = 60, 40
        elif model == 'full':
            domain = [0, 120, 0, 40]
            pde = BearingDevice2D(
                                domain=domain,
                                t=-1.8,
                                E=E, nu=nu,
                                plane_type=plane_type,
                            )
            nx, ny = 120, 40

        volume_fraction = 0.35
        penalty_factor = 3.0
        
        # 'node', 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 
        assembly_method = 'voigt'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500
        tolerance = 1e-2
        state_variable = 'sigma'  # 'u', 'sigma'

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.25

        # mesh_type = 'uniform_quad'
        # mesh_type = 'uniform_crisscross_tri'
        mesh_type = 'uniform_aligned_tri'

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
            integration_order = space_degree + 4
            from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
            analyzer = LagrangeFEMAnalyzer(
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
            space = analyzer.tensor_space
            TGDOF_uh = space.number_of_global_dofs()
            self._log_info(f"分析阶段参数, "
                f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
                f"离散方法={analysis_method}, 网格类型={mesh_type}, "
                f"状态变量={state_variable}, "
                f"位移空间={space.__class__.__name__}, 次数={space.p}, 总自由度={TGDOF_uh}")

        elif analysis_method == 'hzmfem':
            huzhang_space_degree = 1
            integration_order = huzhang_space_degree + 4
            from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
            analyzer = HuZhangMFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=huzhang_space_degree,
                                        integration_order=integration_order,
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                        interpolation_scheme=interpolation_scheme,
                                    )
            space_sigmah = analyzer.huzhang_space
            space_uh = analyzer.tensor_space
            TGDOF_uh = space_uh.number_of_global_dofs()
            TGDOF_sigmah = space_sigmah.number_of_global_dofs()
            self._log_info(f"分析阶段参数, "
                           f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                           f"杨氏模量={pde.E}, 泊松比={pde.nu}, \n"
                           f"离散方法={analysis_method}, 网格类型={mesh_type}, "
                           f"状态变量={state_variable}, "
                           f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 总自由度={TGDOF_uh}, "
                           f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, 总自由度={TGDOF_sigmah}")
        
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
        else:
            raise ValueError(f"不支持的密度位置类型: {density_location}")
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=analyzer, state_variable=state_variable)

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

        if optimizer_algorithm == 'mma': 
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

        self._log_info(f"优化阶段参数, "
                       f"密度类型={density_location}, 密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"优化算法={optimizer_algorithm} , " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_bearing_device_2d")
        save_path.mkdir(parents=True, exist_ok=True)

        
        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history


if __name__ == "__main__":
    test = DensityTopOptHuZhangTest(enable_logging=True)
    
    # test.run.set('test_bridge_2d')
    # test.run.set('test_clamped_beam_2d')
    test.run.set('test_bearing_device_2d')

    rho_opt, history = test.run(analysis_method='hzmfem')
    