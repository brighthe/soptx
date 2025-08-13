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

    @variantmethod('OC_element')
    def run(self, 
            space_degree: Optional[int] = None, 
            filter_type: str = 'none',
            nx: int = 60,
            ny: int = 20,
        ) -> Union[TensorLike, OptimizationHistory]:
        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        domain_length = pde.domain[1] - pde.domain[0]
        domain_height = pde.domain[3] - pde.domain[2]
        pde.init_mesh.set('uniform_quad')

        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

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
                                    density_location='element',
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=self.relative_density,
                                                integrator_order=self.integrator_order,
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        fe_tspace = lagrange_fem_analyzer.tensor_space
        fe_dofs = fe_tspace.number_of_global_dofs()
        
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)

        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=self.volume_fraction)

        from soptx.regularization.filter import Filter
        rmin = (0.04 * nx) / (domain_length / nx)
        filter_regularization = Filter(
                                    mesh=opt_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin
                                )


        from soptx.optimization.oc_optimizer import OCOptimizer
        oc_optimizer = OCOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': 10,
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
        self._log_info(f"开始密度拓扑优化, 网格尺寸={nx}*{ny}, 空间阶数={space_degree}, 物理场自由度={fe_dofs}, 过滤类型={filter_type}, 过滤半径={rmin}, ")
        rho_opt, history = oc_optimizer.optimize(density_distribution=rho)

        # 保存结果
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/element_p{self.space_degree}_filter_{filter_type}_({nx},{ny})")
        save_path.mkdir(parents=True, exist_ok=True)
        save_optimization_history(opt_mesh, history, str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    

    @run.register('test_density_location')
    def run(self) -> Union[TensorLike, OptimizationHistory]:

        # 参数设置
        nx, ny = 30, 10
        density_location = 'element'  # 'density_subelement_gauss_point', 'gauss_integration_point', 'element'
        space_degree = 2
        integration_order = space_degree + 1
        penalty_factor = 3.0
        filter_type = 'none' # density
        
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
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=self.volume_fraction)

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
                                'max_iterations': 43,
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
    

    @run.register('test_optimization')
    def run(self, 
            density_location: str = 'element', 
        ) -> Union[TensorLike, OptimizationHistory]:
        # 参数设置
        nx, ny = 6, 2
        space_degree = 1
        integration_order = space_degree + 1
        filter_type = 'none'

        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        domain_length = pde.domain[1] - pde.domain[0]
        domain_height = pde.domain[3] - pde.domain[2]
        pde.init_mesh.set('uniform_quad')

        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

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
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=self.volume_fraction,
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
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, volume_fraction=self.volume_fraction)
        test = volume_constraint._compute_volume(physical_density=rho)

        from soptx.regularization.filter import Filter
        rmin = (0.04 * nx) / (domain_length / nx)

        from soptx.regularization.matrix_builder import FilterMatrixBuilder
        filter_matrix = FilterMatrixBuilder(
                                    mesh=opt_mesh,
                                    rmin=rmin,
                                    density_location=density_location,
                                    integrator_order=self.integrator_order,
                                    interpolation_order=1,
                                )
        H, Hs = filter_matrix._compute_weighted_matrix_general(rmin=rmin, domain=pde.domain)

        filter_regularization = Filter(
                                    mesh=opt_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                )


        from soptx.optimization.oc_optimizer import OCOptimizer
        oc_optimizer = OCOptimizer(
                            objective=compliance_objective,
                            constraint=volume_constraint,
                            filter=filter_regularization,
                            options={
                                'max_iterations': 300,
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
        self._log_info(f"开始密度拓扑优化, 网格尺寸={nx}*{ny}, 空间阶数={space_degree}, 物理场自由度={fe_dofs}, " 
                       f"密度分布位置={density_location}, "
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        rho_opt, history = oc_optimizer.optimize(density_distribution=rho)

        # 保存结果
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/density_topopt_p{self.space_degree}_density_{density_location}_({nx},{ny})")
        save_path.mkdir(parents=True, exist_ok=True)
        save_optimization_history(opt_mesh, history, str(save_path))
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


    # test.run.set('test_optimization')
    # rho, history = test.run(density_location='gauss_integration_point')
    
    # test.run.set('test_volume_constraint')
    # test.run()

    # test.run.set('test_regularization')
    # rho, history = test.run()

    test.run.set('test_density_location')
    rho_opt, history = test.run()

