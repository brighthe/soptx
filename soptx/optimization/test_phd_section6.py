from typing import Optional, Union
from pathlib import Path
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

    @variantmethod('test_subsec6_4_1')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 60.0, 0, 20.0, 0, 4.0]
        p = -1.0
        E, nu = 1.0, 0.3
        plane_type = '3d'

        nx, ny, nz = 60, 20, 4
        # mesh_type = 'uniform_hex'
        mesh_type = 'uniform_tet'

        space_degree = 2
        # integration_order = space_degree + 1 # 单元密度 + 六面体网格
        integration_order = space_degree*2 + 2 # 单元密度 + 四面体网格

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'

        max_iterations = 500
        change_tolerance = 1e-3
        use_penalty_continuation = True

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 1.5

        from soptx.model.cantilever_3d import CantileverBeam3d
        pde = CantileverBeam3d(
                            domain=domain,
                            p=p, E=E, nu=nu,
                            plane_type=plane_type,
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
                                    design_mesh=design_variable_mesh,
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
            f"模型名称={pde.__class__.__name__}, "
            f"体积约束={volume_fraction}, "
            f"网格类型={displacement_mesh.__class__.__name__},  " 
            f"密度类型={density_location}, "
            f"空间次数={space_degree}, 积分次数={integration_order}, 位移自由度总数={analysis_tgdofs}, \n"
            f"矩阵组装方法={assembly_method}, \n"
            f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
            f"设计变量变化收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
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
    
    @run.register('test_subsec6_4_2')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        bm.set_backend('pytorch') # numpy, pytorch
        # bm.set_default_device('cuda') # cpu, cuda
        device = 'cpu' # cpu, cuda

        domain = [0, 60.0, 0, 20.0, 0, 4.0]
        p = -1.0
        E, nu = 1.0, 0.3
        plane_type = '3d'

        nx, ny, nz = 60, 20, 4
        mesh_type = 'uniform_hex'
        # mesh_type = 'uniform_tet'

        space_degree = 1
        integration_order = space_degree + 1 # 单元密度 + 六面体网格
        # integration_order = space_degree*2 + 2 # 单元密度 + 四面体网格

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'
        # 'mumps', 'cg'
        solve_method = 'mumps'

        max_iterations = 200
        change_tolerance = 1e-2

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 1.5

        from soptx.model.cantilever_3d import CantileverBeam3d
        pde = CantileverBeam3d(
                            domain=domain,
                            p=p, E=E, nu=nu,
                            plane_type=plane_type,
                        )
        pde.init_mesh.set(mesh_type)
        # displacement_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz, device=device)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            device=device,
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
                                    design_mesh=design_variable_mesh,
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
                                    solve_method=solve_method,
                                    topopt_algorithm='density_based',
                                )

        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()

        diff_mode_compliance = 'manual'
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer, 
                                                diff_mode=diff_mode_compliance)

        # state = compliance_objective._analyzer.solve_state(rho_val=rho)

        # dc_auto = compliance_objective.jac(density=rho, state=state, diff_mode='auto')

        # dc_manual = compliance_objective.jac(density=rho, state=state, diff_mode='manual')

        # error_dc = bm.sum(bm.abs(dc_auto - dc_manual))

        diff_mode_volume = 'manual'
        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, 
                                            volume_fraction=volume_fraction,
                                            diff_mode=diff_mode_volume)
        
        # dv_auto = volume_constraint.jac(density=rho, diff_mode='auto')

        # dv_manual = volume_constraint.jac(density=rho, diff_mode='manual')

        # error_dv = bm.sum(bm.abs(dv_auto - dv_manual))

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

        self._log_info(f"开始密度拓扑优化, \n"
            f"设备={device}, 后端={bm.backend_name}, "
            f"目标函数自动微分={diff_mode_compliance}, 体积分数约束自动微分={diff_mode_volume}\n"
            f"模型名称={pde.__class__.__name__}, 体积分数约束={volume_fraction}, \n"
            f"目标函数灵敏度计算方法={diff_mode_compliance}, 体积分数约束灵敏度计算方法={diff_mode_volume}, \n"
            f"网格类型={displacement_mesh.__class__.__name__},  " 
            f"密度类型={density_location}, "
            f"空间次数={space_degree}, 积分次数={integration_order}, 位移自由度总数={analysis_tgdofs}, \n"
            f"矩阵组装方法={assembly_method}, \n"
            f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, "
            f"设计变量变化收敛容差={change_tolerance} \n" 
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

    test.run.set('test_subsec6_4_2')
    rho_opt, history = test.run()