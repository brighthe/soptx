from typing import Optional, Union
from pathlib import Path
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import (save_optimization_history, plot_optimization_history,
                                    save_history_data, load_history_data, plot_optimization_history_comparison)
from soptx.optimization.tools import OptimizationHistory

class DensityTopOptTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test_subsec6_6_2')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        bm.set_backend('numpy') # numpy, pytorch
        # bm.set_default_device('cpu') # cpu, cuda
        device = 'cpu' # cpu, cuda

        domain = [0, 60.0, 0, 20.0, 0, 4.0]
        p = -1.0
        E, nu = 1.0, 0.3
        plane_type = '3d'

        nx, ny, nz = 60, 20, 4
        # nx, ny, nz = 120, 40, 8
        mesh_type = 'uniform_hex'
        # mesh_type = 'uniform_tet'

        space_degree = 1
        integration_order = space_degree + 2 # 单元密度 + 六面体网格
        # integration_order = space_degree*2 + 2 # 单元密度 + 四面体网格

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast', 'symbolic'
        assembly_method = 'symbolic'
        # 'mumps', 'cg'
        solve_method = 'mumps'

        max_iterations = 200
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 1.5

        from soptx.model.cantilever_3d_lfem import CantileverBeam3d
        pde = CantileverBeam3d(
                            domain=domain,
                            p=p, E=E, nu=nu,
                            plane_type=plane_type,
                        )

        pde.init_mesh.set(mesh_type)
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
                                    bisection_tol=1e-3
                                )

        self._log_info(f"开始密度拓扑优化, "
            f"模型名称={pde.__class__.__name__} \n"
            f"体积约束={volume_fraction}, "
            f"网格类型={displacement_mesh.__class__.__name__},  " 
            f"密度类型={density_location}, "
            f"空间次数={space_degree}, 积分次数={integration_order}, 位移自由度总数={analysis_tgdofs}, \n"
            f"矩阵组装方法={assembly_method}, 线性系统求解方法={solve_method}, \n"
            f"后端={bm.get_current_backend().__class__.__name__}, 设备={device} \n"
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
    
    @run.register('test_subsec6_6_3')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu' 
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec6_6_3_canti_3d/json")
        save_path.mkdir(parents=True, exist_ok=True)    
    
        histories = load_history_data(save_path, labels=['manual', 'auto'])

        # 重命名键以美化图例
        histories = {'手动微分': histories['manual'], '自动微分': histories['auto']}

        plot_optimization_history_comparison(
                                        histories,
                                        save_path=f'{save_path}/convergence_comparison.png',
                                        plot_type='objective'
                                    )

        bm.set_backend('pytorch') # numpy, pytorch
        # bm.set_default_device('cuda') # cpu, cuda
        device = 'cuda' # cpu, cuda

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

        from soptx.model.cantilever_3d_lfem import CantileverBeam3d
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

        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()

        diff_mode_compliance = 'auto'
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer, 
                                                diff_mode=diff_mode_compliance)

        diff_mode_volume = 'auto'
        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, 
                                            volume_fraction=volume_fraction,
                                            diff_mode=diff_mode_volume)

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
            f"目标函数灵敏度分析方法={compliance_objective._diff_mode}, "
            f"体积分数约束灵敏度分析方法={volume_constraint._diff_mode} \n"
            f"模型名称={pde.__class__.__name__}, 体积分数约束={volume_fraction}, \n"
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

        save_history_data(history=history, save_path=str(save_path/'json'), label='manual')

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    
    @run.register('test_subsec6_6_4')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu' 
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec6_6_4_canti_3d/json")
        save_path.mkdir(parents=True, exist_ok=True)    
    
        # histories = load_history_data(save_path, labels=['numpy', 'pytorch', 'jax'])
        histories = load_history_data(save_path, labels=['cpu', 'gpu'])

        # 重命名键以美化图例
        # histories = {'NumPy': histories['numpy'], 'PyTorch': histories['pytorch'], 'JAX': histories['jax']}
        histories = {'CPU': histories['cpu'], 'GPU': histories['gpu']}

        plot_optimization_history_comparison(
                                        histories,
                                        save_path=f'{save_path}/convergence_comparison_device.png',
                                        # save_path=f'{save_path}/convergence_comparison_backend.png',
                                        plot_type='objective'
                                    )

        bm.set_backend('pytorch') # numpy, pytorch
        # bm.set_default_device('cuda') # cpu, cuda
        device = 'cpu' # cpu, cuda

        domain = [0, 60.0, 0, 20.0, 0, 4.0]
        p = -1.0
        E, nu = 1.0, 0.3
        plane_type = '3d'

        nx, ny, nz = 60, 20, 4
        # nx, ny, nz = 120, 40, 8
        mesh_type = 'uniform_hex'
        # mesh_type = 'uniform_tet'

        space_degree = 1
        integration_order = space_degree + 1 # 单元密度 + 六面体网格
        # integration_order = space_degree*2 + 2 # 单元密度 + 四面体网格

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element'
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

        from soptx.model.cantilever_3d_lfem import CantileverBeam3d
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

        analysis_tspace = lagrange_fem_analyzer.tensor_space
        analysis_tgdofs = analysis_tspace.number_of_global_dofs()

        diff_mode_compliance = 'manual'
        from soptx.optimization.compliance_objective import ComplianceObjective
        compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer, 
                                                diff_mode=diff_mode_compliance)

        diff_mode_volume = 'manual'
        from soptx.optimization.volume_constraint import VolumeConstraint
        volume_constraint = VolumeConstraint(analyzer=lagrange_fem_analyzer, 
                                            volume_fraction=volume_fraction,
                                            diff_mode=diff_mode_volume)

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
            f"目标函数灵敏度分析方法={compliance_objective._diff_mode}, "
            f"体积分数约束灵敏度分析方法={volume_constraint._diff_mode} \n"
            f"模型名称={pde.__class__.__name__}, 体积分数约束={volume_fraction}, \n"
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
        save_path = Path(f"{base_dir}/test_cantilever_3d_torch")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_history_data(history=history, save_path=str(save_path/'json'), label='pytorch')

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    
    @run.register('test_subsec6_6_4_plot1')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        """绘制柱状图代码"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager

        # ==========================================
        # 1. 字体路径配置 (基于您系统扫描的结果)
        # ==========================================
        # 中文字体：SimHei
        path_zh = '/usr/share/fonts/suanhai_fonts/Sim/simhei.ttf'
        # 西文字体：Times New Roman (常规)
        path_en = '/usr/share/fonts/suanhai_fonts/Times/times.ttf'
        # 西文字体：Times New Roman (粗体) - 用于强调数字
        path_en_bold = '/usr/share/fonts/suanhai_fonts/Times/timesbd.ttf'

        # 加载字体属性对象
        try:
            font_zh = font_manager.FontProperties(fname=path_zh, size=12) 
            font_zh_bold = font_manager.FontProperties(fname=path_zh, size=12, weight='bold')
            font_en = font_manager.FontProperties(fname=path_en, size=12)
            font_en_bold = font_manager.FontProperties(fname=path_en_bold, size=11)
            print("字体加载成功！")
        except Exception as e:
            print(f"字体加载失败: {e}")

        # ==========================================
        # 2. 数据准备
        # ==========================================
        labels = ['CPU', 'GPU']
        # 总耗时
        total_times = np.array([21.087, 4.841])
        # 分析阶段耗时
        analysis_times = np.array([20.677, 4.382])
        # 优化阶段耗时
        optimization_times = np.array([0.317, 0.377])
        # 其他/开销 (用于填补微小差值，保证总高一致)
        other_times = total_times - analysis_times - optimization_times

        # ==========================================
        # 3. 绘图主设置
        # ==========================================
        # 设置 DPI 为 300 (高清印刷标准)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        # 柱子宽度
        width = 0.5
        # 颜色配置 (学术风格：柔和蓝、柔和橙、灰)
        color_analysis = '#5B9BD5' 
        color_opt = '#ED7D31'      
        color_other = '#A5A5A5'    

        # ==========================================
        # 4. 绘制堆叠柱状图
        # ==========================================
        # 绘制分析阶段 (底部)
        p1 = ax.bar(labels, analysis_times, width, label='结构分析阶段', 
                    color=color_analysis, edgecolor='black', linewidth=0.8, zorder=3)

        # 绘制优化阶段 (中间)
        p2 = ax.bar(labels, optimization_times, width, bottom=analysis_times, 
                    label='优化更新阶段', color=color_opt, edgecolor='black', linewidth=0.8, zorder=3)

        # 绘制其他开销 (顶部)
        p3 = ax.bar(labels, other_times, width, bottom=analysis_times + optimization_times, 
                    label='其他/开销', color=color_other, edgecolor='black', linewidth=0.8, zorder=3)

        # ==========================================
        # 5. 精细化标注 (混排字体核心部分)
        # ==========================================
        def add_labels(rects, data_values):
            """在柱子内部添加数值"""
            for rect, val in zip(rects, data_values):
                height = rect.get_height()
                y_pos = rect.get_y() + height / 2
                
                # 只有数值大于1秒才显示，避免文字重叠
                if val > 1.0: 
                    # 注意：这里使用 font_en_bold (Times New Roman Bold)
                    ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                            f'{val:.3f} s',
                            ha='center', va='center', color='white', 
                            fontproperties=font_en_bold) 

        # 添加数值标注
        add_labels(p1, analysis_times)

        # 在柱子顶端添加总耗时 (中文 + 数字)
        for i, total in enumerate(total_times):
            # 这里我们采用 "SimHei" 显示整行，因为单独混排非常复杂。
            # SimHei 的数字在图表中是可以接受的，或者我们可以分段写。
            # 方案：为了美观，整行使用中文粗体
            ax.text(i, total + 0.5, f'总计: {total:.3f} s', 
                    ha='center', va='bottom', 
                    fontproperties=font_zh_bold) 

        # ==========================================
        # 6. 图表修饰 (应用字体)
        # ==========================================

        # X轴标签：CPU/GPU 是英文，使用 Times New Roman
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontproperties=font_en)

        # Y轴标签：中文，使用 SimHei
        ax.set_ylabel('平均单次迭代耗时 (s)', fontproperties=font_zh)

        # Y轴刻度数字：使用 Times New Roman
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en)

        # 图例：中文，使用 SimHei
        ax.legend(loc='upper right', frameon=True, edgecolor='black', prop=font_zh)

        # 网格线
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

        # Y轴范围
        ax.set_ylim(0, 24)

        # 紧凑布局并保存
        plt.tight_layout()
        output_file = 'fig_6_11_final.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已生成: {output_file}")

        # 显示图片 (如果在 Notebook 环境)
        plt.show()

        print("---------")

    @run.register('test_subsec6_6_4_plot2')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        """绘制柱状图代码"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager

        # ==========================================
        # 1. 字体路径配置 (沿用您之前的成功配置)
        # ==========================================
        # 中文字体：SimHei
        path_zh = '/usr/share/fonts/suanhai_fonts/Sim/simhei.ttf'
        # 西文字体：Times New Roman (常规)
        path_en = '/usr/share/fonts/suanhai_fonts/Times/times.ttf'
        # 西文字体：Times New Roman (粗体) - 用于强调数字
        path_en_bold = '/usr/share/fonts/suanhai_fonts/Times/timesbd.ttf'

        # 加载字体属性对象
        try:
            font_zh = font_manager.FontProperties(fname=path_zh, size=12) 
            font_zh_bold = font_manager.FontProperties(fname=path_zh, size=12, weight='bold')
            font_en = font_manager.FontProperties(fname=path_en, size=12)
            font_en_bold = font_manager.FontProperties(fname=path_en_bold, size=11)
            print("字体加载成功！")
        except Exception as e:
            print(f"字体加载失败: {e}")

        # ==========================================
        # 2. 数据准备
        # ==========================================
        labels = ['CPU', 'GPU']

        # 总分析阶段耗时 (用于顶部标注)
        total_analysis_times = np.array([20.677, 4.382])

        # 细分数据 (来自论文正文)
        solver_times = np.array([19.565, 4.187])   # 线性系统求解
        assembly_times = np.array([0.695, 0.127])  # 矩阵组装
        # 计算剩余部分 (边界处理、数据传输等)
        other_times = total_analysis_times - solver_times - assembly_times

        # ==========================================
        # 3. 绘图主设置
        # ==========================================
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        width = 0.5
        # 配色方案 (保持学术风格，与图6.11协调但有所区分)
        # 线性求解 (主导瓶颈) - 深蓝色
        color_solver = '#4472C4' 
        # # 矩阵组装 (加速最快) - 橙色
        # color_assembly = '#ED7D31'      
        # 矩阵组装：改为绿色，避免与左图的“优化阶段(橙色)”混淆
        color_assembly = '#70AD47'  # 推荐使用这种学术绿    
        # 其他 - 灰色
        color_other = '#A5A5A5'    

        # ==========================================
        # 4. 绘制堆叠柱状图
        # ==========================================
        # 底部：线性系统求解 (占比最大)
        p1 = ax.bar(labels, solver_times, width, label='线性系统求解', 
                    color=color_solver, edgecolor='black', linewidth=0.8, zorder=3)

        # 中间：矩阵组装
        p2 = ax.bar(labels, assembly_times, width, bottom=solver_times, 
                    label='矩阵组装', color=color_assembly, edgecolor='black', linewidth=0.8, zorder=3)

        # 顶部：其他
        p3 = ax.bar(labels, other_times, width, bottom=solver_times + assembly_times, 
                    label='其他/开销', color=color_other, edgecolor='black', linewidth=0.8, zorder=3)

        # ==========================================
        # 5. 精细化标注
        # ==========================================
        def add_labels(rects, data_values, show_threshold=0):
            """
            rects: 柱子对象
            data_values: 数据值
            show_threshold: 显示阈值，只有大于该值的才显示
            """
            for rect, val in zip(rects, data_values):
                height = rect.get_height()
                y_pos = rect.get_y() + height / 2
                
                if val > show_threshold: 
                    ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                            f'{val:.3f} s',
                            ha='center', va='center', color='white', 
                            fontproperties=font_en_bold)

        # 添加数值标注
        add_labels(p1, solver_times, show_threshold=0.5)
        # CPU的组装时间(0.695)足够大，可以显示；GPU的(0.127)太小，自动隐藏以保持整洁
        # add_labels(p2, assembly_times)

        # 在柱子顶端添加总耗时 (中文 + 数字)
        for i, total in enumerate(total_analysis_times):
            ax.text(i, total + 0.5, f'总计: {total:.3f} s', 
                    ha='center', va='bottom', 
                    fontproperties=font_zh_bold) 

        # ==========================================
        # 6. 图表修饰
        # ==========================================

        # X轴标签
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontproperties=font_en)

        # Y轴标签 (注意这里是分析阶段的时间，单位依然是s)
        ax.set_ylabel('结构分析阶段耗时 (s)', fontproperties=font_zh)

        # Y轴刻度数字
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en)

        # 图例
        ax.legend(loc='upper right', frameon=True, edgecolor='black', prop=font_zh)

        # 网格线
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

        # Y轴范围 (根据最大值20.677调整)
        ax.set_ylim(0, 24)

        # 保存
        plt.tight_layout()
        output_file = 'fig_6_12_analysis_breakdown.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已生成: {output_file}")

        plt.show()

        print("---------")

    @run.register('test_subsec6_5_2_canti2d_corner')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu' 
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec6_5_2_canti2d_corner/json")
        save_path.mkdir(parents=True, exist_ok=True)    
    
        histories = load_history_data(save_path, labels=['convergence'])

        plot_optimization_history(histories['convergence'], save_path=f'{save_path}/convergence.png')
    
        # 固定参数
        domain = [0, 160.0, 0, 100.0]
        P = -1.0
        E, nu = 1.0, 0.3

        # 测试参数
        nx, ny = 160, 100
        mesh_type = 'uniform_quad'

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        volume_fraction = 0.4
        penalty_factor = 3.0

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'

        optimizer_algorithm = 'oc'  # 'oc', 'mma'
        max_iterations = 200
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 6.0

        from soptx.model.cantilever_2d_lfem import Cantilever2dCorner
        pde = Cantilever2dCorner(
                            domain=domain,
                            P=P, E=E, nu=nu,
                            enable_logging=False
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
                                    disp_mesh=displacement_mesh,
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
                                    bisection_tol=1e-3
                                )

        self._log_info(f"开始密度拓扑优化 \n"
                       f"模型名称={pde.__class__.__name__}, "
                       f"体积约束={volume_fraction}, "
                       f"网格类型={mesh_type},  " 
                       f"密度类型={density_location}, " 
                       f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
                       f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 位移场自由度={analysis_tgdofs}, "
                       f"优化算法={optimizer_algorithm} , 最大迭代次数={max_iterations}, 收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, " 
                       f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec6_5_2_canti2d_corner")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_history_data(history=history, save_path=str(save_path/'json'), label='convergence')

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        
        
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    

    @run.register('test_subsec6_6_simple_bridge_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        P = 1.0
        E, nu = 1, 0.3
        domain = [0, 60, 0, 30]
        plane_type = 'plane_stress'
        
        from soptx.model.simple_bridge_2d_lfem import SimpleBridge2d
        pde = SimpleBridge2d(
                        domain=domain,
                        P=P, E=E, nu=nu, 
                        plane_type=plane_type,
                        enable_logging=False
                    )

        nx, ny = 60, 30
        mesh_type = 'uniform_quad'

        space_degree = 1
        integration_order = space_degree + 1 # 单元密度 + 四边形网格
        # integration_order = space_degree*2 + 3 # 节点密度 + 四边形网格

        volume_fraction = 0.3
        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'
        solve_method = 'mumps'

        max_iterations = 200
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
        rmin = 2.4

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
                                    bisection_tol=1e-3
                                )

        self._log_info(f"开始密度拓扑优化, \n"
            f"模型名称={pde.__class__.__name__}, 体积约束={volume_fraction}, \n"
            f"网格类型={mesh_type}, 密度类型={density_location}, " 
            f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
            f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 积分次数={integration_order}, 位移场自由度={analysis_tgdofs}, \n"
            f"优化算法={optimizer.__class__.__name__}, 最大迭代次数={max_iterations}, "
            f"设计变量变化收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec6_6_simple_bridge_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
    

    @run.register('test_subsec6_6_half_wheel_2d')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        P = 1.0
        E, nu = 1, 0.3
        domain = [0, 120, 0, 60]
        plane_type = 'plane_stress'
        
        from soptx.model.half_wheel_2d_lfem import HalfWheel2d
        pde = HalfWheel2d(
                    domain=domain,
                    P=P, E=E, nu=nu, 
                    plane_type=plane_type,
                    enable_logging=False
                )

        nx, ny = 120, 60
        mesh_type = 'uniform_quad'

        space_degree = 1
        integration_order = space_degree + 1 # 单元密度 + 四边形网格
        # integration_order = space_degree*2 + 3 # 节点密度 + 四边形网格

        volume_fraction = 0.3
        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9

        # 'element', 'node'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'
        solve_method = 'mumps'

        max_iterations = 200
        change_tolerance = 1e-2
        use_penalty_continuation = False

        filter_type = 'sensitivity' # 'none', 'sensitivity', 'density'
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
                                    bisection_tol=1e-3
                                )

        self._log_info(f"开始密度拓扑优化, \n"
            f"模型名称={pde.__class__.__name__}, 体积约束={volume_fraction}, \n"
            f"网格类型={mesh_type}, 密度类型={density_location}, " 
            f"密度网格尺寸={design_variable_mesh.number_of_cells()}, 密度场自由度={rho.shape}, " 
            f"位移网格尺寸={displacement_mesh.number_of_cells()}, 位移有限元空间阶数={space_degree}, 积分次数={integration_order}, 位移场自由度={analysis_tgdofs}, \n"
            f"优化算法={optimizer.__class__.__name__}, 最大迭代次数={max_iterations}, "
            f"设计变量变化收敛容差={change_tolerance}, 惩罚因子连续化={use_penalty_continuation}, \n" 
            f"过滤类型={filter_type}, 过滤半径={rmin}, ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec6_6_half_wheel_2d")
        save_path.mkdir(parents=True, exist_ok=True)    

        save_optimization_history(mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))

        return rho_opt, history
        

if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec6_6_4')
    rho_opt, history = test.run()