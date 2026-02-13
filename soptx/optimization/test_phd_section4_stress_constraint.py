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
    

    def verify_stress_results(self, optimizer, rho_phys):
        """
        验证 AL-MMA 优化器的计算结果，复刻 PolyScript_quad.m 的验证逻辑。
        
        参数:
            optimizer: 训练好的 ALMMMAOptimizer 实例
            rho_phys: 最终的物理密度场 (TensorLike)
            design_variable: 最终的设计变量 (可选，用于调试)
        """
        import numpy as np
        
        # =========================================================================
        # 0. 数据准备与提取
        # =========================================================================
        analyzer = optimizer._al_objective._analyzer
        stress_constraint = optimizer._al_objective._stress_constraint
        
        # 确保数据是 numpy 格式以便绘图和打印
        if hasattr(rho_phys, 'to_numpy'):
            rho_np = rho_phys.to_numpy()
        elif hasattr(rho_phys, 'numpy'): # torch
            rho_np = rho_phys.detach().cpu().numpy()
        else:
            rho_np = np.array(rho_phys)
            
        n_elem = len(rho_np)
        slim = float(stress_constraint._stress_limit)
        
        print(f"\n{'='*20} 应力约束检查结果 {'='*20}")
        
        # =========================================================================
        # 1. 重新计算状态与应力
        # =========================================================================
        # 求解最终状态
        state = analyzer.solve_state(rho_val=rho_phys)
        
        # 计算材料插值系数 E_scale = E_rho / E0
        # 注意：这里我们调用 analyzer 内部的插值方案
        E_rho = analyzer.interpolation_scheme.interpolate_material(
            material=analyzer.material,
            rho_val=rho_phys,
            integration_order=analyzer.integration_order
        )
        E0 = analyzer.material.youngs_modulus
        
        # 转换为 numpy
        if hasattr(E_rho, 'to_numpy'):
            E_rho_np = E_rho.to_numpy()
        else:
            E_rho_np = np.array(E_rho)
            
        E_scale = E_rho_np / E0  # 归一化刚度系数 (对应 MATLAB 的 E)
        
        # 获取 von Mises 应力 (实心材料应力)
        # 假设 state['von_mises'] 返回的是基于实心材料本构计算的应力
        if 'von_mises' in state:
            vm_stress = state['von_mises']
            if not isinstance(vm_stress, np.ndarray):
                vm_stress = np.array(vm_stress)
        else:
            # 如果 state 中没有直接存储 vm，需要根据应力张量计算
            # 这里假设 state['stress'] 存在且为 Voigt 符号 [sig_xx, sig_yy, sig_xy]
            stress = state['stress'] # (N, 3)
            s11, s22, s12 = stress[:, 0], stress[:, 1], stress[:, 2]
            vm_stress = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)

        # =========================================================================
        # 2. 计算归一化应力度量 (Normalized Stress Measure)
        #    SM = E * (sigma_vm / sigma_lim)
        #    这是多项式松弛约束中实际控制的量
        # =========================================================================
        # 注意维度广播
        SM = E_scale.reshape(-1) * vm_stress.reshape(-1) / slim
        
        # =========================================================================
        # 3. 统计分析
        # =========================================================================
        print(f"应力限制 (σ_lim): {slim:.1f} MPa")
        print(f"最大归一化应力 (max(SM)): {np.max(SM):.4f}")
        print(f"平均归一化应力: {np.mean(SM):.4f}")
        
        # 找出违反约束的单元
        tolerance = 0.01 # 1% 容差
        violated_mask = SM > (1.0 + tolerance)
        n_violated = np.sum(violated_mask)
        
        print(f"违反应力约束的单元数量: {n_violated} / {n_elem}")
        
        if n_violated > 0:
            max_violation = (np.max(SM) - 1.0) * 100
            print(f"?? 警告：存在应力约束违反！最大违反程度: {max_violation:.2f}%")
        else:
            print(f"? 所有应力约束均满足！")
            
        # =========================================================================
        # 5. 实体单元统计 (ρ > 0.5)
        # =========================================================================
        print(f"\n{'='*20} 实体单元 (ρ>0.5) 应力统计 {'='*20}")
        solid_mask = rho_np > 0.5
        n_solid = np.sum(solid_mask)
        print(f"实体单元数量: {n_solid}")
        
        if n_solid > 0:
            solid_SM = SM[solid_mask]
            print(f"实体单元最大归一化应力: {np.max(solid_SM):.4f}")
            print(f"实体单元平均归一化应力: {np.mean(solid_SM):.4f}")
            
            n_violated_solid = np.sum(solid_SM > (1.0 + tolerance))
            print(f"违反约束的实体单元: {n_violated_solid}")
            
            # =====================================================================
            # 6. 绘制 von Mises 屈服面
            # =====================================================================
            self._plot_yield_surface(state, E_scale, solid_mask, SM, slim)
        else:
            print("没有实体单元，跳过屈服面绘制。")

    def _plot_yield_surface(self, state, E_scale, solid_mask, SM, slim):
        """
        绘制主应力空间下的 von Mises 屈服面
        """
        # 1. 获取柯西应力张量 (N, 3) -> [xx, yy, xy]
        # 注意：这里的 stress 通常是 Base Material Stress (实心材料应力)
        # 在绘制屈服面时，我们需要实际上受到的应力 (Relaxed Stress)
        # 即: sigma_relaxed = E_scale * sigma_solid
        import numpy as np
        import matplotlib.pyplot as plt
        
        if 'stress' not in state:
            print("State 字典中未找到 'stress' 张量，无法绘制屈服面。")
            return

        stress_solid = state['stress']
        if not isinstance(stress_solid, np.ndarray):
            stress_solid = np.array(stress_solid)
            
        # 应用材料插值 (Relaxation)
        # shape: (N, 3) * (N, 1) -> (N, 3)
        stress_relaxed = stress_solid * E_scale.reshape(-1, 1)
        
        # 提取实体单元的应力
        s_xx = stress_relaxed[solid_mask, 0]
        s_yy = stress_relaxed[solid_mask, 1]
        s_xy = stress_relaxed[solid_mask, 2]
        
        # 2. 计算主应力 (2D)
        # sigma_1,2 = (sx + sy)/2 +/- sqrt(((sx-sy)/2)^2 + txy^2)
        center = (s_xx + s_yy) / 2
        radius = np.sqrt(((s_xx - s_yy) / 2)**2 + s_xy**2)
        
        sigma_1 = center + radius
        sigma_2 = center - radius
        
        # 归一化
        s1_norm = sigma_1 / slim
        s2_norm = sigma_2 / slim
        sm_solid = SM[solid_mask]

        # 3. 绘图
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 绘制 von Mises 边界 (sigma_1^2 - s1*s2 + s2^2 = 1)
        # 这是一个长轴在 y=x 方向的椭圆
        t = np.linspace(0, 2*np.pi, 400)
        # 参数方程绘制单位圆是错误的，von Mises 在主应力空间是椭圆
        # MATLAB 代码画了个圆其实是近似或者示意，这里我们画精确的椭圆
        a = np.sqrt(2/3) # 这里的系数取决于具体推导，直接用等值线画更准确
        
        delta = 0.02
        x = np.arange(-2.0, 2.0, delta)
        y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 - X*Y + Y**2
        CS = ax.contour(X, Y, Z, levels=[1.0], colors='r', linewidths=2.5)
        # 手动添加图例 handle
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='r', lw=2.5)]
        
        # 绘制散点
        sc = ax.scatter(s1_norm, s2_norm, c=sm_solid, cmap='jet', alpha=0.6, s=20)
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\sigma_1 / \sigma_{lim}$', fontsize=14)
        ax.set_ylabel(r'$\sigma_2 / \sigma_{lim}$', fontsize=14)
        ax.set_title('Von Mises Yield Surface Check', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Normalized Stress Measure', fontsize=12)
        
        # Legend
        ax.legend(custom_lines, ['von Mises Limit'], loc='upper right')
        
        # 统计文本
        stats_text = (
            f"Solid Elements: {np.sum(solid_mask)}\n"
            f"Max SM: {np.max(sm_solid):.3f}\n"
            f"Mean SM: {np.mean(sm_solid):.3f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        print("\n? 屈服面图已生成")
        plt.show()
    
    
    @run.register('test_subsec4_6_5_L_bracket')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 1.0, 0, 1.0]
        hole_domain = [0.4, 1.0, 0.4, 1.0]
        P = -2.0
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
                
        from soptx.optimization.volume_objective import VolumeObjective
        objective = VolumeObjective(analyzer=analyzer)

        from soptx.optimization.stress_constraint import StressConstraint
        constraint = StressConstraint(analyzer=analyzer, stress_limit=100.0)

        NC = displacement_mesh.number_of_cells()
        from soptx.optimization.augmented_lagrangian_objective import AugmentedLagrangianObjective
        augmented_lagrangian_objective = AugmentedLagrangianObjective(
                                            volume_objective=objective,
                                            stress_constraint=constraint,
                                            initial_penalty=10.0,
                                            max_penalty=10000.0,
                                            initial_lambda=bm.zeros((NC, 1), dtype=bm.float64),
                                            penalty_update_factor=1.1,
                                        )

        filter_type = 'density' # 'none', 'sensitivity', 'density'
        rmin = 0.05
        from soptx.regularization.filter import Filter
        filter_regularization = Filter(
                                    design_mesh=design_variable_mesh,
                                    filter_type=filter_type,
                                    rmin=rmin,
                                    density_location=density_location,
                                )

        from soptx.optimization.al_mma_optimizer import ALMMMAOptimizer
        optimizer = ALMMMAOptimizer(
                        al_objective=augmented_lagrangian_objective,
                        filter=filter_regularization,
                        options={
                            'max_al_iterations': 10,      # 对应 opt.MaxIter = 150
                            'mma_iters_per_al': 5,        # 对应 opt.MMA_Iter = 5
                            'change_tolerance': 0.002,    # 对应 opt.Tol = 0.002
                            'stress_tolerance': 0.003,    # 对应 opt.TolS = 0.003
                            'alpha': 1.1,                 # 对应 opt.alpha = 1.1
                            'use_penalty_continuation': False,
                        },
                        enable_logging=True,
                    )
        optimizer.options.set_advanced_options(
            move_limit=0.15,      # 对应 opt.Move = 0.15 (应力优化需限制步长防震荡)
            asymp_init=0.2,       # 对应 opt.AsymInit = 0.2
            asymp_incr=1.2,       # 对应 opt.AsymInc = 1.2
            asymp_decr=0.7        # 对应 opt.AsymDecr = 0.7
        )

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
    
if __name__ == "__main__":
    test = DensityTopOptTest(enable_logging=True)

    test.run.set('test_subsec4_6_5_L_bracket')
    rho_opt, history = test.run()