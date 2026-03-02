from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import (save_optimization_history, plot_optimization_history,
                                      save_history_data, load_history_data)
from soptx.optimization.tools import OptimizationHistory

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

# ==========================================
# 1. SOPTX 论文全局配色配置 (模块级常量)
# ==========================================
# 这段配置应该放在脚本的最顶端，所有函数都能访问
SOPTX_COLORS = {
    # 物理量 (用于收敛曲线)
    'compliance': '#d62728',  # 红色 (Tab:red)
    'volume':     '#1f77b4',  # 蓝色 (Tab:blue)
    'stress':     '#2ca02c',  # 绿色 (Tab:green)
    
    # 性能耗时 (用于柱状图)
    'analysis':   '#5B9BD5',  # 柔和蓝 (结构分析)
    'optimization': '#ED7D31',# 柔和橙 (优化更新)
    'overhead':   '#A5A5A5',  # 灰色 (其他开销)
    'total':      '#444444'   # 深灰 (用于总耗时文字)
}

# ==========================================
# 2. 字体配置 (基于您提供的绝对路径)
# ==========================================
# 建议也作为全局变量加载一次，避免每次绘图都重新加载
PATH_ZH = '/usr/share/fonts/suanhai_fonts/Sim/simhei.ttf'
PATH_EN = '/usr/share/fonts/suanhai_fonts/Times/times.ttf'
try:
    # 标签与图例字体 (中文黑体)
    # 建议将 size 也设为变量，方便统一调整
    FONT_ZH = font_manager.FontProperties(fname=PATH_ZH, size=14)
    
    # 刻度数值字体 (西文 Times New Roman)
    FONT_EN = font_manager.FontProperties(fname=PATH_EN, size=12)
    
    print(f"全局字体加载成功: {FONT_ZH.get_name()}, {FONT_EN.get_name()}")
except Exception as e:
    print(f"全局字体加载失败: {e}。将回退到默认字体。")
    FONT_ZH = None
    FONT_EN = None

def plot_optimization_history_comparison(
        histories: dict,
        save_path=None,
        show=True,
        title=None,
        figsize=None,
        linewidth=2.0,
        colors=None,
        plot_type=None  # None/'both': 双纵轴单图, 'compliance': 仅柔顺度, 'volfrac': 仅体积分数
    ):
    """
    绘制不同情形下的对比收敛曲线

    Parameters
    ----------
    histories : dict
        包含多个历史数据的字典, e.g., {'k2': history_obj1, 'k3': history_obj2}
        支持两种数据格式：
        - 扁平字典：{'compliance': [...], 'volfrac': [...]}
        - 嵌套字典：{'scalar_histories': {'compliance': [...], 'volfrac': [...]}, ...}
    plot_type : str or None
        None 或 'both' : 双纵轴单图 (左轴柔顺度，右轴体积分数)
        'compliance'   : 仅柔顺度
        'volfrac'      : 仅体积分数
    """
    # ------------------------------------------
    # 1. 绘图参数设置
    # ------------------------------------------
    if colors is None:
        colors = [
            SOPTX_COLORS['compliance'], # 红色
            SOPTX_COLORS['volume'],     # 蓝色
            '#2ca02c',                  # 绿色
            'black',                    # 黑色
            '#ff7f0e'                   # 橙色
        ]

    # 线型固定：颜色区分 k 值，线型区分物理量
    LINESTYLE_MAP = {
        'compliance': '-',   # 实线
        'volfrac':    '--',  # 虚线
    }

    # 图例前缀
    LABEL_PREFIX_MAP = {
        'compliance': '柔顺度',
        'volfrac':    '体积分数',
    }

    if plot_type is None:
        plot_type = 'both'

    if figsize is None:
        figsize = (8, 5)

    # ------------------------------------------
    # 2. 辅助函数 (数据兼容性)
    # ------------------------------------------
    def get_data(obj, key):
        """
        兼容三种数据来源：
        1. 对象属性 (在线数据)
        2. 扁平字典 (顶层直接有 key)
        3. 嵌套字典 (key 在 scalar_histories 中)
        """
        if not isinstance(obj, dict):
            return getattr(obj, key)
        if key in obj:
            return obj[key]
        if 'scalar_histories' in obj and key in obj['scalar_histories']:
            return obj['scalar_histories'][key]
        raise KeyError(f"Key '{key}' not found in dict or scalar_histories")

    # ------------------------------------------
    # 3. 创建画布与子图
    # ------------------------------------------
    if plot_type == 'both':
        fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
        ax2 = ax1.twinx()
        axes_to_plot = [
            ('compliance', ax1),
            ('volfrac',    ax2),
        ]
    elif plot_type == 'compliance':
        fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
        axes_to_plot = [('compliance', ax1)]
    elif plot_type == 'volfrac':
        fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
        axes_to_plot = [('volfrac', ax1)]
    else:
        raise ValueError("Invalid plot_type. Choose None/'both', 'compliance', or 'volfrac'.")

    # ------------------------------------------
    # 4. 绘图主循环
    # ------------------------------------------
    for data_key, ax in axes_to_plot:
        linestyle = LINESTYLE_MAP[data_key]

        for idx, (label, history) in enumerate(histories.items()):
            color = colors[idx % len(colors)]

            try:
                values = np.array(get_data(history, data_key))
            except Exception as e:
                print(f"Warning: Skipping {label} for {data_key}: {e}")
                continue

            iterations = np.arange(1, len(values) + 1)
            # compliance 标注 "$k = 2, C$" 等，volfrac 暂不注册（后面统一处理）
            if data_key == 'compliance':
                k_num = label.replace('k', '')
                legend_label = f"$k = {k_num}$, 柔顺度"
            else:
                legend_label = None
            ax.plot(iterations, values,
                    color=color, linestyle=linestyle,
                    linewidth=linewidth, label=legend_label)

    # ------------------------------------------
    # 5. 样式修饰
    # ------------------------------------------
    ax1.set_xlabel('迭代步数', fontproperties=FONT_ZH)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if plot_type == 'both':
        ax1.set_ylabel('柔顺度 $c$',    fontproperties=FONT_ZH)
        ax2.set_ylabel('体积分数 $V_f$', fontproperties=FONT_ZH)

        # 智能锁定右轴（体积分数）范围
        all_volfrac_arrays = []
        for history in histories.values():
            try:
                all_volfrac_arrays.append(np.array(get_data(history, 'volfrac')))
            except Exception:
                pass

        if all_volfrac_arrays:
            all_volfrac_flat = np.concatenate(all_volfrac_arrays)
            v_min, v_max = np.min(all_volfrac_flat), np.max(all_volfrac_flat)
            if (v_max - v_min) < 0.01:
                target = np.mean(all_volfrac_flat[-10:])
                margin = 0.05
                ax2.set_ylim(target - margin, target + margin)

        # 判断 volfrac 曲线是否几乎完全重叠
        # 标准：所有曲线逐点最大差值 < 阈值
        volfrac_overlap = False
        if len(all_volfrac_arrays) > 1:
            min_len = min(len(a) for a in all_volfrac_arrays)
            stacked = np.stack([a[:min_len] for a in all_volfrac_arrays])
            if np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0)) < 1e-3:
                volfrac_overlap = True

        # 构建 volfrac 图例
        from matplotlib.lines import Line2D
        lines, labels = ax1.get_legend_handles_labels()  # compliance 的 k=2/3/4

        if volfrac_overlap:
            # 三条重叠，只加一条灰色虚线
            lines  += [Line2D([0], [0], color='gray', linestyle='--', linewidth=linewidth)]
            labels += [r'体积分数 (所有 $k$)']
        else:
            # 分别添加各 k 值的 volfrac 图例
            for idx, (label, _) in enumerate(histories.items()):
                color = colors[idx % len(colors)]
                k_num = label.replace('k', '')
                lines  += [Line2D([0], [0], color=color, linestyle='--', linewidth=linewidth)]
                labels += [f"$k = {k_num}$, 体积分数"]

        ax1.legend(lines, labels,
                   loc='upper right', prop=FONT_ZH, framealpha=0.9, fancybox=False)

    elif plot_type == 'compliance':
        ax1.set_ylabel('柔顺度 $c$', fontproperties=FONT_ZH)
        ax1.legend(loc='upper right', prop=FONT_ZH, framealpha=0.9, fancybox=False)
    elif plot_type == 'volfrac':
        ax1.set_ylabel('体积分数 $V_f$', fontproperties=FONT_ZH)
        ax1.legend(loc='upper right', prop=FONT_ZH, framealpha=0.9, fancybox=False)

    # ------------------------------------------
    # 6. 标题与保存
    # ------------------------------------------
    if title:
        fig.suptitle(title, fontproperties=FONT_ZH, y=0.98)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"对比曲线已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

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
        """绘图代码"""
        # current_file = Path(__file__)
        # base_dir = current_file.parent.parent / 'vtu' 
        # base_dir = str(base_dir)
        # save_path = Path(f"{base_dir}/subsec5_6_2_fixed_fixed_beam2d/json")
        # save_path.mkdir(parents=True, exist_ok=True)    
    
        # histories = load_history_data(save_path, labels=['k2', 'k3', 'k4'])
        # plot_optimization_history_comparison(
        #                         histories,
        #                         save_path=f'{save_path}/convergence_comparison.png',
        #                         plot_type='both'
        #                     )

        #* 中点受载的两端固支梁
        P = -1
        E, nu = 1, 0.3
        plane_type = 'plane_stress' # plane_strain, plane_stress
        mesh_type = 'uniform_crisscross_tri'

        # domain = [0, 80, 0, 40]
        # rmin = 2.0
        # from soptx.model.cantilever_2d_lfem import CantileverMiddle2d
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
        # from soptx.model.cantilever_2d_lfem import Cantilever2dCorner
        # pde = Cantilever2dCorner(
        #             domain=domain,
        #             P=P, 
        #             E=E, nu=nu,
        #             plane_type=plane_type,
        #             load_width=None,
        #         )
        # nx, ny = 80, 50
        # volume_fraction = 0.4

        P = -3
        E, nu = 30, 0.4
        plane_type = 'plane_stress' # plane_strain, plane_stress
        domain = [0, 160, 0, 20]
        rmin = 2.4
        from soptx.model.fixed_fixed_beam_lfem import FixedFixedBeamCenterLoad2d
        pde = FixedFixedBeamCenterLoad2d(
                    domain=domain,
                    P=P, 
                    E=E, nu=nu,
                    plane_type=plane_type,
                    load_width=None,
                )
        nx, ny = 160, 20
        volume_fraction = 0.4

        interpolation_method = 'msimp'
        penalty_factor = 3.0
        void_youngs_modulus = 1e-9
        
        # 'element'
        density_location = 'element'
        relative_density = volume_fraction

        # 'standard', 'voigt', 'fast'
        assembly_method = 'fast'
        solve_method = 'mumps'

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
        
        space_degree = 1
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格
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

        max_iterations = 500
        change_tolerance = 1e-2
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
                f"优化算法={optimizer.__class__.__name__} , 最大迭代次数={max_iterations}, 收敛容限={change_tolerance} \n"
                f"体积分数约束={volume_fraction}, 惩罚因子={penalty_factor}, 空材料杨氏模量={void_youngs_modulus} \n" 
                f"过滤类型={filter_type}, 过滤半径={rmin} ")
        
        rho_opt, history = optimizer.optimize(design_variable=d, density_distribution=rho)

        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/test_subsec5_6_2_lfem")
        save_path.mkdir(parents=True, exist_ok=True)

        save_history_data(history=history, save_path=str(save_path/'json'), label=f'{space_degree}')

        
        save_optimization_history(design_mesh=design_variable_mesh, 
                                history=history, 
                                density_location=density_location,
                                disp_mesh=displacement_mesh,
                                save_path=str(save_path))
        plot_optimization_history(history, save_path=str(save_path))


        return rho_opt, history
    

    @run.register('test_subsec5_6_2_hzmfem')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        """绘图代码"""
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu' 
        base_dir = str(base_dir)
        save_path = Path(f"{base_dir}/subsec5_6_2_fixed_fixed_beam2d/json")
        save_path.mkdir(parents=True, exist_ok=True)    
    
        histories = load_history_data(save_path, labels=['k2', 'k3', 'k4'])
        plot_optimization_history_comparison(
                                histories,
                                save_path=f'{save_path}/convergence_comparison.png',
                                plot_type='both'
                            )
    
        #* 中点受载的两端固支梁
        # P = -1
        # E, nu = 1, 0.3
        # plane_type = 'plane_stress' # plane_strain, plane_stress
        # mesh_type = 'uniform_crisscross_tri'

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

        P = -3
        E, nu = 30, 0.4
        plane_type = 'plane_stress' # plane_strain, plane_stress
        mesh_type = 'uniform_crisscross_tri'
        domain = [0, 160, 0, 20]
        rmin = 2.4
        from soptx.model.fixed_fixed_beam_hzmfem import FixedFixedBeamCenterLoad2d
        pde = FixedFixedBeamCenterLoad2d(
                    domain=domain,
                    P=P, 
                    E=E, nu=nu,
                    plane_type=plane_type,
                    load_width=None,
                )
        nx, ny = 160, 20
        volume_fraction = 0.4

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
        
        space_degree = 4
        integration_order = space_degree*2 + 2 # 单元密度 + 三角形网格
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

        save_history_data(history=history, save_path=str(save_path/'json'), label=f'{space_degree}')
        
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
    test.run.set('test_subsec5_6_2_lfem') 
    rho_opt, history = test.run()