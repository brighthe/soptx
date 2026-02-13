from fealpy.backend import backend_manager as bm

import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from time import time
from typing import List, Optional, Dict
from pathlib import Path

from fealpy.typing import TensorLike
from fealpy.mesh import StructuredMesh, HomogeneousMesh, SimplexMesh, TensorMesh
from fealpy.functionspace import Function

@dataclass
class OptimizationHistory:
    """优化过程的历史记录"""
    # 密度场历史
    physical_densities: List[TensorLike] = field(default_factory=list)
    # 目标函数值历史
    obj_values: List[float] = field(default_factory=list)
    # 约束函数值历史（如体积分数）
    con_values: List[float] = field(default_factory=list)
    # 迭代时间历史
    iteration_times: List[float] = field(default_factory=list)
    # 优化开始时间
    start_time: float = field(default_factory=time)
    # von Mises 应力历史（可选）
    von_mises_stresses: List[TensorLike] = field(default_factory=list)
    
    def log_iteration(self, 
                    iter_idx: int, 
                    obj_val: float, 
                    volfrac: float, 
                    change: float, 
                    penalty_factor: float, 
                    time_cost: float, 
                    physical_density: TensorLike,
                    von_mises_stress: Optional[TensorLike] = None,
                    verbose: bool = True
                ) -> None:
        """记录一次迭代的信息"""
        if isinstance(physical_density, Function):
            rho_Phys = physical_density.space.function(bm.copy(physical_density[:]))
        else:
            rho_Phys = bm.copy(physical_density[:])

        self.physical_densities.append(rho_Phys)
        self.obj_values.append(obj_val)
        self.con_values.append(volfrac)
        self.iteration_times.append(time_cost)

        if von_mises_stress is not None:
            self.von_mises_stresses.append(von_mises_stress)
        
        if verbose:
            print(f"Iteration: {iter_idx + 1}, "
                  f"Objective: {obj_val:.4f}, "
                  f"Volfrac: {volfrac:.4f}, "
                  f"Change: {change:.4f}, "
                  f"Penalty: {penalty_factor:.4f}, "
                  f"Time: {time_cost:.3f} sec")
    
    def get_total_time(self) -> float:
        """获取总优化时间"""
        return time() - self.start_time
    
    def get_average_iteration_time(self) -> float:
        """获取平均每次迭代时间（排除第一次）"""
        if len(self.iteration_times) <= 1:
            return 0.0
        return sum(self.iteration_times[1:]) / (len(self.iteration_times) - 1)
    
    def print_time_statistics(self) -> None:
        """打印时间统计信息"""
        total_time = self.get_total_time()
        avg_time = self.get_average_iteration_time()
        
        print("\nTime Statistics:")
        print(f"Total optimization time: {total_time:.3f} sec")
        if len(self.iteration_times) > 0:
            print(f"First iteration time: {self.iteration_times[0]:.3f} sec")
        if len(self.iteration_times) > 1:
            print(f"Average iteration time (excluding first): {avg_time:.3f} sec")
            print(f"Number of iterations: {len(self.iteration_times)}")
    
    def get_best_iteration(self, minimize: bool = True) -> int:
        """获取最优迭代的索引"""
        if not self.obj_values:
            return -1
        
        if minimize:
            return self.obj_values.index(min(self.obj_values))
        else:
            return self.obj_values.index(max(self.obj_values))
    
    def get_best_density(self, minimize: bool = True) -> Optional[TensorLike]:
        """获取最优迭代的密度场"""
        best_idx = self.get_best_iteration(minimize)
        if best_idx >= 0 and best_idx < len(self.physical_densities):
            return self.physical_densities[best_idx]
        return None

def save_optimization_history(mesh: HomogeneousMesh, 
                            history: OptimizationHistory, 
                            density_location: str, 
                            save_path: Optional[str]=None
                        ) -> None:
    """保存优化过程的所有迭代结果"""
    if save_path is None:
        return
    
    has_stress = (history.von_mises_stresses is not None)
    
    if history.von_mises_stresses:
        iterator = zip(history.physical_densities, history.von_mises_stresses)
    else:
        iterator = zip(history.physical_densities, [None]*len(history.physical_densities))

    for i, (physical_density, von_mises_stress) in enumerate(iterator):
        if density_location in ['element']:
            # 单分辨率单元密度情况：形状为 (NC, )
            mesh.celldata['density'] = physical_density
            
            if von_mises_stress is not None:
                mesh.celldata['von_mises'] = von_mises_stress

        elif density_location in ['node']:
            # 单分辨率节点密度情况：形状为 (NN, )
            mesh.nodedata['density'] = physical_density

        elif density_location in ['element_multiresolution']:
            # 多分辨率单元密度情况：形状为 (NC, n_sub)
            from soptx.analysis.utils import reshape_multiresolution_data
            n_sub = physical_density.shape[-1]
            n_sub_x, n_sub_y = int(bm.sqrt(n_sub)), int(bm.sqrt(n_sub))
            nx_displacement, ny_displacement = int(mesh.meshdata['nx'] / n_sub_x), int(mesh.meshdata['ny'] / n_sub_y)

            rho_phys = reshape_multiresolution_data(nx=nx_displacement, 
                                                    ny=ny_displacement, 
                                                    data=physical_density) # (NC*n_sub, )

            mesh.celldata['density'] = rho_phys

        elif density_location in ['node_multiresolution']:
            # 多分辨率节点密度情况：形状为 (NN, )
            mesh.nodedata['density'] = physical_density

        else:
            raise ValueError(f"不支持的密度数据维度：{physical_density.ndim}")
        
        if isinstance(mesh, StructuredMesh):
            mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vts")
        else:  
            mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vtu")


# def plot_optimization_history(history, save_path=None, show=True, title=None, 
#                             fontsize=20, figsize=(14, 10), linewidth=2.5,
#                             ):
#     """绘制优化过程中目标函数和约束函数的变化
    
#     Parameters
#     ----------
#     history : OptimizationHistory
#         优化历史记录
#     save_path : str, optional
#         保存路径，如不提供则不保存
#     show : bool, optional
#         是否显示图像，默认为 True
#     title : str, optional
#         图表标题，默认为 None
#     fontsize : int, optional
#         标签和刻度字体大小
#     figsize : tuple, optional
#         图形大小
#     linewidth : float, optional
#         线条宽度
#     """
#     import matplotlib.pyplot as plt
    
#     # 准备数据
#     iterations = bm.arange(1, len(history.obj_values) + 1)
#     obj_values = bm.array(history.obj_values)
#     con_values = bm.array(history.con_values)
    
#     # 创建图形
#     fig, ax1 = plt.subplots(figsize=figsize)
    
#     # 设置全局字体大小
#     plt.rcParams.update({'font.size': fontsize})
    
#     # 绘制目标函数曲线（左轴）
#     ax1.set_xlabel('Iteration', fontsize=fontsize+6)
#     ax1.set_ylabel('Compliance, $c$', color='red', fontsize=fontsize+6)
#     ax1.plot(iterations, obj_values, 'r-', label=r'$c$', linewidth=linewidth)
#     ax1.tick_params(axis='y', labelcolor='red', labelsize=fontsize)
#     ax1.tick_params(axis='x', labelsize=fontsize)
    
#     # 创建右轴
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Volfrac, $v_f$', color='blue', fontsize=fontsize+6)
#     ax2.plot(iterations, con_values, 'b--', label=r'$v_f$', linewidth=linewidth)
#     ax2.tick_params(axis='y', labelcolor='blue', labelsize=fontsize)

#     ax1.grid(True, linestyle='--', alpha=0.7)

#     # 添加标题（如果提供）
#     if title is not None:
#         plt.title(title, fontsize=fontsize+6, pad=20, fontweight='bold')
    
#     # 添加图例
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=fontsize+6)

#     plt.tight_layout()
    
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     if show:
#         plt.show()
#     else:
#         plt.close()

#####################################################
#                    绘图和数据保存工具函数
#####################################################

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

def save_history_data(
                    history: OptimizationHistory, 
                    save_path: str, 
                    label: str
                ) -> None:
    """
    保存 history 中用于对比的关键数据（轻量级 JSON 格式）
    
    Parameters
    ----------
    history : OptimizationHistory
        优化历史记录
    save_path : str
        保存目录路径
    label : str
        标签名，如 'k1', 'k2', 'p3' 等
    
    Examples
    --------
    >>> save_history_data(history, './results', label='k=1')
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 只保存绘图需要的标量数据
    data = {
        'label': label,
        'obj_values': [float(v) for v in history.obj_values],
        'con_values': [float(v) for v in history.con_values],
        'iteration_times': [float(t) for t in history.iteration_times],
    }
    
    filepath = save_dir / f"history_{label}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"History saved to: {filepath}")

def load_history_data(
                    save_path: str, 
                    labels: str | List[str]
                ) -> dict | Dict[str, dict]:
    """
    加载保存的 history 数据
    
    Parameters
    ----------
    save_path : str
        保存目录路径
    labels : str | List[str]
        单个标签名或标签名列表，如 'k=1' 或 ['k=1', 'k=2']
    
    Returns
    -------
    dict | Dict[str, dict]
        - 如果 labels 是字符串：返回单个 history 的字典
        - 如果 labels 是列表：返回 {label: history_data} 的字典
    
    Examples
    --------
    >>> # 加载单个
    >>> data = load_history_data('./results', labels='k=1')
    >>> 
    >>> # 加载多个
    >>> histories = load_history_data('./results', labels=['k=1', 'k=2'])
    """
    if isinstance(labels, str):
        # 加载单个
        filepath = Path(save_path) / f"history_{labels}.json"
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    else:
        # 加载多个
        histories = {}
        for label in labels:
            filepath = Path(save_path) / f"history_{label}.json"
            with open(filepath, 'r') as f:
                histories[label] = json.load(f)
        return histories
    
def plot_optimization_history(history, save_path=None, show=True, 
                            figsize=(10, 6), linewidth=2.5):
    """绘制目标函数与体积约束收敛曲线"""
    # ------------------------------------------
    # 数据准备
    # ------------------------------------------
    # 定义一个内部辅助函数来获取数据
    def get_data(obj, key):
        if isinstance(obj, dict):
            return obj[key]  # 字典方式访问
        else:
            return getattr(obj, key) # 对象属性方式访问

    # 获取数据
    try:
        obj_values = np.array(get_data(history, 'obj_values'))
        con_values = np.array(get_data(history, 'con_values'))
    except KeyError as e:
        print(f"数据解析错误: 找不到键值 {e}")
        return
    except AttributeError as e:
        print(f"数据解析错误: 对象缺少属性 {e}")
        return
    iterations = np.arange(1, len(obj_values) + 1)
    
    # 创建画布 (设置高 DPI 以满足印刷要求)c
    fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
    
    # ------------------------------------------
    # 绘制左轴：柔顺度 (Compliance)
    # ------------------------------------------
    # 直接使用全局配色字典
    color_c = SOPTX_COLORS['compliance'] 
    
    # 设置标签 (混合排版：中文使用 SimHei)
    ax1.set_xlabel('迭代步数', fontproperties=FONT_ZH)
    ax1.set_ylabel('柔顺度 $c$', color=color_c, fontproperties=FONT_ZH)
    
    # 绘制曲线
    l1, = ax1.plot(iterations, obj_values, color=color_c, linestyle='-', 
                   linewidth=linewidth, label='柔顺度 $c$')
    
    # 设置刻度颜色
    ax1.tick_params(axis='y', labelcolor=color_c)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 强制设置左轴刻度数值为 Times New Roman
    if FONT_EN:
        for label in ax1.get_xticklabels():
            label.set_fontproperties(FONT_EN)
        for label in ax1.get_yticklabels():
            label.set_fontproperties(FONT_EN)

    # ------------------------------------------
    # 绘制右轴：体积分数 (Volume Fraction)
    # ------------------------------------------
    ax2 = ax1.twinx()
    # 直接使用全局配色字典
    color_v = SOPTX_COLORS['volume']
    
    # 设置标签
    ax2.set_ylabel('体积分数 $V_f$', color=color_v, fontproperties=FONT_ZH)
    
    # 绘制曲线
    l2, = ax2.plot(iterations, con_values, color=color_v, linestyle='--', 
                   linewidth=linewidth, label='体积分数 $V_f$')
    
    ax2.tick_params(axis='y', labelcolor=color_v)
    
    # 强制设置右轴刻度数值为 Times New Roman
    if FONT_EN:
        for label in ax2.get_yticklabels():
            label.set_fontproperties(FONT_EN)

    # ------------------------------------------
    # 智能锁定右轴范围 (保持平稳美观)
    # ------------------------------------------
    v_min, v_max = np.min(con_values), np.max(con_values)
    if (v_max - v_min) < 0.01:
        target = np.mean(con_values[-10:]) 
        margin = 0.05 
        ax2.set_ylim(target - margin, target + margin)

    # ------------------------------------------
    # 图例与保存
    # ------------------------------------------
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    
    # 放置在右上角
    ax1.legend(lines, labels, loc='upper right', prop=FONT_ZH, framealpha=0.9)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_optimization_history_comparison(
    histories: dict,
    save_path=None,
    show=True,
    title=None,
    figsize=None,
    linewidth=2.0,
    colors=None,
    linestyles=None,
    plot_type='both'  # 'both', 'objective', 'volume'
):
    """
    绘制不同情形下的对比收敛曲线
    
    Parameters
    ----------
    histories : dict
        包含多个历史数据的字典, e.g., {'CPU': history_obj1, 'GPU': history_obj2}
    plot_type : str
        'both' (双图并排), 'objective' (仅目标函数), 'volume' (仅体积分数)
    """
    # ------------------------------------------
    # 1. 绘图参数设置
    # ------------------------------------------
    # 默认配色方案：使用 SOPTX 全局色 + 补充对比色
    if colors is None:
        # 顺序：红、蓝、绿、黑、橙 (用于区分不同的 Method)
        colors = [
            SOPTX_COLORS['compliance'], # 红色
            SOPTX_COLORS['volume'],     # 蓝色
            '#2ca02c',                # 绿色
            'black',                    # 黑色
            '#ff7f0e'                 # 橙色
        ]
    
    # 默认线型 (实线、虚线、点划线、点线)
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':'] 
    # 智能设置画布大小
    if figsize is None:
        if plot_type == 'both':
            figsize = (12, 5) # 双图并排，长宽比约 2.4:1
        else:
            figsize = (8, 5)  # 单图，长宽比 1.6:1 (接近黄金比例)

    # ------------------------------------------
    # 2. 辅助函数 (数据兼容性)
    # ------------------------------------------
    def get_data(obj, key):
        """兼容字典(离线数据)和对象(在线数据)"""
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    # ------------------------------------------
    # 3. 创建画布与子图
    # ------------------------------------------
    # 设置 dpi=600 以满足高清印刷需求
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=600)
        axes_to_plot = [('objective', ax1), ('volume', ax2)]
    elif plot_type == 'objective':
        fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
        axes_to_plot = [('objective', ax1)]
    elif plot_type == 'volume':
        fig, ax2 = plt.subplots(figsize=figsize, dpi=600)
        axes_to_plot = [('volume', ax2)]
    else:
        raise ValueError("Invalid plot_type. Choose 'both', 'objective', or 'volume'.")

    # ------------------------------------------
    # 4. 绘图主循环
    # ------------------------------------------
    for p_type, ax in axes_to_plot:
        # 确定数据键名和中文Y轴标签
        if p_type == 'objective':
            data_key = 'obj_values'
            y_label = '柔顺度 $c$'
        else:
            data_key = 'con_values'
            y_label = '体积分数 $V_f$'
        
        # 遍历所有历史数据 (例如 label='CPU', history=data)
        for idx, (label, history) in enumerate(histories.items()):
            # 循环获取颜色和线型
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            
            try:
                values = np.array(get_data(history, data_key))
            except Exception as e:
                print(f"Warning: Skipping {label} for {data_key}: {e}")
                continue
            
            # 假设迭代步从 1 开始
            iterations = np.arange(1, len(values) + 1)
            
            # 绘制曲线
            ax.plot(iterations, values, 
                    color=color, linestyle=linestyle, 
                    linewidth=linewidth, label=label)
        
        # ------------------------------------------
        # 5. 样式修饰 (核心规范化)
        # ------------------------------------------
        # 设置中文标签 (使用全局 SimHei 字体变量)
        ax.set_xlabel('迭代步数', fontproperties=FONT_ZH)
        ax.set_ylabel(y_label, fontproperties=FONT_ZH)
        
        # 设置网格 (半透明虚线)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 设置图例 (右上角，中文支持)
        # framealpha=0.9 防止遮挡背景网格
        # prop=FONT_ZH 确保图例中的中文(如果有)能显示，英文(CPU/GPU)也会使用该字体显示
        ax.legend(loc='upper right', prop=FONT_ZH, framealpha=0.9, fancybox=False)
        
        # --- 关键：强制设置刻度字体为 Times New Roman ---
        if FONT_EN:
            for label in ax.get_xticklabels():
                label.set_fontproperties(FONT_EN)
            for label in ax.get_yticklabels():
                label.set_fontproperties(FONT_EN)

    # ------------------------------------------
    # 6. 标题与保存
    # ------------------------------------------
    if title:
        # 如果有总标题，使用中文
        fig.suptitle(title, fontproperties=FONT_ZH, y=0.98)

    plt.tight_layout()
    
    if save_path:
        # 自动创建目录
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"对比曲线已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
