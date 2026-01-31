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

def plot_optimization_history(history, save_path=None, show=True, title=None, 
                            fontsize=20, figsize=(14, 10), linewidth=2.5,
                            ):
    """绘制优化过程中目标函数和约束函数的变化
    
    Parameters
    ----------
    history : OptimizationHistory
        优化历史记录
    save_path : str, optional
        保存路径，如不提供则不保存
    show : bool, optional
        是否显示图像，默认为 True
    title : str, optional
        图表标题，默认为 None
    fontsize : int, optional
        标签和刻度字体大小
    figsize : tuple, optional
        图形大小
    linewidth : float, optional
        线条宽度
    """
    import matplotlib.pyplot as plt
    
    # 准备数据
    iterations = bm.arange(1, len(history.obj_values) + 1)
    obj_values = bm.array(history.obj_values)
    con_values = bm.array(history.con_values)
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': fontsize})
    
    # 绘制目标函数曲线（左轴）
    ax1.set_xlabel('Iteration', fontsize=fontsize+6)
    ax1.set_ylabel('Compliance, $c$', color='red', fontsize=fontsize+6)
    ax1.plot(iterations, obj_values, 'r-', label=r'$c$', linewidth=linewidth)
    ax1.tick_params(axis='y', labelcolor='red', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    
    # 创建右轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volfrac, $v_f$', color='blue', fontsize=fontsize+6)
    ax2.plot(iterations, con_values, 'b--', label=r'$v_f$', linewidth=linewidth)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=fontsize)

    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加标题（如果提供）
    if title is not None:
        plt.title(title, fontsize=fontsize+6, pad=20, fontweight='bold')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=fontsize+6)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


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

def plot_optimization_history_comparison(
    histories: Dict[str, dict],
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    # --- 论文绘图关键参数 ---
    fontsize: int = 14,          # 论文推荐 12-14，保证缩放后可读
    figsize: Optional[tuple] = None, # 默认为 None，由代码内部决定最佳比例
    linewidth: float = 2.0,      # 线宽 2.0 在论文中视觉效果最佳
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    plot_type: str = 'both',     # 'both', 'objective', 'volume'
):
    """
    绘制符合博士学位论文排版标准的收敛曲线（长方形黄金比例）
    """
    
    # 1. 颜色与线型：选用学术界常用的高对比度配色
    if colors is None:
        # 经典的红、蓝、黑、绿，打印成黑白也能区分灰度
        colors = ['#d62728', '#1f77b4', 'black', '#2ca02c', '#ff7f0e'] 
    if linestyles is None:
        # 实线、虚线、点划线，区分度高
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))] 

    # 2. 全局字体设置：使用 Times New Roman 或类似衬线体
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'], # 论文标准字体
        'font.size': fontsize,
        'mathtext.fontset': 'stix',        # 公式字体与 Times 搭配最好
        'axes.grid': True,                 # 默认开启网格
        'grid.alpha': 0.4,                 # 网格淡一点
        'grid.linestyle': '--'
    })

    # 3. 智能设置画布大小 (figsize) - 核心修改
    # A4纸内容宽度通常在 15-16cm 左右。
    # Matplotlib 默认 dpi=100，所以 6.4 inch ≈ 16cm。
    if figsize is None:
        if plot_type == 'both':
            # 双图并排：宽一点，高保持黄金比
            # 12 inch 宽，5 inch 高 -> 每个子图接近 1.2:1
            figsize = (12, 5) 
        else:
            # 单图：经典的 4:3 或 黄金比例
            # 8 inch * 2.54 = 20cm (稍大，适合缩小插入), 高 5 inch
            figsize = (8, 5) 

    # 创建画布
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes_to_plot = [('objective', ax1), ('volume', ax2)]
    elif plot_type == 'objective':
        fig, ax1 = plt.subplots(figsize=figsize)
        axes_to_plot = [('objective', ax1)]
    elif plot_type == 'volume':
        fig, ax2 = plt.subplots(figsize=figsize)
        axes_to_plot = [('volume', ax2)]
    else:
        raise ValueError("Invalid plot_type")

    # 4. 绘图循环
    for p_type, ax in axes_to_plot:
        # 设置数据键名和标签
        data_key = 'obj_values' if p_type == 'objective' else 'con_values'
        # y_label = 'Compliance, $C$' if p_type == 'objective' else 'Volume Fraction, $V_f$'
        y_label = 'Output displacement, $u_{out}$' if p_type == 'objective' else 'Volume Fraction, $V_f$'
        
        for idx, (label, history) in enumerate(histories.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            
            values = history[data_key]
            # 迭代步数通常从 0 或 1 开始，这里假设从 0 开始
            iterations = range(len(values))
            
            ax.plot(iterations, values, 
                    color=color, linestyle=linestyle, 
                    linewidth=linewidth, label=label)
        
        # 坐标轴修饰
        ax.set_xlabel('Iteration')
        ax.set_ylabel(y_label)
        
        # 图例设置：去掉边框背景，显得更干净，或者放在最佳位置
        ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fancybox=False)
        
        # 科学计数法：如果数值太大或太小（比如 Compliance），强制使用科学计数法
        if p_type == 'objective':
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))

    if title:
        fig.suptitle(title, fontweight='bold', y=0.98)

    plt.tight_layout()
    
    if save_path:
        # 保存为 PDF 或高 DPI 的 PNG
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()