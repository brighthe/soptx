from fealpy.backend import backend_manager as bm

from dataclasses import dataclass, field
from time import time
from typing import List, Optional

from fealpy.typing import TensorLike
from fealpy.mesh import StructuredMesh, HomogeneousMesh

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
    
    def log_iteration(self, 
                    iter_idx: int, 
                    obj_val: float, 
                    volfrac: float, 
                    change: float, 
                    time_cost: float, 
                    physical_density: TensorLike,
                    verbose: bool = True) -> None:
        """记录一次迭代的信息"""
        self.physical_densities.append(bm.copy(physical_density))
        self.obj_values.append(obj_val)
        self.con_values.append(volfrac)
        self.iteration_times.append(time_cost)
        
        if verbose:
            print(f"Iteration: {iter_idx + 1}, "
                  f"Objective: {obj_val:.4f}, "
                  f"Volfrac: {volfrac:.4f}, "
                  f"Change: {change:.4f}, "
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
                            save_path: Optional[str]=None) -> None:
    """保存优化过程的所有迭代结果
    
    Parameters
    ----------
    mesh : 有限元网格对象
    history : 优化历史记录，包含每次迭代的物理密度场
    save_path : str, optional
        保存路径，如不提供则不保存，默认为 None
    """
    if save_path is None:
        return
        
    for i, physical_density in enumerate(history.physical_densities):
        
        # 检查密度数据的维度
        if physical_density.ndim == 2:
            # 高斯积分点密度情况：形状为 (NC, NQ)
            NC, NQ = physical_density.shape
            
            if isinstance(mesh, HomogeneousMesh):
                # 获取新网格（用于可视化）的尺寸
                nx_new = mesh.meshdata['nx']
                ny_new = mesh.meshdata['ny']
                
                                # 推断高斯积分点的排列：假设是正方形排列（如 3x3=9）
                sqrt_NQ = int(NQ**0.5)
                if sqrt_NQ * sqrt_NQ != NQ:
                    raise ValueError(f"高斯积分点数量 {NQ} 不是完全平方数，无法处理非正方形排列")
                
                # 推断原网格的尺寸
                # 原网格: nx_orig=60, ny_orig=20 → 1200个单元
                # 新网格: nx_new=180, ny_new=60 → 10800个单元
                # 每个原单元细分为3×3个新单元
                nx_orig = nx_new // sqrt_NQ  # 180 // 3 = 60
                ny_orig = ny_new // sqrt_NQ  # 60 // 3 = 20
                
                # 验证尺寸是否匹配
                if nx_orig * ny_orig != NC:
                    raise ValueError(f"原网格单元数 {NC} 与推断的网格尺寸 ({nx_orig}, {ny_orig}) 不匹配")
                
                # 创建一维密度数组
                density_1d = bm.zeros(nx_new * ny_new, **bm.context(physical_density))
                
                # 转换密度数据：将每个单元的NQ个高斯积分点密度映射到新网格
                for cell_idx in range(NC):
                    # 计算原单元在原网格中的位置
                    # 编号规则：先y后x
                    i_x = cell_idx // ny_orig  # x方向索引：cell_idx // 20
                    i_y = cell_idx % ny_orig   # y方向索引：cell_idx % 20
                    
                    # 按行优先顺序处理新网格中的3×3子区域
                    for new_row in range(sqrt_NQ):  # 新网格中的行：0,1,2
                        for new_col in range(sqrt_NQ):  # 新网格中的列：0,1,2
                            # 计算在新网格中的全局坐标
                            global_new_x = sqrt_NQ * i_x + new_row  # 全局行
                            global_new_y = sqrt_NQ * i_y + new_col  # 全局列
                            
                            # 计算新网格编号
                            new_cell_index = global_new_x * ny_new + global_new_y
                            
                            # 找到对应的高斯积分点编号（列优先编号）
                            # 新网格位置(new_row, new_col)对应原始高斯积分点的哪个编号？
                            quad_idx = new_col * sqrt_NQ + new_row  # 列优先：列*3+行
                            
                            # 确保索引在有效范围内
                            if 0 <= new_cell_index < len(density_1d):
                                density_1d[new_cell_index] = physical_density[cell_idx, quad_idx]
                            else:
                                raise IndexError(f"新网格索引 {new_cell_index} 超出范围 [0, {len(density_1d)})")
                
                # 将转换后的密度数据赋给网格
                mesh.celldata['density'] = density_1d
                
            else:
                raise NotImplementedError("高斯积分点密度可视化目前只支持 StructuredMesh")
                
        elif physical_density.ndim == 1:
            # 单元密度情况：形状为 (NC,)
            mesh.celldata['density'] = physical_density
            
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
    ax1.set_ylabel('Compliance, c', color='red', fontsize=fontsize+6)
    ax1.plot(iterations, obj_values, 'r-', label='c', linewidth=linewidth)
    ax1.tick_params(axis='y', labelcolor='red', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    
    # 创建右轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volume, v', color='blue', fontsize=fontsize+6)
    ax2.plot(iterations, con_values, 'b--', label='v', linewidth=linewidth)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=fontsize)

    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加标题（如果提供）
    if title is not None:
        plt.title(title, fontsize=fontsize+6, pad=20, fontweight='bold')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=fontsize+6)
    
    # 创建放大子图
    # 找到适合放大的范围
    if len(iterations) > 20:
        start_idx = min(60, len(iterations) - 1)
        if start_idx > 0:  # 确保有足够的数据点
            end_idx = min(120, len(iterations))
            
            # 只有当有足够的数据点时才创建子图
            if end_idx - start_idx > 10:
                sub_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
                sub_ax.plot(iterations[start_idx:end_idx], obj_values[start_idx:end_idx], 'r-', linewidth=linewidth)
                sub_ax.set_xlim(iterations[start_idx], iterations[end_idx-1])
                # 设置 y 轴范围略大于数据范围
                if start_idx < end_idx:
                    y_min = min(obj_values[start_idx:end_idx]) * 0.999
                    y_max = max(obj_values[start_idx:end_idx]) * 1.001
                    sub_ax.set_ylim(y_min, y_max)
                sub_ax.grid(True, linestyle='--', alpha=0.4)
                sub_ax.tick_params(labelsize=fontsize-2)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_intergartor(mesh: HomogeneousMesh, integrator_order=2):

    import matplotlib.pyplot as plt

    qf = mesh.quadrature_formula(integrator_order)
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs) # (NC, NQ, GD)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ps.reshape(-1, 2), showindex=True, 
                color='k', marker='o', markersize=16, fontsize=20, fontcolor='r')
    plt.show()

if __name__ == "__main__":
    # 示例：如何使用 plot_intergartor 函数
    from fealpy.mesh import QuadrangleMesh
    mesh = QuadrangleMesh.from_box(box=[0, 6, 0, 2], nx=6, ny=2)
    plot_intergartor(mesh, integrator_order=3)