from fealpy.mesh import StructuredMesh

def save_optimization_history(mesh, history, save_path: str):
    """保存优化过程的所有迭代结果
    
    Parameters
    ----------
    mesh : StructuredMesh or TetrahedronMesh
        有限元网格对象
    history : OptimizationHistory
        优化历史记录，包含每次迭代的密度场
    save_path : str
        保存路径
    """
    for i, density in enumerate(history.densities):
        mesh.celldata['density'] = density
        if isinstance(mesh, StructuredMesh):
            mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vts")
        else:  
            mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vtu")

def plot_optimization_history(history, save_path=None, show=True):
    """绘制优化过程中目标函数和约束函数的变化
    
    Parameters
    ----------
    history : OptimizationHistory
        优化历史记录
    save_path : str, optional
        保存路径，如不提供则不保存
    show : bool, optional
        是否显示图像，默认为 True
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 准备数据
    iterations = np.arange(1, len(history.obj_values) + 1)
    obj_values = np.array(history.obj_values)
    con_values = np.array(history.con_values)
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制目标函数曲线（左轴）
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Compliance, f', color='red')
    ax1.plot(iterations, obj_values, 'r-', label='f')
    ax1.tick_params(axis='y', labelcolor='red')
    
    # 创建右轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Area, g', color='blue')
    ax2.plot(iterations, con_values, 'b--', label='g')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加标题
    plt.title('Optimization History')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 创建放大子图
    # 找到适合放大的范围
    if len(iterations) > 20:
        start_idx = min(60, len(iterations) - 1)
        if start_idx > 0:  # 确保有足够的数据点
            end_idx = min(120, len(iterations))
            
            # 只有当有足够的数据点时才创建子图
            if end_idx - start_idx > 10:
                sub_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
                sub_ax.plot(iterations[start_idx:end_idx], obj_values[start_idx:end_idx], 'r-')
                sub_ax.set_xlim(iterations[start_idx], iterations[end_idx-1])
                # 设置 y 轴范围略大于数据范围
                if start_idx < end_idx:
                    y_min = min(obj_values[start_idx:end_idx]) * 0.999
                    y_max = max(obj_values[start_idx:end_idx]) * 1.001
                    sub_ax.set_ylim(y_min, y_max)
                sub_ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()