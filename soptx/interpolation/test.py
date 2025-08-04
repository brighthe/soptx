import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 定义9节点二次拉格朗日(Lagrangian)单元的形函数 ---
# 这些是标准的Q9单元形函数，输入为局部坐标 (xi, eta)
def shape_functions_q9(xi, eta):
    """返回给定局部坐标(xi, eta)处的9个形函数值"""
    N = np.zeros(9)
    # 角点节点
    N[0] = 0.25 * xi * (xi - 1) * eta * (eta - 1) # Node at (-1, -1)
    N[1] = 0.25 * xi * (xi + 1) * eta * (eta - 1) # Node at (1, -1)
    N[2] = 0.25 * xi * (xi + 1) * eta * (eta + 1) # Node at (1, 1)
    N[3] = 0.25 * xi * (xi - 1) * eta * (eta + 1) # Node at (-1, 1)
    # 边中点节点
    N[4] = 0.5 * (1 - xi**2) * eta * (eta - 1)    # Node at (0, -1)
    N[5] = 0.5 * xi * (xi + 1) * (1 - eta**2)     # Node at (1, 0)
    N[6] = 0.5 * (1 - xi**2) * eta * (eta + 1)    # Node at (0, 1)
    N[7] = 0.5 * xi * (xi - 1) * (1 - eta**2)     # Node at (-1, 0)
    # 中心节点
    N[8] = (1 - xi**2) * (1 - eta**2)             # Node at (0, 0)
    return N

# --- 2. 设置节点密度值 ---
# 节点编号顺序: 0(-1,-1), 1(1,-1), 2(1,1), 3(-1,1), 4(0,-1), 5(1,0), 6(0,1), 7(-1,0), 8(0,0)
# 我们将左上角 (节点3) 和右下角 (节点1) 的密度设为1，其余为0
rho_nodes = np.zeros(9)
rho_nodes[1] = 1.0  # 右下角节点
rho_nodes[3] = 1.0  # 左上角节点
# 其他节点的密度都为0, 包括中心节点

# --- 3. 创建用于绘图的网格 ---
grid_points = 50
xi_vals = np.linspace(-1.0, 1.0, grid_points)
eta_vals = np.linspace(-1.0, 1.0, grid_points)
XI, ETA = np.meshgrid(xi_vals, eta_vals)

# --- 4. 计算网格上每一点的插值密度和导数 ---
RHO = np.zeros_like(XI)
# 导数即为左上角节点(节点3)对应的形函数 N[3](xi, eta)
DERIVATIVE_RHO_1 = np.zeros_like(XI)

for i in range(grid_points):
    for j in range(grid_points):
        xi = XI[i, j]
        eta = ETA[i, j]
        
        # 获取当前点的所有形函数值
        N = shape_functions_q9(xi, eta)
        
        # 计算插值密度
        RHO[i, j] = np.dot(N, rho_nodes)
        
        # 密度对左上角节点(节点3)的导数就是N[3]的值
        DERIVATIVE_RHO_1[i, j] = N[3]

# --- 5. 可视化结果 (使用发散色图以突显负值) ---
fig = plt.figure(figsize=(16, 7))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 图a: 插值后的密度分布
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# 使用'coolwarm'色图: 蓝色表示负值, 红色表示正值, 白色接近零
surf1 = ax1.plot_surface(XI, ETA, RHO, cmap='coolwarm', edgecolor='none')
ax1.set_title('图(a): 插值后的密度分布 $\\rho(\\xi, \\eta)$', fontsize=16)
ax1.set_xlabel('$\\xi$')
ax1.set_ylabel('$\\eta$')
ax1.set_zlabel('密度 $\\rho$')
ax1.view_init(elev=30, azim=-120) # 调整视角以更好地观察负值区域
fig.colorbar(surf1, shrink=0.5, aspect=10, label='密度值')
ax1.plot_surface(XI, ETA, np.zeros_like(RHO), alpha=0.2, color='gray') # 标示z=0平面

# 图b: 密度对节点3密度的导数分布
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# 同样使用'coolwarm'色图
surf2 = ax2.plot_surface(XI, ETA, DERIVATIVE_RHO_1, cmap='coolwarm', edgecolor='none')
ax2.set_title('图(b): 密度对左上角节点的导数 $\\partial\\rho/\\partial\\rho_3$', fontsize=16)
ax2.set_xlabel('$\\xi$')
ax2.set_ylabel('$\\eta$')
ax2.set_zlabel('导数值')
ax2.view_init(elev=30, azim=-60) # 调整视角
fig.colorbar(surf2, shrink=0.5, aspect=10, label='导数值')
ax2.plot_surface(XI, ETA, np.zeros_like(DERIVATIVE_RHO_1), alpha=0.2, color='gray') # 标示z=0平面

plt.suptitle('使用9节点二次拉格朗日形函数插值密度的测试 (增强可视化)', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- 6. 打印出最小值以供验证 ---
print(f"插值密度的最小值: {np.min(RHO):.4f}")
print(f"密度对节点3导数的最小值: {np.min(DERIVATIVE_RHO_1):.4f}")
