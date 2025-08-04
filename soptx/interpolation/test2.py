import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 定义4节点一次(双线性)拉格朗日单元的形函数 ---
def shape_functions_q4(xi, eta):
    """返回给定局部坐标(xi, eta)处的4个形函数值"""
    N = np.zeros(4)
    N[0] = 0.25 * (1 - xi) * (1 - eta)  # Node at (-1, -1)
    N[1] = 0.25 * (1 + xi) * (1 - eta)  # Node at (1, -1)
    N[2] = 0.25 * (1 + xi) * (1 + eta)  # Node at (1, 1)
    N[3] = 0.25 * (1 - xi) * (1 + eta)  # Node at (-1, 1)
    return N

# --- 2. 设置节点密度值 ---
# 节点编号顺序: 0(-1,-1), 1(1,-1), 2(1,1), 3(-1,1)
# 我们将左上角 (节点3) 和右下角 (节点1) 的密度设为1，其余为0
rho_nodes = np.zeros(4)
rho_nodes[1] = 1.0  # 右下角节点
rho_nodes[3] = 1.0  # 左上角节点

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
        N = shape_functions_q4(xi, eta)
        
        # 计算插值密度
        RHO[i, j] = np.dot(N, rho_nodes)
        
        # 密度对左上角节点(节点3)的导数就是N[3]的值
        DERIVATIVE_RHO_1[i, j] = N[3]

# --- 5. 可视化结果 ---
fig = plt.figure(figsize=(16, 7))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 图a: 插值后的密度分布
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(XI, ETA, RHO, cmap='viridis', edgecolor='none')
ax1.set_title('图(a): 插值后的密度分布 $\\rho(\\xi, \\eta)$', fontsize=16)
ax1.set_xlabel('$\\xi$')
ax1.set_ylabel('$\\eta$')
ax1.set_zlabel('密度 $\\rho$')
ax1.set_zlim(-0.1, 1.1) # 设置Z轴范围以确认没有负值
fig.colorbar(surf1, shrink=0.5, aspect=10, label='密度值')

# 图b: 密度对节点3密度的导数分布
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(XI, ETA, DERIVATIVE_RHO_1, cmap='plasma', edgecolor='none')
ax2.set_title('图(b): 密度对左上角节点的导数 $\\partial\\rho/\\partial\\rho_3$', fontsize=16)
ax2.set_xlabel('$\\xi$')
ax2.set_ylabel('$\\eta$')
ax2.set_zlabel('导数值')
ax2.set_zlim(-0.1, 1.1) # 设置Z轴范围以确认没有负值
fig.colorbar(surf2, shrink=0.5, aspect=10, label='导数值')

plt.suptitle('使用4节点一次拉格朗日形函数插值密度的测试', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- 6. 打印出最小值以供验证 ---
print(f"插值密度的最小值: {np.min(RHO):.4f}")
print(f"密度对节点3导数的最小值: {np.min(DERIVATIVE_RHO_1):.4f}")

