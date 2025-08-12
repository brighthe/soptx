from typing import Tuple, Union

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import QuadrangleMesh, HomogeneousMesh
from fealpy.functionspace import Function

from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
from soptx.interpolation.space import ShepardFunction


def get_barycentric_coordinates(nx: int, ny: int) -> Tuple[Tuple[TensorLike, TensorLike], TensorLike]:
    """获取可视化插值点的重心坐标"""

    domian_reference = [-1, 1, -1, 1]  # [-1, 1] x [-1, 1]
    mesh = QuadrangleMesh.from_box(box=domian_reference, nx=nx, ny=ny)
    node_cartesian = mesh.entity('node')

    # 提取唯一的坐标值
    xi_unique = bm.unique(node_cartesian[:, 0])  # shape (nx+1,)
    eta_unique = bm.unique(node_cartesian[:, 1])  # shape (ny+1,)
    
    # 分别计算重心坐标
    xi_barycentric = bm.concatenate([
        ((1 - xi_unique) / 2).reshape(-1, 1),
        ((1 + xi_unique) / 2).reshape(-1, 1)
    ], axis=1)  # shape (nx+1, 2)
    
    eta_barycentric = bm.concatenate([
        ((1 - eta_unique) / 2).reshape(-1, 1),
        ((1 + eta_unique) / 2).reshape(-1, 1)
    ], axis=1)  # shape (ny+1, 2)

    node_barycentric = (xi_barycentric, eta_barycentric)

    return node_barycentric, node_cartesian

def compute_derivative_density(
                            interpolation_scheme: MaterialInterpolationScheme, 
                            opt_mesh: HomogeneousMesh, 
                            interpolation_order: int, 
                            node_coords: Union[TensorLike, Tuple[TensorLike, TensorLike]],
                            target_node_index: int
                        ) -> Union[Function, ShepardFunction]:
        """计算插值后的密度关于特定节点的导数"""
        derivative_rho_ipoints = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=0,
                                                interpolation_order=interpolation_order,
                                            )
        # 只设置目标节点的导数值为 1
        derivative_rho_ipoints[target_node_index] = 1.0
        
        # 计算插值，这就是该节点对应的"形函数"分布，即导数
        if derivative_rho_ipoints.coordtype == 'barycentric':
            derivative_rho = derivative_rho_ipoints(node_coords)

        elif derivative_rho_ipoints.coordtype == 'cartesian':
            derivative_rho = derivative_rho_ipoints(node_coords)
        
        return derivative_rho

def plot_density_and_derivative(XI, ETA, RHO, DERIVATIVE_RHO, title_suffix=''):

    import matplotlib.pyplot as plt

    # --- 5. 可视化结果 (使用发散色图以突显负值) ---
    fig = plt.figure(figsize=(16, 7))
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    # 图a: 插值后的密度分布
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 使用'coolwarm'色图: 蓝色表示负值, 红色表示正值, 白色接近零
    surf1 = ax1.plot_surface(XI, ETA, RHO, cmap='coolwarm', edgecolor='none')
    ax1.set_title('(a) Interpolated Density Distribution $\\rho(\\xi, \\eta)$', fontsize=16)
    # ax1.set_xlabel('$\\xi$')
    # ax1.set_ylabel('$\\eta$')
    ax1.set_zlabel('Density Value')
    ax1.view_init(elev=30, azim=-120) # 调整视角以更好地观察负值区域
    fig.colorbar(surf1, shrink=0.5, aspect=10, label='Density Value')
    ax1.plot_surface(XI, ETA, bm.zeros_like(RHO), alpha=0.2, color='gray') # 标示z=0平面

    # 图b: 密度对某个节点密度的导数分布
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # 同样使用'coolwarm'色图
    surf2 = ax2.plot_surface(XI, ETA, DERIVATIVE_RHO, cmap='coolwarm', edgecolor='none')
    ax2.set_title('(b) Density Derivative w.r.t. Top-left Node $\\partial\\rho/\\partial\\rho_1$', fontsize=16)
    # ax2.set_xlabel('$\\xi$')
    # ax2.set_ylabel('$\\eta$')
    ax2.set_zlabel('Derivative Value')
    ax2.view_init(elev=30, azim=-60) # 调整视角
    fig.colorbar(surf2, shrink=0.5, aspect=10, label='Derivative Value')
    ax2.plot_surface(XI, ETA, bm.zeros_like(DERIVATIVE_RHO), alpha=0.2, color='gray') # 标示z=0平面

    plt.suptitle('Density Interpolation Test using 9-node Quadratic Lagrangian Shape Functions', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# import numpy as np

def plot_gauss_integration_point_density(mesh, rho_gip):
    """可视化四边形网格上高斯点密度 (底部画网格，点云按 rho 上色)"""

    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    NC, NQ = rho_gip.shape

    # 1) 推断 1D 点数 & 求积
    n = int(round(bm.sqrt(NQ)))
    if n * n != NQ:
        raise ValueError(f"NQ={NQ} 不是平方数，无法推断 tensor-product Gauss 点。")
    integrator_order = n  # 与你当前接口匹配：order=2 -> 3x3

    qf = mesh.quadrature_formula(integrator_order)
    bcs, _ = qf.get_quadrature_points_and_weights()  # 这里只需要 bcs

    # 2) 形函数与物理坐标（按顶点顺序修正列置换）
    N = mesh.shape_function(bcs)              # (NQ, 4)
    node = mesh.entity('node')                # (NN, 2)
    cell = mesh.entity('cell')                # (NC, 4)
    xy   = node[cell]                         # (NC, 4, 2)

    perm = [0, 1, 3, 2]                       # [BL, BR, TR, TL] -> [BL, BR, TL, TR]
    ps = bm.einsum('qj, cjd -> cqd', N[:, perm], xy)   # (NC, NQ, 2)

    # 3) 生成网格边线（去重），构造 LineCollection
    e01 = cell[:, [0, 1]]
    e12 = cell[:, [1, 2]]
    e23 = cell[:, [2, 3]]
    e30 = cell[:, [3, 0]]
    edges = bm.concatenate([e01, e12, e23, e30], axis=0)
    edges_sorted = bm.sort(edges, axis=1)
    uniq_idx = bm.unique(edges_sorted, axis=0, return_index=True)[1]
    edges_unique = edges[uniq_idx]
    segments = node[edges_unique]             # (NE, 2, 2)

    # 4) 画图
    xy_pts = ps.reshape(-1, 2)                # (NC*NQ, 2)
    c_vals = rho_gip.reshape(-1)          # (NC*NQ,)

    fig, ax = plt.subplots(figsize=(7, 7))

    # 4.1) 先画网格（浅灰色）
    lc = LineCollection(segments, linewidths=0.6, colors=(0, 0, 0, 0.25), zorder=0)
    ax.add_collection(lc)

    # 4.2) 再画点云 (按 rho 上色, 0=白, 1=黑，中间灰)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    sc = ax.scatter(xy_pts[:, 0], xy_pts[:, 1], c=c_vals, s=14,
                    edgecolors='none', zorder=2, cmap='gray_r', norm=norm)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(sc, ax=ax, label=r'$\rho$')

    plt.tight_layout()
    plt.show()

