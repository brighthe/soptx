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