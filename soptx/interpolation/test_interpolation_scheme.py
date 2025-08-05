from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.mesh import QuadrangleMesh

class InterpolationSchemeTest():
    def __init__(self) -> None:
        pass

    def get_barycentric_coordinates(self, nx: int, ny: int) -> Tuple[Tuple[TensorLike, TensorLike], TensorLike]:
        """获取可视化插值点的重心坐标"""

        mesh = QuadrangleMesh.from_box(box=[-1, 1, -1, 1], nx=nx, ny=ny)
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
    
    def compute_density_derivative(self, interpolation_scheme, opt_mesh, interpolation_order, 
                                node_barycentric, target_node_index):
        """计算密度关于特定节点的导数"""
        rho_derivative = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=0,
                                                interpolation_order=interpolation_order,
                                            )
        
        # 只设置目标节点为1
        rho_derivative[target_node_index] = 1.0
        
        # 计算插值，这就是该节点对应的"形函数"分布，即导数
        derivative = rho_derivative(node_barycentric)
        
        return derivative
    
    @variantmethod('test_lagrange_interpolation_point_density')
    def run(self, interpolation_order: int) -> None:
        
        opt_mesh = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=1, ny=1)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location='lagrange_interpolation_point',
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho_interpolation_points = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=0,
                                                interpolation_order=interpolation_order,
                                            )
        
        rho_interpolation_points[1] = 1.0  # 左上角节点
        rho_interpolation_points[2] = 1.0  # 右下角节点

        nx, ny = 49, 49
        node_barycentric, node_cartesian = self.get_barycentric_coordinates(nx=nx, ny=ny)

        rho = rho_interpolation_points(node_barycentric) # ((nx+1)*(ny+1), )

        derivative_rho = self.compute_density_derivative(
                                                    interpolation_scheme=interpolation_scheme,
                                                    opt_mesh=opt_mesh,
                                                    interpolation_order=interpolation_order,
                                                    node_barycentric=node_barycentric,
                                                    target_node_index=1  # 左上角节点
                                                ) # ((nx+1)*(ny+1), )


        RHO = rho.reshape((nx+1, ny+1))
        DERIVATIVE_RHO = derivative_rho.reshape((nx+1, ny+1))

        XI, ETA = node_cartesian[:, 0].reshape((nx+1, ny+1)), node_cartesian[:, 1].reshape((nx+1, ny+1))

        import matplotlib.pyplot as plt
        import numpy as np

        # --- 5. 可视化结果 (使用发散色图以突显负值) ---
        fig = plt.figure(figsize=(16, 7))
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
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
        ax1.plot_surface(XI, ETA, np.zeros_like(RHO), alpha=0.2, color='gray') # 标示z=0平面

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
        ax2.plot_surface(XI, ETA, np.zeros_like(DERIVATIVE_RHO), alpha=0.2, color='gray') # 标示z=0平面

        plt.suptitle('Density Interpolation Test using 9-node Quadratic Lagrangian Shape Functions', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # --- 6. 打印出最小值以供验证 ---
        print(f"插值密度的最小值: {np.min(RHO):.4f}")
        print(f"密度对节点3导数的最小值: {np.min(DERIVATIVE_RHO):.4f}")

        print("------------")
        
        

        
if __name__ == "__main__":
    test = InterpolationSchemeTest()
    test.run.set('test_lagrange_interpolation_point_density')
    test.run(interpolation_order=2)