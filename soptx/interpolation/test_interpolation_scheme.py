from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod

class InterpolationSchemeTest():
    def __init__(self) -> None:
        pass

    def get_barycentric_coordinates(self, node):
        # 提取唯一的坐标值
        xi_unique = bm.unique(node[:, 0])  # shape (nx+1,)
        eta_unique = bm.unique(node[:, 1])  # shape (ny+1,)
        
        # 分别计算重心坐标
        xi_barycentric = bm.concatenate([
            ((1 - xi_unique) / 2).reshape(-1, 1),
            ((1 + xi_unique) / 2).reshape(-1, 1)
        ], axis=1)  # shape (nx+1, 2)
        
        eta_barycentric = bm.concatenate([
            ((1 - eta_unique) / 2).reshape(-1, 1),
            ((1 + eta_unique) / 2).reshape(-1, 1)
        ], axis=1)  # shape (ny+1, 2)
        
        return (xi_barycentric, eta_barycentric)

    @variantmethod('test')
    def run(self, 
            interpolation_order: int, 
        ) -> None:

        # 参数设置
        nx, ny = 1, 1
        
        # 设置 pde 和网格
        from fealpy.mesh import QuadrangleMesh


        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        pde.init_mesh.set('uniform_quad')

        opt_mesh = QuadrangleMesh.from_box(box=[0, nx, 0, ny], nx=nx, ny=ny)

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

        rho_nodes = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=0,
                                                interpolation_order=interpolation_order,
                                            )
        
        rho_nodes[1] = 1.0  # 左上角节点
        rho_nodes[2] = 1.0  # 右下角节点

        nx, ny = 49, 49
        show_mesh = QuadrangleMesh.from_box(box=[-1, 1, -1, 1], nx=nx, ny=ny)
        node_cartesian = show_mesh.entity('node')
        node_barycentric = self.get_barycentric_coordinates(node=node_cartesian)

        rho = rho_nodes(node_barycentric)

        RHO = rho.reshape((nx+1, ny+1))
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
        ax1.set_title('图(a): 插值后的密度分布 $\\rho(\\xi, \\eta)$', fontsize=16)
        ax1.set_xlabel('$\\xi$')
        ax1.set_ylabel('$\\eta$')
        ax1.set_zlabel('密度 $\\rho$')
        ax1.view_init(elev=30, azim=-120) # 调整视角以更好地观察负值区域
        fig.colorbar(surf1, shrink=0.5, aspect=10, label='密度值')
        ax1.plot_surface(XI, ETA, np.zeros_like(RHO), alpha=0.2, color='gray') # 标示z=0平面

        # # 图b: 密度对节点3密度的导数分布
        # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # # 同样使用'coolwarm'色图
        # surf2 = ax2.plot_surface(XI, ETA, DERIVATIVE_RHO_1, cmap='coolwarm', edgecolor='none')
        # ax2.set_title('图(b): 密度对左上角节点的导数 $\\partial\\rho/\\partial\\rho_3$', fontsize=16)
        # ax2.set_xlabel('$\\xi$')
        # ax2.set_ylabel('$\\eta$')
        # ax2.set_zlabel('导数值')
        # ax2.view_init(elev=30, azim=-60) # 调整视角
        # fig.colorbar(surf2, shrink=0.5, aspect=10, label='导数值')
        # ax2.plot_surface(XI, ETA, np.zeros_like(DERIVATIVE_RHO_1), alpha=0.2, color='gray') # 标示z=0平面

        plt.suptitle('使用9节点二次拉格朗日形函数插值密度的测试 (增强可视化)', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # --- 6. 打印出最小值以供验证 ---
        print(f"插值密度的最小值: {np.min(RHO):.4f}")
        # print(f"密度对节点3导数的最小值: {np.min(DERIVATIVE_RHO_1):.4f}")

        print("------------")
        
        

        
if __name__ == "__main__":
    test = InterpolationSchemeTest()
    test.run.set('test')
    test.run(interpolation_order=2)