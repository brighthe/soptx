import matplotlib.pyplot as plt

from fealpy.mesh import HomogeneousMesh

def plot_integrator_point_density(mesh: HomogeneousMesh, integrator_order=2):

    qf = mesh.quadrature_formula(integrator_order)
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs) # (NC, NQ, GD)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ps.reshape(-1, 2), showindex=True, 
                color='k', marker='o', markersize=16, fontsize=20, fontcolor='r')
    plt.show()

def plot_interpolation_point_density(mesh: HomogeneousMesh, interpolation_order=2):

    ip = mesh.interpolation_points(p=interpolation_order)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ip.reshape(-1, 2), showindex=True, 
                color='k', marker='o', markersize=16, fontsize=20, fontcolor='r')
    
    plt.show()



if __name__ == "__main__":
    # 示例：如何使用 plot_intergartor 函数
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=1, ny=1)
    plot_integrator_point_density(mesh, integrator_order=3)