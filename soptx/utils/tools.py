import matplotlib.pyplot as plt

from fealpy.mesh import HomogeneousMesh

from soptx.utils.gauss_intergation_point_mapping import get_gauss_integration_point_mapping, get_gauss_integration_point_mapping_old

def plot_gauss_gauss_integration_point_density_old(mesh: HomogeneousMesh, 
                                            nx: int, ny: int,
                                            integration_order=3):

    GD = mesh.geo_dimension()
    NC = mesh.number_of_cells()
    qf = mesh.quadrature_formula(integration_order)
    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = ws.shape[0]
    ps_local = mesh.bc_to_point(bcs) # (NC, NQ, GD)
    ps_local_flat = ps_local.reshape(-1, GD) # (NC*NQ, GD)

    local_to_global, global_to_local  = get_gauss_integration_point_mapping_old(nx=nx, ny=ny,
                                        nq_per_dim=integration_order)
    
    ps_global = ps_local_flat[local_to_global] # (NC*NQ, GD)

    ps_local_test = ps_global[global_to_local].reshape(NC, NQ, GD) # (NC, NQ, GD)


    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ps_local_test.reshape(-1, GD), showindex=True, 
                color='k', marker='o', markersize=10, fontsize=12, fontcolor='r')
    mesh.find_cell(axes, showindex=True, markersize=16, fontsize=20, fontcolor='b')
    plt.show()


def plot_gauss_gauss_integration_point_density(mesh: HomogeneousMesh, 
                                            nx: int, ny: int,
                                            integration_order=3):

    GD = mesh.geo_dimension()
    qf = mesh.quadrature_formula(integration_order)
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps_local = mesh.bc_to_point(bcs) # (NC, NQ, GD)

    local_to_global, global_to_local  = get_gauss_integration_point_mapping(nx=nx, ny=ny,
                                        nq_per_dim=integration_order)
    ps_global = ps_local[local_to_global] # (NC*NQ, GD)
    ps_local_test = ps_global[global_to_local] # (NC, NQ, GD)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ps_local_test.reshape(-1, GD), showindex=True, 
                color='k', marker='o', markersize=10, fontsize=12, fontcolor='r')
    mesh.find_cell(axes, showindex=True, markersize=16, fontsize=20, fontcolor='b')
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

    nx, ny = 6, 2
    mesh = QuadrangleMesh.from_box(box=[0, nx, 0, ny], nx=nx, ny=ny)
    plot_gauss_gauss_integration_point_density(mesh, nx, ny, integration_order=3)