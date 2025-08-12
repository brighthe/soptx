from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Literal
from fealpy.decorator import variantmethod, barycentric, cartesian
from fealpy.mesh import QuadrangleMesh, HomogeneousMesh
from fealpy.functionspace import Function

from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
from soptx.interpolation.tools import get_barycentric_coordinates, compute_derivative_density, plot_density_and_derivative
from soptx.interpolation.space import ShepardFunction

class InterpolationSchemeTest():
    def __init__(self) -> None:
        pass
    
    @variantmethod('test_point_density')
    def run(self, density_location: str, interpolation_order: int) -> None:
        
        domain_physical = [0, 1, 0, 1]
        opt_mesh = QuadrangleMesh.from_box(box=domain_physical, nx=1, ny=1)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho_ipoints = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=0,
                                                interpolation_order=interpolation_order,
                                            )
        
        rho_ipoints[1] = 1.0  # 左上角节点
        rho_ipoints[2] = 1.0  # 右下角节点

        nx, ny = 49, 49

        if density_location == 'lagrange_interpolation_point':

            node_barycentric, node_cartesian = get_barycentric_coordinates(nx=nx, ny=ny)
            node_coords = node_barycentric
            plot_coords  = node_cartesian

        elif density_location == 'shepard_interpolation_point':
            
            mesh = QuadrangleMesh.from_box(box=domain_physical, nx=nx, ny=ny)
            node_coords = mesh.entity('node')
            plot_coords = node_coords

        # 插值后的密度值
        rho = rho_ipoints(node_coords) # ((nx+1)*(ny+1), )

        # 插值后的密度导数值
        derivative_rho = compute_derivative_density(
                                                interpolation_scheme=interpolation_scheme,
                                                opt_mesh=opt_mesh,
                                                interpolation_order=interpolation_order,
                                                node_coords=node_coords,
                                                target_node_index=1  # 左上角节点
                                            ) # ((nx+1)*(ny+1), )

        print(f"插值密度的最小值: {bm.min(rho):.4f}")
        print(f"密度对某个单一节点导数的最小值: {bm.min(derivative_rho):.4f}")

        RHO = rho.reshape((nx+1, ny+1))
        DERIVATIVE_RHO = derivative_rho.reshape((nx+1, ny+1))

        XI, ETA = plot_coords[:, 0].reshape((nx+1, ny+1)), plot_coords[:, 1].reshape((nx+1, ny+1))

        plot_density_and_derivative(
                                XI, ETA, RHO, DERIVATIVE_RHO,
                                title_suffix=f' ({density_location}, p={interpolation_order})'
                            )


        print("------------")
        
    
    @run.register('test_dual_mesh')
    def run(self, density_location: str) -> None:

        relative_density = 0.5
        integrator_order = 2

        domain_physical = [0, 1, 0, 1]
        opt_mesh = QuadrangleMesh.from_box(box=domain_physical, nx=10, ny=10)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho_ipoints = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=relative_density,
                                                integrator_order=integrator_order
                                            )
        rho_ipoints[4] = 1.0  # 左上角节点

        from soptx.interpolation.tools import plot_gauss_integration_point_density
        plot_gauss_integration_point_density(opt_mesh, rho_ipoints)



        print("-------------")
        
if __name__ == "__main__":
    test = InterpolationSchemeTest()
    # test.run.set('test_point_density')
    # test.run(density_location='lagrange_interpolation_point', interpolation_order=2)

    test.run.set('test_dual_mesh')
    test.run(density_location='gauss_integration_point')