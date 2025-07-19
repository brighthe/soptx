from fealpy.backend import backend_manager as bm
from soptx.pde.linear_elasticity_2d import BoxTriLagrangeData2d
class LinearElasticityLagrangeFEMModel():
    def __init__():
        pass

def set_pde():
    pass

def linear_system(self):
    GD = self.mesh.geo_dimension()
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(GD, -1))












from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.solver import cg, spsolve

from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator

import matplotlib.pyplot as plt

from soptx.utils.show import showmultirate, show_error_table
from soptx.solver import LinearElasticIntegrator

def test_linear_elasticity_with_fem(p, pde):
    """

    """
    maxit = 5
    errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{L_2}$']
    errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
    NDof = bm.zeros(maxit, dtype=bm.int32)
    h = bm.zeros(maxit, dtype=bm.float64)
    for i in range(maxit):
        N = 2**(i+1)
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, mesh.geo_dimension()))
        NDof[i] = tensor_space.number_of_global_dofs()
        print(f"gdof:{NDof[i]}")

        linear_elastic_material = LinearElasticMaterial(
                                        name='E1nu03', 
                                        elastic_modulus=pde.E, poisson_ratio=pde.nu, 
                                        hypo=pde.plane_type, 
                                        device=bm.get_device(mesh)
                                    )

        integrator_K = LinearElasticIntegrator(
                            material=linear_elastic_material, 
                            q=tensor_space.p+3, 
                            method=None)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
        integrator_F = VectorSourceIntegrator(
                            source=pde.body_force, 
                            q=tensor_space.p+3
                        )
        lform = LinearForm(tensor_space)    
        lform.add_integrator(integrator_F)
        F = lform.assembly()

        dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet_bc, 
                    threshold=None, 
                    method='interp')
        K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet_bc, check=True)

        uh = tensor_space.function()

        uh[:] = spsolve(K, F, solver='mumps')

        # L2 误差
        e0 = mesh.error(uh, pde.disp_solution)
        errorMatrix[0, i] = e0

        h[i] = 1 / N

        # u_exact = tensor_space.interpolate(pde.solution)
        # errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))

        if i < maxit-1:
            mesh.uniform_refine()

    print("errorMatrix:\n", errorType, "\n", errorMatrix)
    print("NDof:", NDof)
    print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)  # 刻度数字变大
    plt.xlabel('Mesh size $h$', fontsize=20)
    plt.ylabel('$L^2$ Error', fontsize=20)
    plt.show()


if __name__ == "__main__":
    from soptx.pde import PolyDisp2dData, BoxTriData2d
    from fealpy.mesh import TriangleMesh, QuadrangleMesh
    bm.set_backend('numpy')
    p = 3
    # pde = PolyDisp2dData()
    pde_linearform_rhs = BoxTriData2d()
    domain_x, domain_y = pde_linearform_rhs.domain()[1], pde_linearform_rhs.domain()[3]
    nx, ny = 10, 10
    # mesh = UniformMesh2d(extent=[0, nx, 0, ny], h=[domain_x/nx, domain_y/ny], origin=[0.0, 0.0])
    # cip = mesh.cell_to_ipoint(p=p)
    mesh = TriangleMesh.from_box(box=pde_linearform_rhs.domain(), nx=nx, ny=ny, device='cpu')
    # mesh = QuadrangleMesh.from_box(box=[0, domain_x, 0, domain_y], nx=nx, ny=ny)

    test_linear_elasticity_with_fem(pde_linearform_rhs, mesh, p, solver_type='mumps')
