from fealpy.backend import backend_manager as bm

class LinearElasticityHuZhangFEMModel():
    def __init__():
        pass

def set_pde():
    pass








import matplotlib.pyplot as plt





from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.functionspace.huzhang_fe_space import HuZhangFESpace
from fealpy.fem import BlockForm, BilinearForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
# from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.decorator import cartesian
from fealpy.solver import spsolve
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

from soptx.solver.huzhang_stress_integrator import HuZhangStressIntegrator
from soptx.solver.huzhang_mix_integrator import HuZhangMixIntegrator
from soptx.pde.test import LinearElasticPDE
from soptx.pde.linear_elasticity_2d import BoxTriHuZhangData2d

from sympy import symbols, sin, cos



def solve(pde, N, p):
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    # mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=N, ny=N, nz=N)

    GD = mesh.geo_dimension()

    # qf = mesh.quadrature_formula(p+3, 'cell')
    # bcs, ws = qf.get_quadrature_points_and_weights()
    # phi = LagrangeFESpace(mesh, p=p, ctype='C').basis(bcs)

    space0 = HuZhangFESpace(mesh, p=p)


    space = LagrangeFESpace(mesh, p=p-1, ctype='D')
    space1 = TensorFunctionSpace(space, shape=(-1, GD))

    # lambda0 = pde.lambda0
    # lambda1 = pde.lambda1
    lambda0, lambda1 = pde.stress_matrix_coefficient()

    gdof0 = space0.number_of_global_dofs()
    gdof = space.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1))

    bform2 = BilinearForm((space1, space0))
    bform2.add_integrator(HuZhangMixIntegrator())

    A = BlockForm([[bform1,   bform2],
                   [bform2.T, None]])
    A = A.assembly()

    lform1 = LinearForm(space1)
    @cartesian
    def source(x, index=None):
        return pde.source(x)
    # lform1.add_integrator(VectorSourceIntegrator(source=source))
    lform1.add_integrator(VectorSourceIntegrator(source=pde.body_force))

    b = lform1.assembly()

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[gdof0:] = -b

    X = spsolve(A, F, "scipy")

    sigmaval = X[:gdof0]
    uval = X[gdof0:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    uh = space1.function()
    uh[:] = uval
    return sigmah, uh


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 5
    p = 3 # p 需要大于等于 3

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                 ]
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    x, y = symbols('x y')

    pi = bm.pi 
    u0 = (sin(pi*x)*sin(pi*y))**2
    u1 = (sin(pi*x)*sin(pi*y))**2
    # u0 = sin(5*x)*sin(7*y)
    # u1 = cos(5*x)*cos(4*y)

    u = [u0, u1]
    # pde = LinearElasticPDE(u, lambda0, lambda1)
    pde = BoxTriHuZhangData2d(lam=1, mu=0.5)

    for i in range(maxit):
        N = 2**(i+1) 
        sigmah, uh = solve(pde, N, p)
        mesh = sigmah.space.mesh

        # e0 = mesh.error(uh, pde.displacement) 
        # e1 = mesh.error(sigmah, pde.stress)
        e0 = mesh.error(uh, pde.displacement_solution) 
        e1 = mesh.error(sigmah, pde.stress_solution)

        h[i] = 1/N
        errorMatrix[0, i] = e1
        errorMatrix[1, i] = e0 
        print(N, e0, e1)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()