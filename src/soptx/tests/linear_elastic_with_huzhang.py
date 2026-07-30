
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from huzhang_fe_space_with_corner_relaxation import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator

from fealpy.decorator import cartesian

from fealpy.fem import BilinearForm,ScalarMassIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator,BoundaryFaceSourceIntegrator
from fealpy.fem import DivIntegrator
from fealpy.fem import BlockForm,LinearBlockForm

from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

from fealpy.solver import spsolve
from scipy.sparse import csr_matrix, coo_matrix, bmat, spdiags
from scipy.sparse.linalg import spsolve as scipy_spsolve 

import sys
import time

def displacement_boundary_condition(space, g : callable):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    ldof = space.dof.number_of_local_dofs()
    gdof = space.dof.number_of_global_dofs()

    if 'neumann' not in mesh.edgedata:
        bdedge = mesh.boundary_edge_flag()
    else:
        bdedge = mesh.edgedata['neumann']

    e2c = mesh.edge_to_cell()[bdedge]
    en  = mesh.edge_unit_normal()[bdedge]
    cell2dof = space.cell_to_dof()[e2c[:, 0]]
    NBF = bdedge.sum()

    cellmeasure = mesh.entity_measure('edge')[bdedge]
    qf = mesh.quadrature_formula(p+2, 'edge')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(bcs)

    bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]

    symidx = [[0, 1], [1, 2]]
    phin = bm.zeros((NBF, NQ, ldof, 2), dtype=space.ftype)
    gval = bm.zeros((NBF, NQ, 2), dtype=space.ftype)
    for i in range(3):
        flag = e2c[:, 2] == i
        phi = space.basis(bcsi[i])[e2c[flag, 0]]
        phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * en[flag, None, None], axis=-1)
        phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * en[flag, None, None], axis=-1)
        points = mesh.bc_to_point(bcsi[i])[e2c[flag, 0]]
        gval[flag] = g(points)

    b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
    cell2dof = space.cell_to_dof()[e2c[:, 0]]
    r = bm.zeros(gdof, dtype=phi.dtype)
    bm.add.at(r, cell2dof, b)

    return r

def solve(pde, N, p):
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]],
                    dtype=bm.float64)
    cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]],
                    dtype=bm.int32)
    mesh = TriangleMesh(node, cell)
    mesh.uniform_refine(N)
    # TODO 修改
    if not hasattr(mesh, 'data'):
        mesh.data = {}
    mesh.data['corner'] = node[:-1]

    space0 = HuZhangFESpace2d(mesh, p=p)

    space = LagrangeFESpace(mesh, p=p-1, ctype='D')
    space1 = TensorFunctionSpace(space, shape=(-1, 2))

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1

    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1))

    bform2 = BilinearForm((space1,space0))
    bform2.add_integrator(HuZhangMixIntegrator())

    M = bform1.assembly()
    B = bform2.assembly()
    M = M.to_scipy()
    B = B.to_scipy()

    # TODO 将通用的应力基函数修正为满足胡张单元角点稳定性条件的基函数
    TM = space0.TM
    # TM = bm.eye(gdof0)
    M = TM.T @ M @ TM
    B = TM.T @ B

    A = bmat([[M, B],
              [B.T, None]], format='csr')

    lform1 = LinearForm(space1)

    @cartesian
    def source(x, index=None):
        return pde.source(x)
    lform1.add_integrator(VectorSourceIntegrator(source=source))

    b = lform1.assembly()
    a = displacement_boundary_condition(space0, pde.displacement) # 转换前

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = TM.T@a
    F[gdof0:] = -b

    X = scipy_spsolve(A, F)

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

    # TODO 修改
    # p = int(sys.argv[1])
    p = 3

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                 ]
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    x, y = symbols('x y')

    pi = bm.pi 
    u0 = (sin(pi*x)*sin(pi*y))
    u1 = (sin(pi*x)*sin(pi*y))
    u0 = sin(5*x)*sin(7*y)
    u1 = cos(5*x)*cos(4*y)

    u = [u0, u1]
    pde = LinearElasticPDE(u, lambda0, lambda1)
    for i in range(maxit):
        N = i 
        sigmah, uh = solve(pde, N, p)
        mesh = sigmah.space.mesh

        e0 = mesh.error(uh, pde.displacement) 
        e1 = mesh.error(sigmah, pde.stress)

        h[i] = mesh.entity_measure('edge').max()
        errorMatrix[0, i] = e1
        errorMatrix[1, i] = e0 
        print(N, e0, e1)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























