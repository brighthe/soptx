
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
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

def plot_on_tri_lattice(x, y, z, axes):
    """
    Plot values z given at points (x,y) that form a triangular lattice (or arbitrary scattered points).
    Produces two figures: a 3D surface and a 2D filled contour.
    Inputs:
      x, y, z : 1D arrays of the same length
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    assert x.shape == y.shape == z.shape, "x, y, z must be same length"

    # Build triangulation (matplotlib will triangulate the scattered points)
    tri = mtri.Triangulation(x, y)

    # --- 3D surface ---
    surf = axes.plot_trisurf(tri, z, linewidth=0.2, edgecolor='gray')  # no explicit color map
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

def plot_function(uh, u, com):

    fig1 = plt.figure(figsize=(8,6))
    axes = fig1.add_subplot(111, projection='3d')

    space = uh.space
    mesh = space.mesh
    NC = mesh.number_of_cells()

    p = 10
    bcs = bm.multi_index_matrix(p, 2)/p
    points = mesh.bc_to_point(bcs)
    vals0 = uh(bcs)[..., com]
    vals1 = u(points)[..., com]
    for c in range(NC):
        ptsc = points[c]
        plot_on_tri_lattice(ptsc[:,0], ptsc[:,1], vals0[c], axes)
        plot_on_tri_lattice(ptsc[:,0], ptsc[:,1], vals1[c], axes)

def mass_matrix(space):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    phi = space.basis(bcs)

    num = bm.array([1, 2, 1], dtype=space.ftype)
    A = bm.einsum('q, c, cqld, cqmd, d->clm', ws, cellmeasure, phi, phi, num)

    cell2dof = space.cell_to_dof()
    I = bm.broadcast_to(cell2dof[:, None], A.shape)
    J = bm.broadcast_to(cell2dof[..., None], A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof), dtype=phi.dtype)
    return A

def source(space, sig):
    # TODO 计算 L2 投影的右端项
    mesh = space.mesh

    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    phi = space.basis(bcs)

    points = mesh.bc_to_point(bcs)
    sigval = sig(points) 

    num = bm.array([1, 2, 1], dtype=space.ftype)
    val = bm.einsum('cqld, cqd, d, c, q->cl', phi, sigval, num, cellmeasure, ws)

    cell2dof = space.cell_to_dof()

    F = bm.zeros(gdof, dtype=space.ftype)
    bm.add.at(F, cell2dof, val)
    return F

def dirichlet_bc(space, A, F, g):
    # TODO 处理应力边界条件 (本质边界)
    uh, isbddof = space.set_dirichlet_bc(g)

    F = F - A@uh
    F[isbddof] = uh[isbddof]
    gdof = space.number_of_global_dofs()

    bdIdx = bm.zeros(gdof, dtype=bm.int32)
    bdIdx[isbddof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    A = T@A@T + Tbd

    return A, F

def solve(pde, N, p):
    #* 第一种网格
    # node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]],
    #                 dtype=bm.float64)
    # cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]],
    #                 dtype=bm.int32)
    # mesh = TriangleMesh(node, cell)
    # mesh.uniform_refine(N)

    # if not hasattr(mesh, 'data'):
    #     mesh.data = {}
    # mesh.data['corner'] = node[:-1]

    #* 第二种网格
    from fealpy.mesh import QuadrangleMesh
    nx, ny = 2, 2
    domain = [0, 1, 0, 1]
    node = bm.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0]], dtype=bm.float64)
    cell = bm.array([[0, 1, 2, 3]], dtype=bm.int32)
    qmesh = QuadrangleMesh(node, cell).from_box(box=domain, nx=nx, ny=ny)
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    isLeftCell = bm.zeros((nx, ny), dtype=bm.bool)
    isLeftCell[0, 0::2] = True
    isLeftCell[1, 1::2] = True
    if nx > 2:
        isLeftCell[2::2, :] = isLeftCell[0, :]
    if ny > 3:
        isLeftCell[3::2, :] = isLeftCell[1, :]
    isLeftCell = isLeftCell.reshape(-1)
    lcell = cell[isLeftCell]
    rcell = cell[~isLeftCell]
    import numpy as np
    newCell = np.r_['0',
                    lcell[:, [1, 2, 0]],
                    lcell[:, [3, 0, 2]],
                    rcell[:, [0, 1, 3]],
                    rcell[:, [2, 3, 1]]]
    mesh = TriangleMesh(node, newCell)
    mesh.uniform_refine(N)

    x_min, x_max = domain[0], domain[1]
    y_min, y_max = domain[2], domain[3]

    is_x_bd = (bm.abs(node[:, 0] - x_min) < 1e-9) | (bm.abs(node[:, 0] - x_max) < 1e-9)
    is_y_bd = (bm.abs(node[:, 1] - y_min) < 1e-9) | (bm.abs(node[:, 1] - y_max) < 1e-9)
    is_corner = is_x_bd & is_y_bd
    corner_coords = node[is_corner]
    if not hasattr(mesh, 'data'):
        mesh.data = {}
    mesh.data['corner'] = corner_coords

    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # mesh.add_plot(axes)
    # mesh.find_node(axes, showindex=True)
    # mesh.find_edge(axes, showindex=True)
    # mesh.find_cell(axes, showindex=True)
    # plt.show()

    space = HuZhangFESpace2d(mesh, p=p)
    _, corner2dof = space.dof.node_to_internal_dof()
    gdof0 = space.number_of_global_dofs()

    M = mass_matrix(space) 
    F = source(space, pde.stress) 

    # TODO 将通用的应力基函数修正为满足胡张单元角点稳定性条件的基函数
    TM = space.TM
    M = TM.T @ M @ TM
    F = TM.T @ F

    M, F = dirichlet_bc(space, M, F, pde.stress)

    sigmah = space.function()
    sigmah[:] = scipy_spsolve(M, F)

    return sigmah


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 4
    
    # TODO 修改
    # p = int(sys.argv[1])
    p = 3

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$'
                 ]
    errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
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
        N = i+1 
        sigmah = solve(pde, N, p)
        mesh = sigmah.space.mesh

        e1 = mesh.error(sigmah, pde.stress)

        h[i] = mesh.entity_measure('edge').max()
        errorMatrix[0, i] = e1
        print(N, e1)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























