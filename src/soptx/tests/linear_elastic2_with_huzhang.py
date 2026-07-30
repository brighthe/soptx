import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat, spdiags, csr_matrix
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, VectorSourceIntegrator
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.tools.show import show_error_table, showmultirate

from linear_elastic_pde import LinearElasticPDE
from linear_elastic_with_huzhang import displacement_boundary_condition 

# 定义本质应力边界处理函数 (参考 test_L2_projection.py)
def apply_stress_boundary_condition(space, A, F, g_stress, threshold=None):
    """
    处理本质应力边界条件 (Traction BC).
    在混合元中，指定 sigma*n 是对应力空间的本质边界限制。
    """
    # 1. 插值边界上的应力值
    # 注意：set_dirichlet_bc 内部会调用 boundary_interpolate
    # 我们需要确保 mesh.edgedata['dirichlet'] 已经被正确标记为 Traction 边界
    uh, isbddof = space.set_dirichlet_bc(g_stress, threshold=threshold)

    # 2. 修改线性系统 (A x = F)
    # 将已知自由度移至右端项: F = F - A * known_u
    F = F - A @ uh
    
    # 3. 强制对角线为 1，非对角线为 0，右端项为已知值
    F[isbddof] = uh[isbddof]
    
    gdof = space.number_of_global_dofs()
    bdIdx = bm.zeros(gdof, dtype=bm.int32)
    bdIdx[isbddof] = 1
    
    Tbd = spdiags(bdIdx, 0, gdof, gdof) # 边界DOF对角矩阵
    T = spdiags(1 - bdIdx, 0, gdof, gdof) # 内部DOF对角矩阵
    
    # A_new = T * A * T + I_bd
    A = T @ A @ T + Tbd
    
    return A, F

def solve_mixed_bc(pde, N, p):
    # 1. 网格生成与角点标记
    # node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=bm.float64)
    # cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]], dtype=bm.int32)
    # mesh = TriangleMesh(node, cell)
    # mesh.uniform_refine(N)
    # mesh.meshdata['corner'] = node[:-1]

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
    mesh.meshdata['corner'] = corner_coords

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # mesh.add_plot(axes)
    # mesh.find_node(axes, showindex=True)
    # mesh.find_edge(axes, showindex=True)
    # mesh.find_cell(axes, showindex=True)
    # plt.show()

    # ==========================================
    # 2. 标记混合边界
    # ==========================================
    bc = mesh.entity_barycenter('edge')
    is_right_bd = (bc[:, 0] > 1.0 - 1e-9) # x=1 边界 (应力/面力边界)
    is_boundary = mesh.boundary_edge_flag()
    is_disp_bd = is_boundary & (~is_right_bd) # 其他边界 (位移边界)

    # 为了适配现有的函数接口，我们需要设置 edgedata
    # 'neumann' 在 displacement_boundary_condition 中被读取，用于计算自然边界积分(位移BC)
    mesh.edgedata['neumann'] = is_disp_bd
    
    # 'dirichlet' 在 space.set_dirichlet_bc 中被读取，用于插值本质边界值(应力BC)
    mesh.edgedata['dirichlet'] = is_right_bd

    # 3. 空间定义
    # from huzhang_fe_space_with_corner_relaxation import HuZhangFESpace2d
    # space0 = HuZhangFESpace2d(mesh, p=p) # 应力空间
    from soptx.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d 
    space0 = HuZhangFESpace2d(mesh, p=p, use_relaxation=False) # 应力空间
    space = LagrangeFESpace(mesh, p=p-1, ctype='D') # 位移空间
    space1 = TensorFunctionSpace(space, shape=(-1, 2))

    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    # 4. 组装矩阵 (M: Mass, B: Div)
    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(lambda0=pde.lambda0, lambda1=pde.lambda1))
    M = bform1.assembly()

    bform2 = BilinearForm((space1, space0))
    bform2.add_integrator(HuZhangMixIntegrator())
    B = bform2.assembly()

    M = M.to_scipy()
    B = B.to_scipy()

    # 5. 角点基函数变换 (Corner Relaxation)
    TM = space0.TM.to_scipy()
    # 注意：如果 TM 非单位阵，本质边界条件的处理需要非常小心。
    # TM 将原始自由度(系数)变换为满足稳定性的组合自由度。
    # 这里我们先应用 TM 变换系统，然后再强加边界条件。
    # 假设边界插值得到的系数 uh 是针对变换后的基函数的（或者 TM 在边界处不影响本质值的强加）。
    # 对于 x=1 边界，不涉及角点(0,0), (0,1) 等奇异点，通常影响较小。
    M = TM.T @ M @ TM
    B = TM.T @ B
    
    # 构建分块矩阵 [M  B]
    #              [B^T 0]
    A = bmat([[M, B], [B.T, None]], format='csr')

    # 6. 组装右端项
    # Part A: 源项 (div sigma = -f) -> 对应位移测试函数 v
    lform1 = LinearForm(space1)
    lform1.add_integrator(VectorSourceIntegrator(source=pde.source))
    b = lform1.assembly() # 形状 (gdof1, )

    # Part B: 位移边界条件 (Natural BC for Stress) -> 对应应力测试函数 tau
    # 计算 <g_u, tau*n>_Gamma_u
    # 内部会自动读取 mesh.edgedata['neumann'] (我们已将其设为位移边界)
    a = displacement_boundary_condition(space0, pde.displacement) 
    
    # 应用 TM 变换到右端项
    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = TM.T @ a
    F[gdof0:] = -b

    # 7. 应用本质应力边界条件 (Essential BC for Stress)
    # 针对右边界 (x=1)，我们已知 sigma*n
    # 这里我们对总系统 A 和 F 进行修改，只影响左上角 block (应力-应力部分)
    # 注意：我们要传入总矩阵 A，因为修改行/列会破坏分块结构
    
    # 获取应力空间的本质边界值
    # 注意：我们需要在此处仅传入应力部分的 F[:gdof0] 进行修改，还是整体修改？
    # 通常 dirichlet_bc 函数是对单空间矩阵操作。对于混合系统，最好手动处理左上角，
    # 或者编写专门针对 Block 的函数。
    # 简单起见，我们先提取 M 部分的修改逻辑，但因为 A 已经是 Block 了，我们需要对 A 的前 gdof0 行/列操作。
    
    # ---------------------------------------------------------
    # 修正策略：直接在全局 Block 矩阵上应用。
    # set_dirichlet_bc 返回的 isbddof 长度是 gdof0。
    # 我们需要将其映射到全局 A 的索引上。
    
    uh_stress, isbddof_stress = space0.set_dirichlet_bc(pde.stress)
    
    # 扩展全系统向量
    uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
    uh_global[:gdof0] = uh_stress
    
    isbddof_global = bm.zeros(A.shape[0], dtype=bool)
    isbddof_global[:gdof0] = isbddof_stress
    
    # 修改右端项: F = F - A * u_known
    F = F - A @ uh_global
    
    # 强加边界值
    F[isbddof_global] = uh_global[isbddof_global]
    
    # 修改矩阵 A (置 1 置 0)
    # 构造对角掩码矩阵
    total_dof = A.shape[0]
    bdIdx = bm.zeros(total_dof, dtype=bm.int32)
    bdIdx[isbddof_global] = 1
    
    Tbd = spdiags(bdIdx, 0, total_dof, total_dof)
    T = spdiags(1 - bdIdx, 0, total_dof, total_dof)
    
    A = T @ A @ T + Tbd
    # ---------------------------------------------------------

    # 8. 求解
    X = scipy_spsolve(A, F)

    sigmaval = X[:gdof0]
    uval = X[gdof0:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    uh = space1.function()
    uh[:] = uval

    return sigmah, uh

if __name__ == "__main__":
    from sympy import symbols, sin, cos, pi
    
    # 设置 PDE 参数
    lambda0 = 4
    lambda1 = 1
    p = 3
    maxit = 4
    
    # 构造精确解 (可以使用 linear_elastic_with_huzhang.py 中的算例)
    x, y = symbols('x y')
    u0 = sin(pi*x)*sin(pi*y) + x 
    u1 = cos(pi*x)*cos(pi*y) + y
    u = [u0, u1]
    
    pde = LinearElasticPDE(u, lambda0, lambda1)

    errorType = ['$|| \\sigma - \\sigma_h||_{0}$', '$|| u - u_h||_{0}$']
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    print(f"Testing Mixed Boundary Conditions (p={p}):")
    print("Left/Top/Bottom: Displacement BC (Natural)")
    print("Right (x=1): Traction/Stress BC (Essential)")
    print("-" * 50)

    for i in range(maxit):
        N = i + 1
        sigmah, uh = solve_mixed_bc(pde, N, p)
        mesh = sigmah.space.mesh

        e0 = mesh.error(uh, pde.displacement)
        e1 = mesh.error(sigmah, pde.stress)

        h[i] = mesh.entity_measure('edge').max()
        errorMatrix[0, i] = e1
        errorMatrix[1, i] = e0
        print(f"Refine {N}: ||u-uh||={e0:.4e}, ||sig-sigh||={e1:.4e}")

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix, errorType, propsize=20)
    plt.show()