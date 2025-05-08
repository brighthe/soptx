import numpy as np
from scipy.interpolate import griddata

from fealpy.backend import backend_manager as bm
from soptx.solver import ElasticFEMSolver, AssemblyMethod
from soptx.material import DensityBasedMaterialConfig, DensityBasedMaterialInstance
from soptx.material import LevelSetMaterialConfig, LevelSetAreaRationMaterialInstance
from fealpy.mesh import UniformMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.pde import Cantilever2dData1

def calc_curvature(phi, dx, dy):
    """计算水平集函数的曲率"""
    # 获取有限差分计算所需的矩阵
    matrix = matrix4diff(phi)
    
    # 计算x和y方向上的一阶导数(中心差分)
    phix = (matrix['i_plus_1'] - matrix['i_minus_1']) / (2 * dx)
    phiy = (matrix['j_plus_1'] - matrix['j_minus_1']) / (2 * dy)
    
    # 计算交叉导数
    phiyx_bk, phiyx_fw = upwind_diff(phiy, dx, 'x')
    phixy = (phiyx_bk + phiyx_fw) / 2
    
    # 计算二阶导数
    phixx = (matrix['i_plus_1'] - 2 * phi + matrix['i_minus_1']) / dx**2
    phiyy = (matrix['j_plus_1'] - 2 * phi + matrix['j_minus_1']) / dy**2
    
    # 计算曲率
    denominator = (phix**2 + phiy**2)**1.5 + 100 * np.finfo(float).eps
    curvature = (phixx * phiy**2 - 2 * phix * phiy * phixy + phiyy * phix**2) / denominator
    
    # 向量化结果
    return curvature.flatten()

def matrix4diff(phi):
    """生成用于有限差分计算的矩阵"""
    # 初始化四个矩阵
    i_minus_1 = np.zeros_like(phi)
    i_plus_1 = np.zeros_like(phi)
    j_minus_1 = np.zeros_like(phi)
    j_plus_1 = np.zeros_like(phi)
    
    # 设置i_minus_1 (x方向左移一个单位)
    i_minus_1[:, 0] = phi[:, -1]  # 周期性边界条件
    i_minus_1[:, 1:] = phi[:, :-1]
    
    # 设置i_plus_1 (x方向右移一个单位)
    i_plus_1[:, -1] = phi[:, 0]  # 周期性边界条件
    i_plus_1[:, :-1] = phi[:, 1:]
    
    # 设置j_minus_1 (y方向下移一个单位)
    j_minus_1[0, :] = phi[-1, :]  # 周期性边界条件
    j_minus_1[1:, :] = phi[:-1, :]
    
    # 设置j_plus_1 (y方向上移一个单位)
    j_plus_1[-1, :] = phi[0, :]  # 周期性边界条件
    j_plus_1[:-1, :] = phi[1:, :]
    
    return {
        'i_minus_1': i_minus_1,
        'i_plus_1': i_plus_1,
        'j_minus_1': j_minus_1,
        'j_plus_1': j_plus_1
    }


def upwind_diff(phi, dx, str_direction):
    """计算迎风向有限差分"""
    matrix = matrix4diff(phi)
    
    if str_direction == 'x':
        # x方向上的差分
        back_diff = (phi - matrix['i_minus_1']) / dx
        fawd_diff = (matrix['i_plus_1'] - phi) / dx
    elif str_direction == 'y':
        # y方向上的差分
        back_diff = (phi - matrix['j_minus_1']) / dx
        fawd_diff = (matrix['j_plus_1'] - phi) / dx
    
    return back_diff, fawd_diff


p = 1
nx, ny = 32, 22
h = [1, 1]
domain_width = nx * h[0]  
domain_height = ny * h[1]  
mesh_fe = UniformMesh(
                extent=[0, nx, 0, ny], h=[1, 1], origin=[0.0, 0.0], 
                device='cpu'
            )
mesh_ls = UniformMesh(
                extent=[0, nx+1, 0, ny+1], h=[1, 1], origin=[-0.5, -0.5], 
                device='cpu'
            )

NN_ls = mesh_ls.number_of_nodes()
node_ls = mesh_ls.entity('node')
node_ls_x = node_ls[:, 0]
node_ls_y = node_ls[:, 1]
node_fe = mesh_fe.entity('node')

# 定义初始结构的 "骨架" 点
cx = domain_width / 200 * bm.array([33.33, 100, 166.67, 0, 66.67, 133.33, 200, 33.33, 100, 166.67, 0, 66.67, 133.33, 200, 33.33, 100, 166.67])
cy = domain_height / 100 * bm.array([0, 0, 0, 25, 25, 25, 25, 50, 50, 50, 75, 75, 75, 75, 100, 100, 100])

# 计算初始水平集函数
phi_tmp = np.zeros((NN_ls, len(cx)))
for i in range(len(cx)):
    phi_tmp[:, i] = -np.sqrt((node_ls_x - cx[i])**2 + (node_ls_y - cy[i])**2) + domain_height / 10
phi_ls = -(bm.max(phi_tmp, axis=1))

# 设置边界条件
boundary_idx = bm.nonzero(mesh_ls.boundary_node_flag())[0]
phi_ls[boundary_idx] = -1e-6

node_fe_x = node_fe[:, 0]
node_fe_y = node_fe[:, 1]

# 将水平集函数投影到有限元节点
from scipy.interpolate import griddata
points = np.column_stack((node_ls_x, node_ls_y))
phi_fe = griddata(points, phi_ls, np.column_stack((node_fe_x, node_fe_y)), method='cubic')

mesh_fe.nodedata['phi_fe'] = phi_fe
mesh_fe.to_vtk('/home/heliang/FEALPy_Development/soptx/soptx/vtu/fe_phi.vts')

mesh_ls.nodedata['phi_ls'] = phi_ls
mesh_ls.to_vtk('/home/heliang/FEALPy_Development/soptx/soptx/vtu/ls_phi.vts')

material_config = LevelSetMaterialConfig(
                    elastic_modulus=1,            
                    minimal_modulus=1e-9,         
                    poisson_ratio=0.3,            
                    plane_assumption="plane_stress",
                    device=mesh_fe.device,    
                )
materials = LevelSetAreaRationMaterialInstance(config=material_config)
pde = Cantilever2dData1(
            xmin=0, xmax=nx * h[0],
            ymin=0, ymax=ny * h[1],
            T = -1
        )
GD = mesh_fe.geo_dimension()
space_C = LagrangeFESpace(mesh=mesh_fe, p=p, ctype='C')
space_D = LagrangeFESpace(mesh=mesh_fe, p=p-1, ctype='D')
tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
solver = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=AssemblyMethod.STANDARD,
                solver_type='direct',
                solver_params={'solver_type': 'mumps'}, 
            )
KE = solver.get_base_local_stiffness_matrix()

cell_fe = mesh_fe.entity('cell')
phi_fe_elem = phi_fe[cell_fe]
# E1 = materials.calculate_elastic_modulus(levelset=phi_fe_elem)
# KE1 = solver.compute_local_stiffness_matrix(density=phi_fe_elem)

# solver.update_status(phi_fe_elem)
# K = solver._assemble_global_stiffness_matrix()
# F = solver._assemble_global_force_vector()

# uh = solver.solve().displacement
max_iter = 100
for iter_num in range(max_iter):
    print(f"\n迭代 {iter_num+1}/{max_iter}")
    
    # 1. 有限元分析
    solver.update_status(phi_fe_elem)
    K = solver._assemble_global_stiffness_matrix()
    F = solver._assemble_global_force_vector()
    uh = solver.solve().displacement

    mean_compliance = bm.dot(F, uh[:])

    volume_ratio = bm.sum(phi_fe > 0) / len(phi_fe)

    phi_ls_2d = phi_ls.reshape(ny+2, nx+2)
    curvature = calc_curvature(phi_ls_2d, h[0], h[1])
    print("------------------------")

