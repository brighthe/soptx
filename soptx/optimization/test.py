import numpy as np
from fealpy.mesh import QuadrangleMesh
mesh = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=10, ny=10)
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace


qf = mesh.quadrature_formula(q=3)
bcs, ws = qf.get_quadrature_points_and_weights()

space= LagrangeFESpace(mesh, p=2, ctype='C')
phi = space.basis(bcs)  # (1, NQ, ldof)
phi_matrix = phi[0]  # 形状变为 (9, 9)

# 方法1: 计算矩阵的秩
rank = np.linalg.matrix_rank(phi_matrix)
print(f"矩阵的秩: {rank}")
print(f"矩阵是否满秩: {rank == 9}")
print("---------")