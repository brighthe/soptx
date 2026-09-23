# -*- coding: utf-8 -*-
"""PA 部分装配走查: 构建四个内核, 再走一遍 y = G^T B^T D B G x."""

import argparse

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import (
    ElementRestriction,
    GeometricFactors,
    LinearElasticQFunction,
    ReferenceBasis,
    physical_gradient,
    physical_gradient_transpose,
    weighted_stress,
)
from soptx.fem.levels import PartialAssembly, quadrature_geometry
from soptx.materials import IsotropicLinearElasticMaterial

bm.set_backend('numpy')

MESHES = {'tri': TriangleMesh, 'quad': QuadrangleMesh}

parser = argparse.ArgumentParser(description='PA 部分装配走查')
parser.add_argument('-m', default='tri', choices=list(MESHES), help='网格类型')
parser.add_argument('-n', type=int, default=2, help='每方向网格剖分数')
parser.add_argument('-p', type=int, default=1, help='拉格朗日元次数')
args = parser.parse_args()
n, p = args.n, args.p

GD = 2
mesh = MESHES[args.m].from_box([0, 1, 0, 1], nx=n, ny=n)
space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))
material = IsotropicLinearElasticMaterial(hypothesis='plane_strain',
                                        lame_lambda=1.0, shear_modulus=0.75,
                                        device=bm.get_device(mesh))
coef = bm.ones(mesh.number_of_cells(), dtype=bm.float64)  
integrator = LinearElasticIntegrator(material=material, coef=coef)

# build 阶段: 只依赖 (单元形状, p, q), 与网格无关, 一次算定
ref_basis = ReferenceBasis.build(space.scalar_space,
                                integrator.quadrature_order(space))

# 只依赖网格拓扑 (cell2dof) 与自由度排序, 与节点坐标和设计变量都无关, 不必经由 pa 取;
# 自由度排序在这里一次换成 (NC, ldof, GD) 的分量布局
g = ElementRestriction.from_integrator(integrator, space, layout='component')

# setup: 依赖网格坐标与设计变量, 每张网格算一次; 与 PartialAssembly.build 调同一个函数
ctx = integrator.fetch_context(space)
jacobi_inverse, weighted_measure = quadrature_geometry(ctx)

# 几何因子: J^{-1}
geo = GeometricFactors(jacobi_inverse=jacobi_inverse)

# 逐点算子 D: 本构矩阵 + 积分权重 w_q |J| + 相对密度 rho
qf = LinearElasticQFunction(elastic_matrix=material.elastic_matrix()[0, 0],
                            weighted_measure=weighted_measure,
                            coef=coef)

# 四个内核拼成 PA 算子; PartialAssembly.build 内部做的正是上面这几步
pa = PartialAssembly(space, restriction=g, reference_basis=ref_basis,
                    geometric_factors=geo, qfunction=qf)

# 逐单元存储: 随网格规模线性增长
print('cell2dof      ', g.cell2dof.shape)          # 单元限制 G
print('jacobi_inverse', geo.jacobi_inverse.shape)  # Jacobi 逆 J^-1
print('weighted_coef ', qf.weighted_coef.shape)    # 逐点标量 w|J|rho

# 全网格共享: 与网格规模无关
print('ref grad      ', ref_basis.grad.shape)      # 参考基函数梯度 B_hat_phi
print('elastic_matrix', qf.elastic_matrix.shape)   # 本构矩阵 D
print('strain_map    ', qf.strain_map.shape)       # Voigt 对称化 S

# apply: 每次 MatVec
x = bm.arange(space.number_of_global_dofs(), dtype=bm.float64)

# G: 全局 -> 单元, (NC, ldof, GD[, NB])
x_E = g.gather(x)

# B: 单元 -> 积分点, (NC, NQ, GD, GD[, NB])
grad_u = physical_gradient(
                x_E,
                reference_grad=ref_basis.grad,
                jacobi_inverse=geo.jacobi_inverse
            )

# D: 逐点块对角矩阵乘向量, 形状不变
s_Q = weighted_stress(
                grad_u,
                weighted_coef=qf.weighted_coef,
                elastic_matrix=qf.elastic_matrix,
                strain_map=qf.strain_map
            )

# B^T: 积分点 -> 单元, (NC, ldof, GD[, NB])
y_E = physical_gradient_transpose(
                s_Q,
                reference_grad=ref_basis.grad,
                jacobi_inverse=geo.jacobi_inverse
            )

# G^T: 单元 -> 全局
y = g.scatter_add(y_E)

print('y == pa @ x   ', bool(bm.all(y == pa @ x)))

# update: 每次设计更新; 只重算 weighted_coef, 其余内核一个都不动
j_before, gr_before = geo.jacobi_inverse, ref_basis.grad
wc_before = qf.weighted_coef
pa.update(0.5 * coef)
print('weighted_coef 重算', bool(bm.allclose(qf.weighted_coef, 0.5 * wc_before)))
print('jacobi 未动       ', geo.jacobi_inverse is j_before)
print('参考基未动         ', ref_basis.grad is gr_before)
