# -*- coding: utf-8 -*-
"""EA 单元装配走查: 构建单元矩阵, 走一遍 y = G^T K_e G x, 再演示更新."""

import argparse

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import ElementRestriction
from soptx.fem.levels import ElementAssembly
from soptx.materials import IsotropicLinearElasticMaterial

bm.set_backend('numpy')

MESHES = {'tri': TriangleMesh, 'quad': QuadrangleMesh}

parser = argparse.ArgumentParser(description='EA 单元装配走查')
parser.add_argument('-m', default='tri', choices=list(MESHES), help='网格类型')
parser.add_argument('-n', type=int, default=2, help='每方向网格剖分数')
parser.add_argument('-p', type=int, default=1, help='拉格朗日元次数')
args = parser.parse_args()
n, p = args.n, args.p

mesh = MESHES[args.m].from_box([0, 1, 0, 1], nx=n, ny=n)
GD = mesh.geo_dimension()
space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))
material = IsotropicLinearElasticMaterial(hypothesis='plane_strain',
                                        lame_lambda=1.0, shear_modulus=0.75,
                                        device=bm.get_device(mesh))
coef = bm.ones(mesh.number_of_cells(), dtype=bm.float64)   
integrator = LinearElasticIntegrator(material=material, coef=coef)

# setup 阶段: 依赖网格几何与拓扑, 每张网格算一次
# 单元矩阵 K_e: EA 没有独立的 build 段, 参考基在 assembly 内部现算, 与几何、材料、密度一起积进 K_e;
# K_e 的初值含 coef, 按阶段属于第一次 update
K_e = integrator.assembly(space)  # (NC, ldof * GD, ldof * GD)

# 单元限制 G: 扁平布局 (NC, ldof * GD), 只依赖网格拓扑 (cell2dof), 单元内自由度顺序与 K_e 的行列一致
g = ElementRestriction.from_integrator(integrator, space, layout='flat')

# G 与 K_e 拼成 EA 算子
ea = ElementAssembly(space, restriction=g, element_matrices=K_e)

# 逐单元存储: 随网格规模线性增长
print('cell2dof      ', g.cell2dof.shape)            # 单元限制 G
print('K_e           ', ea.element_matrices.shape)   # 单元矩阵 K_e

# 全网格共享: 无, 参考基与本构矩阵都已积进 K_e, 不单独保留

# apply 阶段: 每次 MatVec
x = bm.arange(space.number_of_global_dofs(), dtype=bm.float64)

# G: 全局 -> 单元
x_E = g.gather(x)  # (NC, ldof * GD[, NB])

# K_e: 逐单元小矩阵乘向量
y_E = bm.einsum('cij, cj... -> ci...', K_e, x_E)  # (NC, ldof * GD[, NB])

# G^T: 单元 -> 全局
y = g.scatter_add(y_E)

print('y == ea @ x   ', bool(bm.all(y == ea @ x)))

# update 阶段: 单元密度 (NC, ) 下 K_e 对 rho_e 线性, EA 由 K_e^0 逐单元缩放即可, 不必重新积分
# K_e^0: coef 全为 1 时的单元矩阵; 这里初始 coef 全为 1, K_e 即 K_e^0, 但 update 会原地写 K_e,
# 故另存一份作基准 (分析器里这一份就是敏度用的实体单元矩阵缓存)
K_e0 = bm.copy(K_e)
ea = ElementAssembly(space, restriction=g, element_matrices=K_e,
                    reference_matrices=K_e0, integrator=integrator)
new_coef = 0.5 * coef
ea.update(new_coef)  # 原地: K_e <- new_coef_e * K_e^0

integrator.coef = new_coef
print('单元密度: 缩放 ~ 重新积分',
      bool(bm.allclose(ea.element_matrices, integrator.assembly(space))))

# update 之后算子作用随之改变: K_e 整体减半, y 也减半
print('单元密度: ea @ x ~ 0.5 y  ', bool(bm.allclose(ea @ x, 0.5 * y)))
