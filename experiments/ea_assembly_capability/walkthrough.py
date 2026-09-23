# -*- coding: utf-8 -*-
"""EA 单元装配走查: 构建单元矩阵, 走一遍 y = G^T K_e G x, 再检验 K_e 并演示更新."""

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

# build + setup 阶段: 一次算出全部单元矩阵 K_e, 几何, 材料与密度都已乘进去
K_e = integrator.assembly(space)  # (NC, ldof * GD, ldof * GD)

# 单元限制 G: 扁平布局 (NC, ldof * GD), 单元内自由度顺序与 K_e 的行列一致
g = ElementRestriction.from_integrator(integrator, space, layout='flat')

# G 与 K_e 拼成 EA 算子
ea = ElementAssembly(space, restriction=g, element_matrices=K_e)

# apply 阶段: 每次 MatVec
x = bm.arange(space.number_of_global_dofs(), dtype=bm.float64)

# G: 全局 -> 单元, (NC, ldof * GD[, NB])
x_E = g.gather(x)

# K_e: 逐单元小矩阵乘向量, 形状不变
y_E = bm.einsum('cij, cj... -> ci...', K_e, x_E)

# G^T: 单元 -> 全局
y = g.scatter_add(y_E)

print('y == ea @ x   ', bool(bm.all(y == ea @ x)))

# K_e 自检: 只看 K_e 本身, 不借助其他层级
# 对称: 双线性形式 a(u, v) 对称
print('K_e 对称       ', bool(bm.allclose(K_e, bm.swapaxes(K_e, 1, 2))))

# 刚体模态: 平移与转动不产生应变, 必在每个 K_e 的零空间里
def translation_x(points):
    x = points[..., 0]
    return bm.stack([bm.ones_like(x), bm.zeros_like(x)], axis=-1)

def translation_y(points):
    x = points[..., 0]
    return bm.stack([bm.zeros_like(x), bm.ones_like(x)], axis=-1)

def rotation(points):
    return bm.stack([-points[..., 1], points[..., 0]], axis=-1)

# 线性场被插值精确表示; 经 interpolate 排成全局自由度, 再由 G 取到单元, 布局与 K_e
# 自动一致: (gdof, 3) -> (NC, ldof * GD, 3)
modes = bm.stack([space.interpolate(mode)[:]
                for mode in (translation_x, translation_y, rotation)], axis=-1)
modes_E = g.gather(modes)

# 与 apply 阶段同一个收缩, 三个模态作为批量维一次算完: (NC, ldof * GD, 3)
residual = bm.einsum('cij, cj... -> ci...', K_e, modes_E)

# 相对 K_e 的量级判零, 只差舍入误差
print('刚体模态在零空间',
    bool(bm.max(bm.abs(residual)) <= 1e-12 * bm.max(bm.abs(K_e))))

# update 阶段: 能否不重新积分, 取决于密度是逐单元还是逐点
# 单元密度 (NC, ): K_e 对 rho_e 线性, EA 由 K_e^0 逐单元缩放即可, 不必重新积分
new_coef = 0.5 * coef
K_e_scaled = new_coef[:, None, None] * K_e

# 对照: 改 coef 后重新积分; assembly 只缓存几何量, coef 每次现读
integrator.coef = new_coef
K_e_integrated = integrator.assembly(space)

ea.update(K_e_scaled)

print('单元密度: 缩放 ~ 重新积分', bool(bm.allclose(K_e_scaled, K_e_integrated)))

# 逐点密度 (NC, NQ): rho 在单元内变化, 提不出 K_e, EA 只能重新积分
# 积分点数与 assembly 所用求积一致
NC = mesh.number_of_cells()
quadrature = mesh.quadrature_formula(integrator.quadrature_order(space))
NQ = quadrature.number_of_quadrature_points()
coef_q = bm.reshape(0.3 + 0.6 * bm.abs(bm.cos(bm.linspace(0.0, 11.0, NC * NQ))),
                    (NC, NQ))

integrator.coef = coef_q
ea.update(integrator.assembly(space))

# 用单元平均密度缩放 K_e^0 只是近似, 与重新积分不等
K_e_mean = bm.mean(coef_q, axis=1)[:, None, None] * K_e
print('逐点密度: 均值缩放 ~ 重新积分 (应为 False)',
    bool(bm.allclose(K_e_mean, ea.element_matrices)))
