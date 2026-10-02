"""EA 单元装配: 在同一网格上依次走标准 EA 与参考 EA 的 setup / update / apply.

只考虑单元密度 s_e (NC, ), 此时 K_e = s_e K_e^0.

1. 标准 EA (ElementAssembly): 常驻 K_e, update 带新系数重新积分, apply 走 y = sum_e G_e^T K_e G_e x;
2. 参考 EA (SharedReferenceElementAssembly): 同一平移类的单元共用一份参考单元矩阵, 常驻 N_k 份 K_k^0
   与 s_e, update 只换 s_e, apply 走 y = sum_e s_e G_e^T K_k(e)^0 G_e x, k(e) = e mod N_k;
   依赖 from_box 的单元编号约定.
"""

import argparse

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import ElementRestriction
from soptx.fem.levels import ElementAssembly, SharedReferenceElementAssembly
from soptx.materials import IsotropicLinearElasticMaterial

MESHES = {'tri': TriangleMesh, 'quad': QuadrangleMesh,
          'tet': TetrahedronMesh, 'hex': HexahedronMesh}

# 平移类数 N_k: 三角形每格 2 个, 四面体每格 6 个, 四边形与六面体每格 1 个
NUM_CLASSES = {'tri': 2, 'quad': 1, 'tet': 6, 'hex': 1}


def parse_args(argv=None):
    """解析 EA 单元装配走查的命令行参数."""
    parser = argparse.ArgumentParser(description='EA 单元装配走查')
    parser.add_argument('--mesh', default='tri', choices=list(MESHES), help='网格类型')
    parser.add_argument('-n', '--n', type=int, default=2, help='每方向网格剖分数')
    parser.add_argument('-p', '--p', type=int, default=1, help='拉格朗日元次数')
    args = parser.parse_args(argv)

    if args.n < 1:
        parser.error('-n 须为正整数')
    if args.p < 1:
        parser.error('-p 须为正整数')
    return args


def main(argv=None):
    """在同一网格上依次走标准 EA 与参考 EA 的 setup / update / apply."""
    args = parse_args(argv)
    n, p = args.n, args.p

    bm.set_backend('numpy')

    GD = 2 if args.mesh in ('tri', 'quad') else 3
    mesh = MESHES[args.mesh].from_box([0, 1] * GD, *(n, ) * GD)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))
    material = IsotropicLinearElasticMaterial(hypothesis='plane_strain' if GD == 2 else '3D',
                                            lame_lambda=1.0, shear_modulus=0.75,
                                            device=bm.get_device(mesh))

    # 共用的输入: 单元密度 s_e 取非均匀值, 均匀缩放核对不出 s_e 的逐单元顺序; x 为 MatVec 的输入向量
    NC = mesh.number_of_cells()
    coef = bm.linspace(0.2, 1.0, NC, dtype=bm.float64)  # (NC, )
    x = bm.arange(space.number_of_global_dofs(), dtype=bm.float64)

    # ========================================================================
    # 标准 EA: 常驻逐单元的 K_e
    # ========================================================================
    # setup 阶段: 每张网格一次
    # 积分子暂不带 coef, assembly 逐单元积分出 K_e^0 (s_e = 1 时的 K_e; 下标 e 逐单元各异, 上标 0 指实体材料)
    # keep_data 缓存积分所需的几何数据, update 重新积分时直接复用
    integrator_sEA = LinearElasticIntegrator(material=material, method='fast').keep_data(True)
    K_e0 = integrator_sEA.assembly(space)  # (NC, ldof * GD, ldof * GD)

    # 单元限制 G: 扁平布局 (NC, ldof * GD), 单元内顺序与 K_e 行列一致
    g = ElementRestriction.from_integrator(integrator_sEA, space, layout='flat')
    ea = ElementAssembly(space, restriction=g, element_matrices=K_e0, integrator=integrator_sEA)

    # update 阶段: 每个设计步一次; 带 coef 重新积分, 常驻 K_e = s_e K_e^0
    ea.update(coef)
    K_e = ea.element_matrices  # (NC, ldof * GD, ldof * GD)

    # apply 阶段: 每次 MatVec, y = sum_e G_e^T K_e G_e x
    x_E = g.gather(x)                             # G: 全局 -> 单元, (NC, ldof * GD)
    y_E = bm.einsum('cij, cj -> ci', K_e, x_E)    # 逐单元小矩阵乘向量
    y = g.scatter_add(y_E)                        # G^T: 单元 -> 全局

    print('标准 EA       y == ea @ x ', bool(bm.all(y == ea @ x)))

    # ========================================================================
    # 参考 EA: 常驻 N_k 份 K_k^0 与 s_e, 不另存 K_e
    # ========================================================================
    # setup 阶段: 只积第一个格子的 N_k 个单元, 按 from_box 编号每类恰有一个, 作为该类的代表
    # 分析器在单元密度下也用本类, 但取 N_k = NC 直接引用 K_e^0 (K_k0 = K_e0), 其余流程不变
    N_k = NUM_CLASSES[args.mesh]
    integrator_rEA = LinearElasticIntegrator(material=material, index=bm.arange(N_k), method='fast')
    K_k0 = integrator_rEA.assembly(space)  # (N_k, ldof * GD, ldof * GD)

    # G 复用标准 EA 的
    ref = SharedReferenceElementAssembly(space, restriction=g, reference_matrices=K_k0)

    # update 阶段: 只换 s_e, 不积分也不缩放矩阵
    ref.update(coef)

    # apply 阶段: y = sum_e s_e G_e^T K_k(e)^0 G_e x
    x_E = g.gather(x)                                                      # (NC, ldof * GD)
    x_G = bm.reshape(x_E, (NC // N_k, N_k, -1))                            # 单元 e = g * N_k + k, 第 1 维即 k(e)
    y_G = bm.einsum('kij, gkj -> gki', K_k0, x_G)                          # 同类单元共用 K_k^0 批量乘
    y_E = bm.reshape(y_G, (NC, -1)) * bm.reshape(ref.scale, (NC, 1))       # 再乘 s_e
    y_ref = g.scatter_add(y_E)

    # s_e 后乘, 且同类单元坐标只在舍入意义下相等, 与标准 EA 只一致到舍入
    print(f'参考 EA N_k={N_k}  y == ref @ x', bool(bm.all(y_ref == ref @ x)),
          '| ~ 标准 EA', bool(bm.allclose(y_ref, y)))


if __name__ == '__main__':
    main()
