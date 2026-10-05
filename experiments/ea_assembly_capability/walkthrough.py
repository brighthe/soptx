"""EA 单元装配: 在同一网格上依次走标准 EA 与参考 EA 的 setup / update / apply / diagonal, 再施加
Dirichlet 条件用 Jacobi-PCG 求解, 以 FA 直接解为基准.

只考虑单元密度 s_e (NC, ), 此时 K_e = s_e K_e^0.

1. 标准 EA (ElementAssembly): 常驻 K_e, update 带新系数重新积分, apply 走 y = sum_e G_e^T K_e G_e x;
2. 参考 EA (SharedReferenceElementAssembly): 同一平移类的单元共用一份参考单元矩阵, 常驻 N_k 份 K_k^0
   与 s_e, update 只换 s_e, apply 走 y = sum_e s_e G_e^T K_k(e)^0 G_e x, k(e) = e mod N_k;
   依赖 ``create_box_mesh`` 的单元编号约定, 平移类由其返回的 ``classes`` 给出;
3. 求解: 两种 EA 包成 ConstrainedOperator (A = Pi_I K Pi_I + Pi_D), 用 diagonal 做 Jacobi 预条件跑 PCG;
   FA 装出全局 CSR 后直接解, 作为位移基准, 并对比三者的常驻字节数.
"""

import argparse

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import ElementRestriction
from soptx.fem.levels import ElementAssembly, FullAssembly, SharedReferenceElementAssembly
from soptx.fem.operators import ConstrainedOperator
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import MESH_TYPES, create_box_mesh
from soptx.solvers import DiagonalPreconditioner, cg, spsolve


def parse_args(argv=None):
    """解析 EA 单元装配走查的命令行参数."""
    parser = argparse.ArgumentParser(description='EA 单元装配走查')
    parser.add_argument('--mesh', default='tri', choices=MESH_TYPES, help='网格类型')
    parser.add_argument('-n', '--n', type=int, default=2, help='每方向网格剖分数')
    parser.add_argument('-p', '--p', type=int, default=1, help='拉格朗日元次数')
    args = parser.parse_args(argv)

    if args.n < 1:
        parser.error('-n 须为正整数')
    if args.p < 1:
        parser.error('-p 须为正整数')
    return args


def main(argv=None):
    """在同一网格上依次走标准 EA 与参考 EA 的 setup / update / apply / diagonal, 再求解并与 FA 对照."""
    args = parse_args(argv)
    n, p = args.n, args.p

    bm.set_backend('numpy')

    GD = 2 if args.mesh in ('tri', 'quad') else 3
    # 网格与 from_box 逐位一致, 另给出平移类划分 classes
    mesh, classes = create_box_mesh(args.mesh, [0, 1] * GD, *(n, ) * GD)
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

    # 前一项核对手工 apply 与 @ 一致; 后一项独立核对 update 后的 K_e 确为 s_e K_e^0,
    # atol 按 K_e^0 的量级取, 免得近零元素卡在 rtol 上
    print('标准 EA       y == ea @ x ', bool(bm.all(y == ea @ x)),
          '| K_e ~ s_e K_e^0', bool(bm.allclose(K_e, coef[:, None, None] * K_e0,
                                                rtol=1e-12, atol=1e-12 * float(bm.max(bm.abs(K_e0))))))

    # diagonal: 逐单元取 K_e 的对角再散加, 供 Jacobi 类预条件使用
    d_E = bm.einsum('cii -> ci', K_e)             # (NC, ldof * GD)
    d = g.scatter_add(d_E)
    print('标准 EA       d == ea.diagonal()', bool(bm.all(d == ea.diagonal())))

    # ========================================================================
    # 参考 EA: 常驻 N_k 份 K_k^0 与 s_e, 不另存 K_e
    # ========================================================================
    # setup 阶段: 只积第一个格子的 N_k 个单元, 按编号约定每类恰有一个, 作为该类的代表
    # 分析器在单元密度下也用本类, 但取 N_k = NC 直接引用 K_e^0 (K_k0 = K_e0), 其余流程不变
    N_k = classes.num_classes
    integrator_rEA = LinearElasticIntegrator(material=material, index=classes.representatives(),
                                            method='fast')
    K_k0 = integrator_rEA.assembly(space)  # (N_k, ldof * GD, ldof * GD)

    # G 复用标准 EA 的
    ref = SharedReferenceElementAssembly(space, restriction=g, reference_matrices=K_k0)

    # update 阶段: 只换 s_e, 不积分也不缩放矩阵
    ref.update(coef)

    # apply 阶段: y = sum_e s_e G_e^T K_k(e)^0 G_e x
    x_E = g.gather(x)                                                      # (NC, ldof * GD)
    x_B = bm.reshape(x_E, (NC // N_k, N_k, -1))                            # 单元 e = b * N_k + k (b 为格子), 第 1 维即 k(e)
    y_B = bm.einsum('kij, bkj -> bki', K_k0, x_B)                          # 同类单元共用 K_k^0 批量乘
    y_E = bm.reshape(y_B, (NC, -1)) * bm.reshape(ref.scale, (NC, 1))       # 再乘 s_e
    y_ref = g.scatter_add(y_E)

    # s_e 后乘, 且同类单元坐标只在舍入意义下相等, 与标准 EA 只一致到舍入; 容差收紧到舍入量级,
    # 单元编号与平移类哪怕只错一部分也会暴露
    print(f'参考 EA N_k={N_k}  y == ref @ x', bool(bm.all(y_ref == ref @ x)),
          '| ~ 标准 EA', bool(bm.allclose(y_ref, y, rtol=1e-12, atol=1e-12 * float(bm.max(bm.abs(y))))))

    # diagonal: 每类取 K_k^0 的对角, 乘 s_e 后散加
    d_B = bm.reshape(ref.scale, (NC // N_k, N_k, 1)) * bm.einsum('kii -> ki', K_k0)[None, :, :]
    d_ref = g.scatter_add(bm.reshape(d_B, (NC, -1)))
    print(f'参考 EA N_k={N_k}  d == ref.diagonal()', bool(bm.all(d_ref == ref.diagonal())))

    # ========================================================================
    # 求解: 施加 Dirichlet 条件后用 Jacobi-PCG 解 K u = F, 以 FA 直接解为基准
    # ========================================================================
    # 问题: 左端 x = 0 固支, 全体节点受 -y 向单位节点力; shape=(-1, GD) 下各节点的分量交错编号
    gdof = space.number_of_global_dofs()
    isDDof = space.is_boundary_dof(threshold=lambda pts: bm.abs(pts[..., 0]) < 1e-12, method='interp')
    F = bm.zeros((gdof // GD, GD), dtype=bm.float64)
    F = bm.reshape(bm.set_at(F, (slice(None), 1), -1.0), (-1, ))

    # FA: 首次装配前建 CSR 骨架 (pattern), 再把 K_e 求和成全局 CSR; 每个设计步都要重新积分并
    # 重新求和, 骨架常驻供复用. EA 两者都免, 代价是只能走迭代法
    integrator_FA = LinearElasticIntegrator(material=material, coef=coef, method='fast')
    fa = FullAssembly.build(space, integrator_FA)

    # FA 直接解: 齐次 Dirichlet 下对称消元等价于只解内部块 K_II u_I = F_I
    isIDof = ~isDDof
    K_II = fa.matrix.to_scipy().tocsr()[isIDof][:, isIDof]
    u_FA = bm.set_at(bm.zeros(gdof, dtype=bm.float64), isIDof, spsolve(K_II, F[isIDof]))

    # 对角另与 FA 的 CSR 主对角核对; 二者求和次序不同, 只一致到舍入
    print('标准 EA / 参考 EA diagonal ~ FA',
          bool(bm.allclose(d, fa.diagonal(), rtol=1e-12, atol=1e-12 * float(bm.max(bm.abs(d))))),
          bool(bm.allclose(d_ref, fa.diagonal(), rtol=1e-12, atol=1e-12 * float(bm.max(bm.abs(d))))))

    # EA: 不改写任何矩阵, 把算子包成 A = Pi_I K Pi_I + Pi_D; A 的对角在 Dirichlet 处为 1,
    # 其余转发 level.diagonal(), DiagonalPreconditioner 在 setup 时向 A 要
    for name, level in (('标准 EA', ea), (f'参考 EA N_k={N_k}', ref)):
        A = ConstrainedOperator(level, gd=0.0, isDDof=isDDof)
        u0 = A.init_solution(dtype=bm.float64)    # 边界取给定值, 内部为零
        b = A.apply(F, u0)                        # 边界值的贡献移到右端
        M = DiagonalPreconditioner().setup(A)
        u, info = cg(A, b, x0=u0, M=M, atol=1e-14, rtol=1e-12, returninfo=True, print_level=0)
        err = float(bm.linalg.norm(u - u_FA) / bm.linalg.norm(u_FA))
        print(f'{name:<13} PCG {info["niter"]} 步 | 位移 ~ FA 直接解, 相对误差 {err:.1e}')

    # 常驻字节数: FA 的 values / crow / col 与骨架共享内存, 骨架另有逐单元槽位映射 slot_base 与
    # row_deg; EA 为单元矩阵 (或 K_k^0 与 s_e) 加 cell2dof. 小规模下只示意量级关系, 量化结论见
    # results_analysis.md
    pattern = fa.pattern
    slot_bytes = sum(int(a.nbytes) for a in (pattern.slot_base, pattern.row_deg) if a is not None)
    print(f'常驻字节 FA {fa.persistent_bytes()} + 骨架槽位 {slot_bytes} | 标准 EA {ea.persistent_bytes()}'
          f' | 参考 EA {ref.persistent_bytes()}')

    # 分析器中的实际调用链 (LagrangeFEMAnalyzer):
    # - assemble_stiff_matrix: 单元密度下构造 N_k = NC 的 SharedReferenceElementAssembly (K_k^0 即
    #   _solid_stiffness_matrix 缓存的 K_e^0), 其余系数走 create_level('ea') -> ElementAssembly.build;
    # - apply_bc ('matrix_free' 变体): 包成 ConstrainedOperator 并修正右端, 同本段;
    # - _build_solver: precond='jacobi' 时 DiagonalPreconditioner().setup(A), 再交给 cg


if __name__ == '__main__':
    main()
