"""精确子结构流程: 局部装配, Schur 补缩聚, 接口空间投影, 接口装配求解与内部位移恢复."""

import argparse
from math import isfinite


def parse_args(argv=None):
    """解析精确子结构分析的命令行参数."""
    parser = argparse.ArgumentParser(description="精确子结构")
    parser.add_argument(
        "--domain", type=float, nargs="+",
        default=[0.0, 78.0, 0.0, 13.0, 0.0, 13.0],
        help="求解域区间端点 x_min x_max y_min y_max [z_min z_max]",
    )
    parser.add_argument(
        "--n-sub", type=int, nargs="+", default=[78, 13, 13],
        help="各方向子结构数, 个数须等于空间维数",
    )
    parser.add_argument(
        "--n-fine", type=int, default=5, help="子结构各方向细单元数",
    )
    parser.add_argument(
        "--trace-kind", choices=["linear_corner", "full_trace"],
        default="linear_corner", help="接口空间; 供后续缩聚与求解使用",
    )
    parser.add_argument(
        "--solver", choices=["scipy", "mumps"], default="scipy",
        help="接口系统与同网格 FA 对照共用的直接求解器",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="整体密度场随机种子",
    )
    parser.add_argument(
        "--mem-limit-gb", type=float, default=35.0,
        help="进程虚拟地址空间上限 (GiB); 默认 35 GiB",
    )
    args = parser.parse_args(argv)

    domain = args.domain
    if len(domain) not in (4, 6) or not all(isfinite(v) for v in domain):
        parser.error("--domain 须为 4 或 6 个有限数")
    if any(domain[2 * d] >= domain[2 * d + 1] for d in range(len(domain) // 2)):
        parser.error("--domain 各方向须满足下端小于上端")
    if any(domain[2 * d] != 0.0 for d in range(len(domain) // 2)):
        parser.error(
            "--domain 下端须全为 0: 同网格 FA 自检直接以原点为下角的 full_mesh 与 pde 配对"
        )
    if len(args.n_sub) != len(domain) // 2 or min(args.n_sub) <= 0:
        parser.error("--n-sub 须为正整数, 个数等于空间维数")
    if args.n_fine < 2:
        parser.error("--n-fine 至少为 2")
    if args.seed < 0:
        parser.error("--seed 不能为负数")
    if not isfinite(args.mem_limit_gb) or args.mem_limit_gb < 1 / 2**30:
        parser.error("--mem-limit-gb 须为有限正数且至少为 1 字节")
    return args


def main(argv=None):
    """依次创建子结构、生成材料场并装配局部刚度."""
    args = parse_args(argv)

    import resource

    GIB = 2**30
    limit = int(args.mem_limit_gb * GIB)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    import numpy as np
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure import (
        ExactSchurCondensation,
        GlobalAssembler,
        StructuredSubstructureLayout,
        build_substructures,
    )
    from soptx.problems.elasticity import CantileverCorner2d, FullMBBBeam3d

    bm.set_backend("numpy")

    # ------------------------------------------------------------------
    # 算例配置: 物理问题, 离散与接口空间, 数值求解
    # ------------------------------------------------------------------
    # 物理问题: 求解域, 材料 (含 SIMP 插值), 载荷与支承.
    domain = tuple(args.domain)
    dim = len(domain) // 2

    # 材料: 实体为各向同性线弹性, 单元模量按 SIMP 由密度插值 (由参考子结构执行).
    E_base = 1.0
    nu = 0.3
    hypothesis = "3D" if dim == 3 else "plane_stress"
    penalty = 3.0

    # 载荷与支承.
    if dim == 3:
        pde = FullMBBBeam3d(
            domain=domain, P=-1.0, E=E_base, nu=nu, plane_type=hypothesis,
        )
    else:
        pde = CantileverCorner2d(
            domain=domain, P=-1.0, E=E_base, nu=nu, plane_type=hypothesis,
        )

    # 离散与接口空间: 各方向子结构数 n_sub, 子结构内各方向细单元数 n_fine, 接口空间 trace_kind.
    n_sub = tuple(args.n_sub)
    n_fine = (args.n_fine,) * dim
    trace_kind = args.trace_kind

    # 数值求解: 接口系统与同网格 FA 对照共用的直接求解器.
    solver = args.solver

    # ------------------------------------------------------------------
    # 子结构划分: 整体布局, 参考子结构, 子结构排列
    # ------------------------------------------------------------------
    # 求解域划分为 prod(n_sub) 个互不重叠的同构子结构.
    domain_size = domain[1::2]
    layout = StructuredSubstructureLayout(
        domain_size, n_sub, n_fine, E_base=E_base, nu=nu,
        hypothesis=hypothesis,
    )
    assembler = GlobalAssembler(layout)
    # 结构化划分下全部子结构同构, 与 j 无关的几何量 (边界/内部自由度划分,
    # 单位密度单元刚度, 角点插值矩阵) 由参考子结构只构造一次.
    prototype, sub_meshes, positions = build_substructures(
        layout, penal=penalty, rho_min=0.0,
    )

    # ------------------------------------------------------------------
    # 局部刚度装配: 密度场, 局部刚度 K^j (原型内 SIMP 插值)
    # ------------------------------------------------------------------
    M = len(sub_meshes)
    n_dof = prototype.n_total_dofs
    print(
        f"子结构 {M:,} 个, 细单元 {M * prototype.n_cells:,} 个"
    )
    print(
        f"[局部装配] 稠密刚度数组预计 "
        f"{M * n_dof**2 * 8 / GIB:.2f} GiB",
        flush=True,
    )

    # 密度场: 各细单元独立服从 U(density_range), 由 seed 复现.
    density_range = (0.5, 0.9)
    seed = args.seed
    density = bm.asarray(
        np.random.default_rng(seed).uniform(
            *density_range, size=layout.total_fine,
        )
    )
    print(
        f"[输入] 求解域 {domain}, 随机密度 U{density_range}, seed {seed}"
    )

    # 形状: density 为 total_fine; local_density 为 (M, *n_fine), M 为子结构总数,
    # 第 0 维按 x 优先字典序与 sub_meshes 同序; local_stiffness 为 (M, n_b + n_i, n_b + n_i).
    local_density = layout.split_global_cell_field(density)
    local_stiffness = prototype.assemble_local_stiffness_batch(local_density)
    print(
        f"[局部装配] local_density {tuple(local_density.shape)}, "
        f"local_stiffness {tuple(local_stiffness.shape)}"
    )
    # ---------------------------------------------------------------------------
    # 基于静力缩聚构造局部缩聚刚度
    # ---------------------------------------------------------------------------
    condensor = ExactSchurCondensation(prototype.i_dofs, prototype.b_dofs)
    condensed, recovery = condensor.condense(local_stiffness)
    # 走查额外释放: K^j 在缩聚后不再使用, 文档代码未写这一步.
    del local_stiffness
    print(f'[局部缩聚] K_s {tuple(condensed.shape)}, N_int {tuple(recovery.shape)}')

    # # ---------------------------------------------------------------------------
    # # §3.3 接口装配
    # # ---------------------------------------------------------------------------
    # system = assembler.assemble_trace_system(
    #     sub_meshes, condensor, trace_basis=trace,
    # )
    # # 走查额外检查: P = I 要求 full_trace 接口系统沿用 build_interface_dofs 的编号.
    # # linear_corner 下 P 的列数由 solve_constrained_system 校验.
    # if trace_kind == 'full_trace':
    #     assert np.array_equal(bm.to_numpy(system.global_dofs), bm.to_numpy(interface_dofs))
    # print(f'[§3.3] 接口系统 {tuple(system.stiffness.shape)}')

    # # ---------------------------------------------------------------------------
    # # §3.4 边界处理与求解
    # # ---------------------------------------------------------------------------
    # # 静力缩聚不缩聚载荷, 载荷与支承必须落在接口自由度上, 否则抛出 ValueError.
    # conditions = project_problem_conditions_to_interface_system(
    #     pde, assembler, interface_view,
    # )
    # macro_force = projection.T @ conditions.interface_force
    # constraints = projection[conditions.interface_fixed_dofs]
    # solved = solve_constrained_system(
    #     system, macro_force, constraints, solver=args.solver,
    # )
    # macro_u = solved.displacement
    # print(f'[§3.4] {solved.mode}, 约束秩 {solved.constraint_rank}, '
    #       f'平衡相对残差 {solved.equilibrium_relative_residual:.3e}, '
    #       f'约束相对残差 {solved.constraint_relative_residual:.3e}')

    # # ---------------------------------------------------------------------------
    # # §3.5 完整位移恢复
    # # ---------------------------------------------------------------------------
    # interface_u = bm.asarray(
    #     projection @ bm.to_numpy(macro_u), dtype=bm.float64,
    # )
    # displacement = assembler.recover_full_displacement(
    #     sub_meshes, condensor, interface_view, interface_u,
    # )
    # compliance = float(bm.dot(conditions.full_force, displacement))
    # print(f'[§3.5] displacement {tuple(displacement.shape)}, 柔顺度 {compliance:.10e}')

    # # ---------------------------------------------------------------------------
    # # 自检: 同网格 FA. full_trace 应达机器精度; linear_corner 的差来自接口迹降阶.
    # # ---------------------------------------------------------------------------
    # fa = make_fa_analyzer(
    #     layout.full_mesh, pde, layout.material, solve_method=args.solver,
    # )
    # fa_u = fa.solve_state(rho_val=bm.reshape(density, (-1,)))
    # fa_u = np.asarray(bm.to_numpy(fa_u['displacement'][:]), dtype=np.float64)
    # u = np.asarray(bm.to_numpy(displacement), dtype=np.float64)
    # fa_compliance = float(np.dot(np.asarray(bm.to_numpy(conditions.full_force)), fa_u))
    # error = np.linalg.norm(u - fa_u) / np.linalg.norm(fa_u)
    # print(f'[自检] 位移相对误差 {error:.3e}, '
    #       f'柔顺度相对差 {abs(compliance - fa_compliance) / abs(fa_compliance):.3e}')

    # print(f'[全程] 峰值 RSS {peak_rss_bytes() / GIB:.2f} GiB')


if __name__ == "__main__":
    main()
