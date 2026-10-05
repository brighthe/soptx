"""精确子结构流程: 局部装配, Schur 补缩聚, 接口迹降阶, 接口装配求解与内部位移恢复."""

import argparse
from math import isfinite, prod
from time import perf_counter


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
        "--chunk-size", type=int, default=256,
        help="单次装配并缩聚的子结构数; 局部刚度 K^j 只在分块内存在",
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
    if args.chunk_size <= 0:
        parser.error("--chunk-size 须为正整数")
    if not isfinite(args.mem_limit_gb) or args.mem_limit_gb < 1 / 2**30:
        parser.error("--mem-limit-gb 须为有限正数且至少为 1 字节")
    return args


def main(argv=None):
    """依次创建子结构, 生成密度场, 逐块完成局部静力缩聚与接口迹降阶, 组装求解接口方程并恢复全场位移."""
    args = parse_args(argv)

    import resource

    GIB = 2**30
    limit = int(args.mem_limit_gb * GIB)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    from soptx.backend import backend_manager as bm
    from soptx.fem.matrix import CSRChunkAccumulator
    from soptx.fem.substructure import (
        GlobalAssembler,
        InterfaceSystem,
        StructuredSubstructureLayout,
        build_interface_space,
        build_substructures,
        solve_constrained_system,
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

    # 数值求解: 接口系统与同网格 FA 对照共用的直接求解器; 局部缩聚每次处理 chunk_size 个子结构.
    solver = args.solver
    chunk_size = args.chunk_size

    # ------------------------------------------------------------------
    # 子结构划分: 整体布局, 局部密度, 全局装配器, 参考子结构与子结构实例
    # ------------------------------------------------------------------
    # 构造整体布局: 将求解域划分为互不重叠的同构子结构.
    domain_size = domain[1::2]
    layout = StructuredSubstructureLayout(
        domain_size=domain_size,
        n_sub=n_sub,
        n_fine=n_fine,
        E_base=E_base,
        nu=nu,
        hypothesis=hypothesis,
    )
    # 构造子结构单元密度
    density = bm.linspace(0.5, 0.9, prod(layout.total_fine))
    local_density = layout.split_global_cell_field(density)
    print(f"[输入] 求解域 {domain}, 密度 (0.5, 0.9)")
    # 构造全局装配器: 提供全局接口刚度的逐块散加入口, 供接口空间调用.
    assembler = GlobalAssembler(layout)
    # 构造参考子结构与子结构实例: 与 j 无关的量只在 prototype 上构造一次, SIMP 参数存入 prototype.
    prototype, sub_meshes, sub_positions = build_substructures(
        assembler=assembler,
        penal=penalty,
        rho_min=0.0,
    )

    # ------------------------------------------------------------------
    # 构造接口空间
    # ------------------------------------------------------------------
    interface_space = build_interface_space(
        kind=trace_kind,
        assembler=assembler,
        sub_meshes=sub_meshes,
        prototype=prototype,
    )
    trace = interface_space.trace_basis
    n_q = interface_space.n_trace_dofs

    print(
        f"[局部缩聚] 子结构 {len(sub_meshes):,} 个, 单子结构 "
        f"K^j {(prototype.n_total_dofs,) * 2}, "
        f"T_q^j {(prototype.n_i, n_q)}; chunk_size {chunk_size}, "
        f"单分块 K^j 至多 {chunk_size * prototype.n_total_dofs**2 * 8 / GIB:.2f} GiB"
    )
    print(
        f"[迹降阶] {trace.name}: Psi {tuple(trace.matrix.shape)}, "
        f"单子结构 K_r^j {(n_q, n_q)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 构造全局接口刚度
    # ------------------------------------------------------------------
    print("[接口编号] 开始建立 CSR 模式", flush=True)
    pattern = interface_space.pattern
    accumulator = CSRChunkAccumulator(pattern)
    print(
        f"[接口编号] local_dofs {tuple(interface_space.local_dofs.shape)}, "
        f"CSR 模式 {pattern.shape}, 非零位置 {pattern.nnz:,}",
        flush=True,
    )
    i_dofs, b_dofs = prototype.i_dofs, prototype.b_dofs
    Psi = trace.matrix

    n_substructures = len(sub_meshes)
    n_batches = (n_substructures + chunk_size - 1) // chunk_size
    stiffness_batches = prototype.iter_local_stiffness_batches(
        local_density, chunk_size=chunk_size,
    )
    for batch_index in range(n_batches):
        batch_number = batch_index + 1
        expected_start = batch_index * chunk_size
        expected_end = min(expected_start + chunk_size, n_substructures)
        print(
            f"[批次 {batch_number}/{n_batches}] 开始, "
            f"子结构 [{expected_start}, {expected_end})",
            flush=True,
        )
        batch_started = perf_counter()
        step_started = perf_counter()
        start, end, K_local = next(stiffness_batches)
        print(f"  局部刚度装配完成, 耗时 {perf_counter() - step_started:.3f} s", flush=True)
        step_started = perf_counter()
        K_ii = K_local[..., i_dofs[:, None], i_dofs]
        K_ib = K_local[..., i_dofs[:, None], b_dofs]
        K_bb = K_local[..., b_dofs[:, None], b_dofs]

        B = K_ib if trace_kind == "full_trace" else K_ib @ Psi
        K_qq = trace.project_stiffness(K_bb)
        print(f"  刚度分块与迹投影完成, 耗时 {perf_counter() - step_started:.3f} s", flush=True)

        step_started = perf_counter()
        T_q = bm.linalg.solve(K_ii, -B)
        print(f"  内部消元完成, 耗时 {perf_counter() - step_started:.3f} s", flush=True)

        step_started = perf_counter()
        K_r = K_qq + bm.matrix_transpose(B) @ T_q
        print(f"  局部迹刚度计算完成, 耗时 {perf_counter() - step_started:.3f} s", flush=True)
        if start == 0:
            print(
                f"[首批局部装配] 子结构 [{start}, {end}), K {tuple(K_local.shape)}; "
                f"K_ii {tuple(K_ii.shape)}, K_ib {tuple(K_ib.shape)}, "
                f"K_bb {tuple(K_bb.shape)}",
                flush=True,
            )
            print(
                f"[首批迹空间消元] B {tuple(B.shape)}, T_q {tuple(T_q.shape)}, "
                f"K_qq {tuple(K_qq.shape)}, K_r {tuple(K_r.shape)}",
                flush=True,
            )

        step_started = perf_counter()
        accumulator.add(start, K_r)
        print(f"  全局散加完成, 耗时 {perf_counter() - step_started:.3f} s", flush=True)
        # 本阶段仅组装刚度, 恢复算子不跨批保存; 后续恢复需逐块重算.
        del K_local, K_ii, K_ib, K_bb, B, K_qq, T_q, K_r
        # 百分比表示装配覆盖的子结构比例, 不代表总运行时间比例.
        print(
            f"[批次 {batch_number}/{n_batches}] 完成, "
            f"累计 {end}/{n_substructures} ({end / n_substructures:.2%}), "
            f"本批 {perf_counter() - batch_started:.3f} s",
            flush=True,
        )

    system = InterfaceSystem(
        stiffness=accumulator.to_csr(),
        global_dofs=interface_space.global_dofs,
    )
    print(
        f"[接口装配] {trace.name}: N_q = {len(system.global_dofs):,}, "
        f"K_Q {tuple(system.stiffness.shape)}, 非零元 {system.stiffness.nnz:,}",
        flush=True,
    )

    # 组装全局接口载荷与约束
    print("[接口载荷] 开始构造载荷与约束", flush=True)
    step_started = perf_counter()
    load, constraints = interface_space.constrained_conditions(pde)
    print(
        f"[接口载荷] F_Q {tuple(load.shape)}, 约束 C_D {constraints.shape}, "
        f"耗时 {perf_counter() - step_started:.3f} s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 支承约束与接口求解
    # ------------------------------------------------------------------
    # 求解接口方程.
    print(f"[接口求解] 开始, solver={solver}", flush=True)
    step_started = perf_counter()
    solved = solve_constrained_system(
        system=system,
        load=load,
        constraints=constraints,
        solver=solver,
    )
    Q = solved.displacement
    print(
        f"[接口求解] {solved.mode}, 约束秩 {solved.constraint_rank}, "
        f"平衡相对残差 {solved.equilibrium_relative_residual:.3e}, "
        f"约束相对残差 {solved.constraint_relative_residual:.3e}, "
        f"柔顺度 F_Q^T Q = {float(bm.dot(load, Q)):.10e}, "
        f"耗时 {perf_counter() - step_started:.3f} s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 子结构位移恢复与全场拼接
    # ------------------------------------------------------------------
    print("[位移恢复] 开始逐批恢复并拼接全场位移", flush=True)
    recovery_started = perf_counter()
    displacement = bm.zeros((layout.total_full_dofs,), dtype=bm.float64)
    n_substructures = len(sub_meshes)
    n_batches = (n_substructures + chunk_size - 1) // chunk_size
    stiffness_batches = prototype.iter_local_stiffness_batches(
        local_density, chunk_size=chunk_size,
    )
    for batch_index in range(n_batches):
        batch_number = batch_index + 1
        print(f"[恢复批次 {batch_number}/{n_batches}] 开始", flush=True)
        batch_started = perf_counter()
        step_started = perf_counter()
        start, end, K_local = next(stiffness_batches)
        print(
            f"  子结构 [{start}, {end}), 局部刚度重装配完成, "
            f"耗时 {perf_counter() - step_started:.3f} s",
            flush=True,
        )

        q = Q[interface_space.local_dofs[start:end]]
        u_b = trace.expand_displacement(q)

        K_ii = K_local[..., i_dofs[:, None], i_dofs]
        K_ib = K_local[..., i_dofs[:, None], b_dofs]
        rhs = -(K_ib @ u_b[..., None])
        step_started = perf_counter()
        u_i = bm.linalg.solve(K_ii, rhs)[..., 0]
        print(
            f"  内部位移单右端求解完成, "
            f"耗时 {perf_counter() - step_started:.3f} s",
            flush=True,
        )
        if start == 0:
            print(
                f"[首批位移恢复] q {tuple(q.shape)}, u_b {tuple(u_b.shape)}, "
                f"右端 {tuple(rhs.shape)}, u_i {tuple(u_i.shape)}",
                flush=True,
            )
        del K_local, K_ii, K_ib, rhs, q

        global_dofs = bm.stack(
            [
                layout.get_substructure_global_dofs(pos, sub_mesh)
                for pos, sub_mesh in zip(
                    sub_positions[start:end], sub_meshes[start:end]
                )
            ],
            axis=0,
        )
        displacement = bm.set_at(
            displacement,
            bm.reshape(global_dofs[:, b_dofs], (-1,)),
            bm.reshape(u_b, (-1,)),
        )
        displacement = bm.set_at(
            displacement,
            bm.reshape(global_dofs[:, i_dofs], (-1,)),
            bm.reshape(u_i, (-1,)),
        )
        del global_dofs, u_b, u_i
        print(
            f"[恢复批次 {batch_number}/{n_batches}] 全场写回完成, "
            f"累计 {end}/{n_substructures} ({end / n_substructures:.2%}), "
            f"本批 {perf_counter() - batch_started:.3f} s",
            flush=True,
        )

    print(
        f"[位移恢复] 全场 displacement {tuple(displacement.shape)}, "
        f"耗时 {perf_counter() - recovery_started:.3f} s",
        flush=True,
    )

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
