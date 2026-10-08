"""三维 MBB 梁的 full_trace 精确子结构拓扑优化入口.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from math import isfinite, prod

# 固定模型与算法参数. 
FINE_CELL_SIZE = 1.0
E0, EMIN, NU = 1.0, 1.0e-7, 0.3
LOAD, SUPPORT = -1.0, "end_corners"
TRACE = "full_trace"
INTEGRATION_ORDER = 2
VOLUME_FRACTION, INITIAL_DENSITY = 0.12, 0.12
SIMP_PENALTY, FILTER_RADIUS_CELLS = 3.0, 3.0
CONVERGENCE_TOLERANCE, CONVERGENCE_WINDOW = 2.0e-4, 5
VOLUME_TOLERANCE = 1.0e-6
MG_COARSE_MAX_DOFS = 20000
OC_OPTIONS = dict(move_limit=0.2, damping_coef=0.5, initial_lambda=1.0e9,
                  bisection_tol=1.0e-3, design_variable_min=0.0)


def parse_args(argv=None):
    """解析网格, 求解与输出参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-sub", type=int, nargs=3, default=(78, 13, 13), metavar=("NX", "NY", "NZ"),
        help="各方向子结构数 (默认: 78 13 13)",
    )
    parser.add_argument(
        "--n-fine", type=int, default=5,
        help="每个子结构各方向细单元数 m (默认: 5)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=32,
        help="局部刚度装配, 内部消元与位移恢复时每批最多处理的子结构数 (默认: 32)",
    )
    parser.add_argument(
        "--solver", choices=("cg", "scipy", "mumps"), default="cg",
        help="接口求解器 (默认: cg); scipy 和 mumps 使用稀疏直接法",
    )
    parser.add_argument("--cg-tol", type=float, default=None,
                        help="CG 真实相对残差容差 (默认: 1e-6)")
    parser.add_argument("--cg-maxiter", type=int, default=None,
                        help="CG 最大迭代次数 (默认: 20000)")
    parser.add_argument("--precond", choices=("none", "jacobi", "mg"), default=None,
                        help="CG 预条件子 (默认: mg); mg 使用隐式 Schur, 仅支持 numpy/cpu")
    parser.add_argument("--max-iter", type=int, default=300, help="最大分析次数 (默认: 300)")
    parser.add_argument("--symmetry", choices=("z", "none"), default="z",
                        help="设计对称约束 (默认: z, 每轮把灵敏度投影到关于 z 中面对称的子空间; "
                             "none 时不加约束, 长时间运行中舍入误差可能逐渐破坏对称)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="结果保存目录 (默认: 脚本目录下 outputs/接口_求解器), 重复运行覆盖同名文件")
    parser.add_argument(
        "--vtu-fields", nargs="+",
        choices=("density", "design_density", "displacement"), default=["density"],
        help="每轮及最终 VTU 保存的场, 可多选 (默认: density); displacement 写为向量 u",
    )
    parser.add_argument(
        "--backend", choices=("numpy", "pytorch"), default="numpy",
        help="全流程张量计算后端 (默认: numpy)",
    )
    parser.add_argument(
        "--device", default="cpu", help="密度与消元等数值计算设备: cpu, cuda 或 cuda:N (默认: cpu)",
    )
    args = parser.parse_args(argv)
    if args.solver != "cg" and any(
        value is not None for value in (args.cg_tol, args.cg_maxiter, args.precond)
    ):
        parser.error("cg-tol, cg-maxiter 和 precond 仅用于 solver=cg")
    if args.solver == "cg":
        args.cg_tol = 1.0e-6 if args.cg_tol is None else args.cg_tol
        args.cg_maxiter = 20000 if args.cg_maxiter is None else args.cg_maxiter
        args.precond = "mg" if args.precond is None else args.precond
        if not isfinite(args.cg_tol) or not 0.0 < args.cg_tol < 1.0:
            parser.error("cg-tol 必须为 (0, 1) 内的有限数")
        if args.cg_maxiter < 1:
            parser.error("cg-maxiter 必须为正整数")
    if args.device != "cpu" and args.device != "cuda" and not (
        args.device.startswith("cuda:") and args.device[5:].isascii() and args.device[5:].isdigit()
    ):
        parser.error("device 必须为 cpu, cuda 或 cuda:N")
    if args.backend == "numpy" and args.device != "cpu":
        parser.error("numpy 后端仅支持 cpu")
    if args.chunk_size < 1 or args.max_iter < 1:
        parser.error("chunk-size 与 max-iter 必须为正整数")
    if min(args.n_sub) < 1:
        parser.error("n-sub 各方向必须为正整数")
    if args.n_fine < 2:
        parser.error("n-fine 至少为 2，以保留内部节点")
    if args.precond == "mg" and (args.backend != "numpy" or args.device != "cpu"):
        parser.error("MG 当前仅支持 backend=numpy, device=cpu")
    args.vtu_fields = list(dict.fromkeys(args.vtu_fields))
    if args.output_dir is None:
        folder = f"{TRACE}_{args.solver}" + ("_mg" if args.precond == "mg" else "")
        args.output_dir = Path(__file__).resolve().parent / "outputs" / folder
    return args


def main(argv=None):
    """装配, 求解, 恢复与灵敏度分析后执行密度过滤和 OC 更新."""
    args = parse_args(argv)
    source_path = Path(__file__).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    print(f"[版本] {source_path}, SHA-256 {source_sha256}", flush=True)
    use_mg = args.solver == "cg" and args.precond == "mg"
    n_sub = tuple(args.n_sub)
    n_fine = (args.n_fine,) * 3
    grid = tuple(a * b for a, b in zip(n_sub, n_fine))
    count_estimate = prod(n_sub)
    nd = 3 * prod(m + 1 for m in n_fine)
    ni = 3 * prod(m - 1 for m in n_fine)
    nb = nd - ni
    nq = 3 * prod(m + 1 for m in grid) - count_estimate * ni
    csr_bytes_upper = count_estimate * nb * nb * 16 + (nq + 1) * 8
    symbolic_bytes = count_estimate * (nb // 3)**2 * 8 * 8
    assembly_budget = (csr_bytes_upper + symbolic_bytes
                       + args.chunk_size * nd * nd * 8 * 8 + nq * 8 * 20)
    print(f"[规模] 子结构 {count_estimate}, 局部接口 {nb}, 全局接口 {nq}", flush=True)
    if use_mg:
        nc, ndof = prod(grid), 3 * prod(n + 1 for n in grid)
        factor_bytes = count_estimate * ni * ni * 8
        assembly_budget = (factor_bytes + nc * 24 * 8 * 5 + ndof * 8 * 24
                           + prod((n + 1) // 2 for n in grid) * 24 * 24 * 8 * 5
                           + args.chunk_size * nd * nd * 8 * 5)
        print(f"[内存估算] 内部分解缓存 {factor_bytes / 2**30:.2f} GiB, "
              f"EA/MG 规划预算 {assembly_budget / 2**30:.2f} GiB, 非实测峰值", flush=True)
    else:
        print(f"[内存估算] CSR 上界 {csr_bytes_upper / 2**30:.2f} GiB, "
              f"装配规划预算 {assembly_budget / 2**30:.2f} GiB; 不含直接法分解因子",
              flush=True)
    try:
        available = next(int(line.split()[1]) * 1024
                         for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        available = None
    if available is not None and assembly_budget > 0.8 * available:
        raise MemoryError("规划预算超过当前可用内存的 80%; 请先释放内存或增加可用资源.")
    import numpy as np
    from soptx.backend import backend_manager as bm
    from soptx.fem.matrix import CSRChunkAccumulator
    from soptx.fem.substructure import (
        GlobalAssembler, InterfaceSystem, StructuredSubstructureLayout,
        build_interface_space, build_substructures, solve_constrained_system,
    )
    from soptx.postprocess.vtk_export import write_vtu
    from soptx.problems.elasticity import FullMBBBeam3d
    from soptx.topology.filters import (
        apply_structured_density_filter, apply_structured_density_filter_adjoint,
    )
    from soptx.topology.optimizers import OCOptimizer

    if args.solver == "mumps":
        try:
            from mumps import DMumpsContext
        except ImportError as exc:
            raise RuntimeError("MUMPS 依赖不可用, 需要能导入 mumps.DMumpsContext 的环境.") from exc

    device = args.device
    if args.backend == "pytorch":
        import torch
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("当前 PyTorch 环境中 CUDA 不可用.")
            index = torch.cuda.current_device() if device == "cuda" else int(device[5:])
            if index >= torch.cuda.device_count():
                raise ValueError(f"CUDA 设备编号越界: {index}.")
            device = f"cuda:{index}"
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device("cpu")
    # 全流程使用所选后端. CPU 装配与数值计算通过设备参数区分.
    compute_context = dict(dtype=bm.float64, device=device)
    index_context = dict(dtype=bm.int64, device=device)
    cpu_context = dict(dtype=bm.float64, device="cpu")
    solver = args.solver
    domain_size = tuple(FINE_CELL_SIZE * n for n in grid)
    domain = tuple(value for length in domain_size for value in (0.0, length))
    spacing: tuple[float, float, float] = (FINE_CELL_SIZE, FINE_CELL_SIZE, FINE_CELL_SIZE)
    radius = FILTER_RADIUS_CELLS * spacing[0]
    volume = VOLUME_FRACTION
    emin, penalty = EMIN / E0, SIMP_PENALTY
    oc_options = dict(OC_OPTIONS)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = dict(trace=TRACE, domain=domain, mode="optimization",
                  source_path=str(source_path), source_sha256=source_sha256,
                  fine_cell_size=FINE_CELL_SIZE, estimated_assembly_bytes=assembly_budget, backend=args.backend, device=device,
                  cpu_stages=(["mesh", "local_assembly", "implicit_schur", "mg", "cg", "output"]
                              if use_mg else ["mesh", "local_assembly", "global_assembly", solver, "output"]),
                  interface_operator="implicit_schur" if use_mg else "csr",
                  mg_coarse_max_dofs=MG_COARSE_MAX_DOFS if use_mg else None,
                  n_sub=n_sub, n_fine=n_fine, grid=grid, spacing=spacing, integration_order=INTEGRATION_ORDER,
                  E0=E0, Emin=EMIN, nu=NU, penalty=penalty, volfrac=volume,
                  initial_density=INITIAL_DENSITY, load=LOAD, filter_radius_cells=FILTER_RADIUS_CELLS,
                  filter_type="density", filter_radius=radius, optimizer=oc_options,
                  support=SUPPORT, symmetry=args.symmetry,
                  solver=solver, cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
                  precond=args.precond, cg_warm_start=(solver == "cg"), chunk_size=args.chunk_size, max_iter=args.max_iter,
                  tolerance=CONVERGENCE_TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
                  volume_tolerance=VOLUME_TOLERANCE, vtu_fields=args.vtu_fields)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[配置] {TRACE}, 网格 {grid}, 支承 {SUPPORT}, 对称约束 {args.symmetry}, 计算后端 {args.backend}, 数值设备 {device}, 求解器 {solver}, 输出 {output}", flush=True)
    print("[初始化] 创建布局、参考子结构和接口空间", flush=True)
    layout = StructuredSubstructureLayout(
        domain_size=domain_size, n_sub=n_sub, n_fine=n_fine,
        E_base=E0, nu=NU, hypothesis="3D",
    )
    assembler = GlobalAssembler(layout)
    prototype, sub_meshes, positions = build_substructures(
        assembler, integration_order=INTEGRATION_ORDER, penal=penalty, rho_min=emin,
    )
    problem = FullMBBBeam3d(
        domain=domain, E=E0, nu=NU, P=LOAD,
        support=SUPPORT, load_subdivisions=(grid[0], grid[2]),
    )
    print("[初始化] 投影载荷与支承约束", flush=True)
    mg_state = None
    if use_mg:
        mg_state = prepare_mg(layout, assembler, prototype, sub_meshes, positions, problem, grid)
    else:
        space = build_interface_space(
            kind=TRACE, assembler=assembler, sub_meshes=sub_meshes, prototype=prototype,
        )
        # 完整接口直接投影载荷和固定自由度, 不构造全局单位迹映射.
        from scipy.sparse import csr_matrix
        load, fixed = space.project_conditions(problem)
        fixed_numpy = np.asarray(bm.to_numpy(fixed), dtype=np.int64)
        constraints = csr_matrix(
            (np.ones(len(fixed_numpy)), (np.arange(len(fixed_numpy)), fixed_numpy)),
            shape=(len(fixed_numpy), space.n_global),
        )
        print("[接口编号] 开始构建 CSR 符号模式", flush=True)
        pattern = space.pattern
        print(f"[接口编号] CSR {pattern.sparse_shape}, 非零位置 {pattern.nnz:,}", flush=True)
    internal, boundary = prototype.i_dofs, prototype.b_dofs
    internal_local = bm.asarray(internal, **index_context)
    boundary_local = bm.asarray(boundary, **index_context)
    cell2dof_local = bm.asarray(prototype.cell2dof, **index_context)
    ke_unit = bm.asarray(prototype.KE_unit[0], **compute_context)
    count = len(sub_meshes)
    batches = (count + args.chunk_size - 1) // args.chunk_size
    rho = bm.full(grid, INITIAL_DENSITY, **compute_context)
    # 等体积网格: 体积分数对设计密度的梯度 W^T 1 / N, 同时用于物理体积判断. 取与 run_fa.py 的
    # 归一化约束灵敏度相同的尺度, OC 乘子的二分序列才与 FA 逐轮一致
    weights = apply_structured_density_filter_adjoint(
        gradient=bm.ones(grid, **compute_context), rmin=radius, spacing=spacing,
    ) / (grid[0] * grid[1] * grid[2])
    # 理论上已对称, 取与 z 向镜像的平均, 消去过滤求和顺序留下的舍入差异
    volume_gradient = 0.5 * (weights + bm.flip(weights, axis=2)) if args.symmetry == "z" else weights
    physical = None
    displacement = None
    previous_interface_displacement = None
    history = []
    recent = []
    converged = False
    export_mesh = None
    iteration_dir = output / "iterations"
    iteration_dir.mkdir(exist_ok=True)
    pvd_root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    pvd_collection = ET.SubElement(pvd_root, "Collection")
    run_started = perf_counter()
    for iteration in range(1, args.max_iter + 1):
        started = perf_counter()
        print(f"[迭代 {iteration}/{args.max_iter}] 密度过滤", flush=True)
        physical = apply_structured_density_filter(density=rho, rmin=radius, spacing=spacing)
        local_density = layout.split_global_cell_field(bm.asarray(physical.reshape(-1), **cpu_context))
        if use_mg:
            displacement, solved, assembly_seconds, solve_seconds, recovery_seconds = solve_mg(
                mg_state, prototype, local_density, physical, args,
                previous_interface_displacement, iteration)
            previous_interface_displacement = solved.displacement
            compliance = solved.compliance
            stamp = perf_counter()
            energy_local = bm.zeros((count, prototype.n_cells), **compute_context)
            for start in range(0, count, args.chunk_size):
                end = min(start + args.chunk_size, count)
                print(f"[恢复 {start // args.chunk_size + 1}/{batches}] 计算单元能量", flush=True)
                local_u = np.stack([
                    displacement[layout.get_substructure_global_dofs(pos, mesh)]
                    for pos, mesh in zip(positions[start:end], sub_meshes[start:end])])
                ue = local_u[:, cell2dof_local]
                energy_local[start:end] = bm.sum((ue @ ke_unit) * ue, axis=-1)
                del local_u, ue
                print(f"  全场恢复及单元能量完成, 累计 {end}/{count}", flush=True)
            mg_recovery_seconds = recovery_seconds
        else:
            accumulator = CSRChunkAccumulator(pattern)
            stiffness_batches = prototype.iter_local_stiffness_batches(local_density, chunk_size=args.chunk_size)
            for batch in range(batches):
                stamp = perf_counter()
                print(f"[装配 {batch + 1}/{batches}] 开始", flush=True)
                start, end, K = next(stiffness_batches)
                print(f"  局部装配完成，耗时 {perf_counter() - stamp:.3f} s", flush=True)
                K = bm.asarray(K, **compute_context)
                Kii = K[..., internal_local[:, None], internal_local]
                Kib = K[..., internal_local[:, None], boundary_local]
                Kbb = K[..., boundary_local[:, None], boundary_local]
                # full_trace 的 Psi = I, 直接计算完整边界 Schur 补.
                print("  刚度分块完成, 完整接口无迹降阶", flush=True)
                T = bm.linalg.solve(Kii, -Kib)
                print("  内部消元完成", flush=True)
                reduced = Kbb + bm.matrix_transpose(Kib) @ T
                accumulator.add(start, bm.asarray(reduced, **cpu_context))
                del K, Kii, Kib, Kbb, T, reduced
                print(f"  接口刚度散加完成，累计 {end}/{count}，本批 {perf_counter() - stamp:.3f} s", flush=True)
            system = InterfaceSystem(stiffness=accumulator.to_csr(), global_dofs=space.global_dofs)
            assembly_seconds = perf_counter() - started
            stamp = perf_counter()
            print("[接口求解] 开始", flush=True)
            cg_options = (dict(cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
                               precond=args.precond, x0=previous_interface_displacement)
                          if solver == "cg" else {})
            solved = solve_constrained_system(
                system=system, load=load, constraints=constraints, solver=solver, **cg_options)
            if solver == "cg":
                previous_interface_displacement = solved.displacement
                print(f"[CG] 迭代 {solved.iterations}, "
                      f"真实相对残差 {solved.equilibrium_relative_residual:.3e}", flush=True)
            Q = bm.asarray(solved.displacement, **compute_context)
            compliance = float(bm.sum(bm.asarray(load, **compute_context) * Q))
            if not isfinite(compliance) or compliance <= 0:
                raise FloatingPointError("柔顺度必须为有限正数")
            solve_seconds = perf_counter() - stamp
            print(f"[接口求解] C={compliance:.10e}，平衡残差 {solved.equilibrium_relative_residual:.3e}，约束残差 {solved.constraint_relative_residual:.3e}", flush=True)

            # 单右端恢复内部位移, 同时计算单位刚度单元能量.
            stamp = perf_counter()
            energy_local = bm.zeros((count, prototype.n_cells), **compute_context)
            displacement = bm.zeros((layout.total_full_dofs,), **compute_context)
            stiffness_batches = prototype.iter_local_stiffness_batches(local_density, chunk_size=args.chunk_size)
            for batch in range(batches):
                print(f"[恢复 {batch + 1}/{batches}] 开始", flush=True)
                start, end, K = next(stiffness_batches)
                print("  局部刚度重装配完成", flush=True)
                q_local = bm.asarray(Q[bm.asarray(space.local_dofs[start:end], **index_context)], **compute_context)
                ub = q_local
                K = bm.asarray(K, **compute_context)
                Kii = K[..., internal_local[:, None], internal_local]
                Kib = K[..., internal_local[:, None], boundary_local]
                ui = bm.linalg.solve(Kii, -(Kib @ ub[..., None]))[..., 0]
                print("  内部位移单右端求解完成", flush=True)
                local_u = bm.zeros((end - start, prototype.n_total_dofs), **compute_context)
                local_u[:, boundary_local] = ub
                local_u[:, internal_local] = ui
                ue = local_u[:, cell2dof_local]
                energy_local[start:end] = bm.sum((ue @ ke_unit) * ue, axis=-1)
                for offset, (pos, mesh) in enumerate(zip(positions[start:end], sub_meshes[start:end])):
                    dofs = layout.get_substructure_global_dofs(pos, mesh)
                    displacement[bm.asarray(dofs, **index_context)] = local_u[offset]
                del K, Kii, Kib, ub, ui, local_u, ue, q_local
                print(f"  全场写回及单元能量完成，累计 {end}/{count}", flush=True)
            del system, accumulator
        energy = layout.merge_substructure_cell_field(
            prototype.cell_to_grid_field(energy_local)).reshape(grid)
        if not bm.all(bm.isfinite(energy)) or not bm.all(bm.isfinite(displacement)):
            raise FloatingPointError("恢复位移或单元能量非有限")
        energy_work = float(bm.sum((emin + (1 - emin) * physical**penalty) * energy))
        energy_error = abs(energy_work - compliance) / compliance
        recovery_seconds = perf_counter() - stamp + (mg_recovery_seconds if use_mg else 0.0)
        dc_physical = -penalty * (1 - emin) * physical**(penalty - 1) * energy
        dc = apply_structured_density_filter_adjoint(gradient=dc_physical, rmin=radius, spacing=spacing)
        if not bm.all(bm.isfinite(dc)):
            raise FloatingPointError("柔顺度灵敏度非有限")
        # 取与 z 向镜像的平均, 投影到对称设计的子空间: 共用变量的梯度为镜像两单元之和, 取平均
        # 只差常数因子, 不影响 OC 的乘子; 浮点加法可交换, 结果逐位对称
        if args.symmetry == "z":
            dc = 0.5 * (dc + bm.flip(dc, axis=2))
        relative = None if not history else abs(compliance - history[-1]["compliance"]) / compliance
        if relative is not None:
            recent.append(relative)
            recent = recent[-CONVERGENCE_WINDOW:]
        converged = len(recent) == CONVERGENCE_WINDOW and all(value < CONVERGENCE_TOLERANCE for value in recent)
        # 逐轮记录关于 z 中面的不对称量; 开启 z 对称约束时设计密度的不对称量应恒为 0
        symmetry_z = float(bm.max(bm.abs(physical - bm.flip(physical, axis=2))))
        design_symmetry_z = float(bm.max(bm.abs(rho - bm.flip(rho, axis=2))))
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=float(physical.mean()),
                      relative_change=relative, symmetry_error_z=symmetry_z,
                      design_symmetry_error_z=design_symmetry_z, equilibrium_residual=float(solved.equilibrium_relative_residual),
                      solver_iterations=solved.iterations, solver_converged=solved.converged,
                      constraint_residual=float(solved.constraint_relative_residual), energy_relative_error=energy_error,
                      assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
                      recovery_seconds=recovery_seconds, analysis_seconds=perf_counter() - started)
        if use_mg:
            record["interface_residual"] = solved.interface_relative_residual
        del solved, local_density, energy_local
        # 每轮分析完成后保存所选场, VTU 发布后才更新 PVD.
        export_started = perf_counter()
        if export_mesh is None:
            full_mesh = layout.full_mesh
            export_mesh = SimpleNamespace(entity=lambda kind: bm.to_numpy(full_mesh.entity(kind)))
        cell_data = {}
        point_data = {}
        if "density" in args.vtu_fields:
            cell_data["density"] = bm.to_numpy(physical).reshape(-1)
        if "design_density" in args.vtu_fields:
            cell_data["design_density"] = bm.to_numpy(rho).reshape(-1)
        if "displacement" in args.vtu_fields:
            point_data["u"] = bm.to_numpy(displacement).reshape(-1, 3)
        frame = iteration_dir / f"iter_{iteration:04d}.vtu"
        temporary_base = iteration_dir / f".iter_{iteration:04d}.tmp"
        temporary_vtu = Path(str(temporary_base) + ".vtu")
        try:
            write_vtu(mesh=export_mesh, filepath=str(temporary_base),
                      cell_data=cell_data, point_data=point_data)
            temporary_vtu.replace(frame)
        finally:
            temporary_vtu.unlink(missing_ok=True)
        relative_frame = frame.relative_to(output).as_posix()
        ET.SubElement(pvd_collection, "DataSet", timestep=str(iteration),
                      group="", part="0", file=relative_frame)
        temporary_pvd = output / ".evolution.pvd.tmp"
        try:
            ET.ElementTree(pvd_root).write(temporary_pvd, encoding="utf-8", xml_declaration=True)
            temporary_pvd.replace(output / "evolution.pvd")
        finally:
            temporary_pvd.unlink(missing_ok=True)
        del cell_data, point_data
        record["vtu_file"] = relative_frame
        record["export_seconds"] = perf_counter() - export_started
        print(f"[输出] {relative_frame}, 字段 {args.vtu_fields}", flush=True)
        history.append(record)
        print(f"[迭代结果] C={compliance:.10e}，体积分数 {physical.mean():.6f}，相对变化 {relative}，能量相对差 {energy_error:.3e}", flush=True)
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # 停止时保留刚完成分析的密度, 避免最终柔顺度与密度错位.
        if converged or iteration == args.max_iter:
            break
        print("[OC] 更新设计密度", flush=True)
        updated_density = OCOptimizer.update_design_variable(
            design_variable=rho,
            objective_gradient=dc,
            constraint_gradient=volume_gradient,
            constraint_function=lambda candidate: bm.sum(volume_gradient * candidate) - volume,
            **oc_options,
        )
        rho_new = bm.asarray(updated_density, **compute_context)
        if not bm.all(bm.isfinite(rho_new)) or bm.min(rho_new) < 0 or bm.max(rho_new) > 1:
            raise FloatingPointError("OC 更新得到无效密度")
        candidate_volume = float(bm.sum(weights * rho_new))
        if candidate_volume > volume + VOLUME_TOLERANCE:
            raise RuntimeError("OC 乘子搜索未满足物理体积约束")
        print(f"[OC] 最大密度变化 {bm.max(bm.abs(rho_new - rho)):.3e}", flush=True)
        rho = rho_new

    if physical is None or displacement is None or not history:
        raise RuntimeError("优化未完成任何分析, 无法保存最终结果.")
    np.save(output / "design_density_final.npy", bm.to_numpy(rho))
    np.save(output / "density_final.npy", bm.to_numpy(physical))
    np.save(output / "displacement_final.npy", bm.to_numpy(displacement))
    # 最后一帧与最终 NPY 对应同一次分析, 复用已导出的所选场.
    final_temporary = output / ".result_final.vtu.tmp"
    try:
        shutil.copyfile(output / history[-1]["vtu_file"], final_temporary)
        final_temporary.replace(output / "result_final.vtu")
    finally:
        final_temporary.unlink(missing_ok=True)
    # x 向两端支承不同, 只作诊断记录, 不施加约束
    symmetry_x = float(bm.max(bm.abs(physical - bm.flip(physical, axis=0))))
    summary = dict(history[-1], converged=converged,
                   termination="converged" if converged else "max_iter",
                   symmetry_error_x=symmetry_x,
                   total_seconds=perf_counter() - run_started)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[完成] {summary['termination']}，结果保存到 {output}", flush=True)
    return 0



def prepare_mg(layout, assembler, prototype, sub_meshes, positions, problem, grid):
    """准备各轮共用的网格, 接口编号与 MG 几何层级.

    Parameters
    ----------
    layout, assembler, prototype : object
        全场布局, 装配器与参考子结构.
    sub_meshes, positions : sequence
        子结构网格与位置.
    problem : FullMBBBeam3d
        固定工况.
    grid : tuple
        全局细网格单元数.

    Returns
    -------
    SimpleNamespace
        不含密度相关分解的共用数据.
    """
    import numpy as np
    from soptx.fem.kernels import ElementRestriction
    from soptx.fem.multigrid import StructuredHexGrid, StructuredHexHierarchy
    from soptx.fem.substructure import (
        InterfaceDofsView, project_problem_conditions_to_interface_system,
    )
    ndof = layout.total_full_dofs
    global_interface = np.asarray(layout.build_interface_dofs(sub_meshes), dtype=np.int64)
    conditions = project_problem_conditions_to_interface_system(
        problem, assembler, InterfaceDofsView(global_dofs=global_interface))
    force = np.asarray(conditions.full_force)
    fixed = np.asarray(conditions.full_fixed_dofs, dtype=np.int64)
    free_interface = np.setdiff1d(global_interface, fixed)
    force = force.copy()
    force[fixed] = 0.0
    norm_force = float(np.linalg.norm(force))
    if norm_force == 0:
        raise ValueError("CG(MG) 需要非零自由载荷.")
    internal_map = np.stack([
        np.asarray(layout.get_substructure_global_dofs(pos, mesh))[prototype.i_dofs]
        for pos, mesh in zip(positions, sub_meshes)])
    space = layout.space_full
    if space.dof_priority or prototype.space.dof_priority:
        raise ValueError("当前 MG 路径要求节点优先的三分量自由度排列.")
    structured = StructuredHexGrid.from_mesh(layout.full_mesh)
    if structured.shape != grid:
        raise ValueError("完整网格编号与结构化布局不符.")
    ke = np.asarray(prototype.KE_unit[:1])
    if not np.allclose(prototype.KE_unit, ke, rtol=1e-12, atol=1e-14):
        raise ValueError("局部网格不能共享单一参考刚度.")
    # 明确比较单元局部节点顺序, 不只检查矩阵维数.
    full_nodes = layout.full_mesh.entity("node")
    local_nodes = prototype.space.mesh.entity("node")
    full_cell = layout.full_mesh.entity("cell")[0]
    local_cell = prototype.space.mesh.entity("cell")[0]
    full_offsets = full_nodes[full_cell] - full_nodes[full_cell].min(axis=0)
    local_offsets = local_nodes[local_cell] - local_nodes[local_cell].min(axis=0)
    if not np.allclose(full_offsets, local_offsets, rtol=0, atol=1e-12):
        raise ValueError("参考单元与全场单元局部顺序不一致.")
    restriction = ElementRestriction(space.cell_to_dof(), ndof)
    mask = np.zeros(ndof, dtype=bool)
    mask[fixed] = True
    # 细层为 EA 算子, 至少粗化一次以生成最粗层直接法所需的 CSR.
    hierarchy = StructuredHexHierarchy(
        space, mask, ke[0], coarse_max_dofs=min(MG_COARSE_MAX_DOFS, ndof - 1))
    return SimpleNamespace(space=space, restriction=restriction, ke=ke, mask=mask,
                           hierarchy=hierarchy, internal_map=internal_map, ndof=ndof,
                           free_interface=free_interface, fixed=fixed, force=force,
                           norm_force=norm_force)


def solve_mg(state, prototype, local_density, physical, args, previous, iteration):
    """按本轮密度重建内部 Cholesky 与 MG, 求解并恢复完整位移.

    Parameters
    ----------
    state : SimpleNamespace
        共用网格和 MG 几何层级.
    prototype : object
        参考子结构.
    local_density, physical : numpy.ndarray
        本轮局部及全场物理密度.
    args : argparse.Namespace
        求解参数.
    previous : numpy.ndarray or None
        上轮自由接口位移, 用作热启动.
    iteration : int
        优化轮次.

    Returns
    -------
    displacement : numpy.ndarray
        全场位移.
    solved : SimpleNamespace
        接口位移, 柔顺度与真实残差.
    assembly_seconds, solve_seconds, recovery_seconds : float
        各阶段耗时.
    """
    import numpy as np
    from soptx.fem.levels import SharedReferenceElementAssembly
    from soptx.fem.operators import ConstrainedOperator
    from soptx.fem.substructure.implicit import (
        InteriorBlockSolver, ImplicitSchurOperator, RestrictedPreconditioner,
    )
    from soptx.solvers import CGSolver

    started = perf_counter()
    interior = InteriorBlockSolver(state.internal_map, state.ndof)
    for begin, end, local_k in prototype.iter_local_stiffness_batches(
            local_density, chunk_size=args.chunk_size):
        print(f"[装配] 内部块 {begin}:{end}, 局部刚度装配完成", flush=True)
        interior.factor_batch(begin, local_k[:, prototype.i_dofs[:, None], prototype.i_dofs])
        print(f"  内部 Cholesky 分解完成, 累计 {end}/{len(state.internal_map)}", flush=True)
    del local_k
    ratio = EMIN / E0
    coef = ratio + (1.0 - ratio) * physical.reshape(-1)**SIMP_PENALTY
    ea = SharedReferenceElementAssembly(state.space, state.restriction, state.ke, scale=coef)
    full_operator = ConstrainedOperator(ea, isDDof=state.mask)
    schur = ImplicitSchurOperator(full_operator, interior, state.free_interface, state.fixed)
    print("[MG] 按当前物理密度更新全部粗层算子", flush=True)
    state.hierarchy.update(coef)
    mg = state.hierarchy.build_multigrid(coarse_solver="scipy")
    try:
        mg.setup(full_operator)
        preconditioner = RestrictedPreconditioner(mg, state.free_interface, state.ndof)
        assembly_seconds = perf_counter() - started
        print(f"[接口求解] CG(MG), 层数 {state.hierarchy.num_levels}, "
              f"自由接口 {schur.shape[0]}", flush=True)
        started = perf_counter()

        def monitor(step, norm, residual, final):
            if step % 10 == 0 or final:
                print(f"[CG 第 {iteration} 轮] 迭代 {step}, "
                      f"监控相对残差 {float(norm) / state.norm_force:.3e}", flush=True)

        solver = CGSolver(M=preconditioner, rtol=0.0,
                          atol=args.cg_tol * state.norm_force,
                          maxit=args.cg_maxiter, norm_type="unpreconditioned", monitor=monitor)
        q, info = solver.setup(schur).solve(
            state.force[state.free_interface],
            x0=None if previous is None else previous.copy())
        solve_seconds = perf_counter() - started
        started = perf_counter()
        print("[恢复] 内部单右端回代并核对全场真实残差", flush=True)
        u = schur.recover(q)
        residual = full_operator @ u - state.force
        full_relative = float(np.linalg.norm(residual)) / state.norm_force
        interface_relative = float(np.linalg.norm(residual[state.free_interface])) / state.norm_force
        compliance = float(state.force @ u)
        if (not info["converged"] or not np.isfinite(u).all()
                or not isfinite(full_relative) or full_relative > args.cg_tol
                or not isfinite(interface_relative) or interface_relative > args.cg_tol):
            raise RuntimeError(f"CG(MG) 未达到真实残差容差: 全场 {full_relative:.3e}, "
                               f"接口 {interface_relative:.3e}.")
        if not isfinite(compliance) or compliance <= 0:
            raise FloatingPointError("柔顺度必须为有限正数.")
        constraint = float(np.linalg.norm(u[state.fixed])) / max(float(np.linalg.norm(u)), 1e-30)
        solved = SimpleNamespace(displacement=q, compliance=compliance,
                                 iterations=int(info["niter"]), converged=True,
                                 equilibrium_relative_residual=full_relative,
                                 constraint_relative_residual=constraint,
                                 interface_relative_residual=interface_relative)
        print(f"[接口求解] C={compliance:.10e}, 全场残差 {full_relative:.3e}, "
              f"接口残差 {interface_relative:.3e}, 约束残差 {constraint:.3e}", flush=True)
        return u, solved, assembly_seconds, solve_seconds, perf_counter() - started
    finally:
        if mg.coarse_solver is not None:
            mg.coarse_solver.close()


if __name__ == "__main__":
    raise SystemExit(main())
