"""三维 MBB 梁的 linear_corner 精确子结构拓扑优化入口.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from shutil import copyfile
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from math import isfinite

# 固定模型与算法参数. 命令行默认值直接定义在 parse_args 中.
FINE_CELL_SIZE = 1.0
E0, EMIN, NU = 1.0, 1.0e-7, 0.3
LOAD, SUPPORT = -1.0, "end_lines"
TRACE, SOLVER = "linear_corner", "mumps"
INTEGRATION_ORDER = 2
VOLUME_FRACTION, INITIAL_DENSITY = 0.12, 0.12
SIMP_PENALTY, FILTER_RADIUS_CELLS = 3.0, 3.0
CONVERGENCE_TOLERANCE, CONVERGENCE_WINDOW = 2.0e-4, 5
VOLUME_TOLERANCE = 1.0e-6
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
        "--chunk-size", type=int, default=256,
        help="局部刚度装配、内部消元与位移恢复时每批最多处理的子结构数",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="最大分析次数 (默认: 300)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="结果目录 (默认: 优化 outputs, 单次分析 outputs_analysis)")
    parser.add_argument(
        "--backend", choices=("numpy", "pytorch"), default="numpy",
        help="全流程张量计算后端 (默认: numpy)",
    )
    parser.add_argument(
        "--device", default="cpu", help="密度与消元等数值计算设备: cpu, cuda 或 cuda:N (默认: cpu)",
    )
    parser.add_argument("--physical-density", type=Path,
                        help="读取物理密度 NPY, 只分析一次, 不过滤或执行 OC")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        dirname = "outputs_analysis" if args.physical_density is not None else "outputs"
        args.output_dir = Path(__file__).resolve().parent / dirname
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
    return args


def main(argv=None):
    """装配, 求解, 恢复与灵敏度分析后执行密度过滤和 OC 更新."""
    args = parse_args(argv)
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    source_hash = sha256(source_bytes).hexdigest()
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
    n_sub = tuple(args.n_sub)
    n_fine = (args.n_fine,) * 3
    solver = SOLVER
    # 复现长度约定: 细单元边长为 1, 并非原文明示.
    grid = tuple(a * b for a, b in zip(n_sub, n_fine))
    spacing = (FINE_CELL_SIZE,) * 3
    domain_size = tuple(n * FINE_CELL_SIZE for n in grid)
    domain = tuple(value for length in domain_size for value in (0.0, length))
    radius = FILTER_RADIUS_CELLS * spacing[0]
    volume = VOLUME_FRACTION
    emin, penalty = EMIN / E0, SIMP_PENALTY
    oc_options = dict(OC_OPTIONS)
    analysis_only = args.physical_density is not None
    n_analyses = 1 if analysis_only else args.max_iter
    input_physical = None
    output = args.output_dir.resolve()
    if analysis_only:
        source = args.physical_density.resolve()
        if output == source.parent or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("单次分析须使用新的或空的输出目录, 避免覆盖已有结果.")
        loaded = np.load(source, allow_pickle=False)
        if not isinstance(loaded, np.ndarray) or loaded.shape != grid:
            raise ValueError(f"物理密度必须为形状 {grid} 的 NPY 数组.")
        if loaded.dtype.kind not in "fiu" or not np.all(np.isfinite(loaded)):
            raise ValueError("物理密度必须为有限实数数组.")
        if np.any(loaded < 0.0) or np.any(loaded > 1.0):
            raise ValueError("物理密度必须位于 [0, 1].")
        input_physical = bm.asarray(loaded, **compute_context)
        del loaded
    output.mkdir(parents=True, exist_ok=True)
    # 保留启动时的入口源码, 便于追溯实际运行的磁盘版本.
    (output / "run_linear_corner.snapshot.py").write_bytes(source_bytes)
    config = dict(trace=TRACE, domain=domain, backend=args.backend, device=device,
                  mode="analysis_only" if analysis_only else "optimization",
                  physical_density_source=str(args.physical_density.resolve()) if analysis_only else None,
                  fine_cell_size=FINE_CELL_SIZE, length_convention_status="复现推断, 原文未明确",
                  analysis_limit=n_analyses, filter_applied=not analysis_only,
                  source_path=str(source_path), source_sha256=source_hash,
                  iteration_output="iterations/iter_NNNN.vtu", evolution_index="evolution.pvd",
                  cpu_stages=["mesh", "local_assembly", "global_assembly", "mumps", "output"],
                  n_sub=n_sub, n_fine=n_fine, grid=grid, spacing=spacing, integration_order=INTEGRATION_ORDER,
                  E0=E0, Emin=EMIN, nu=NU, penalty=penalty, volfrac=volume,
                  initial_density=INITIAL_DENSITY, load=LOAD, filter_radius_cells=FILTER_RADIUS_CELLS,
                  filter_type="density", filter_radius=radius, optimizer=oc_options,
                  support=SUPPORT, support_status="建模选择，未确认与论文一致",
                  solver=solver, chunk_size=args.chunk_size, max_iter=args.max_iter,
                  tolerance=CONVERGENCE_TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
                  volume_tolerance=VOLUME_TOLERANCE)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[配置] 接口 {TRACE}, 后端 {args.backend}, 设备 {device}, 模式 {config['mode']}", flush=True)
    print(f"[版本] {source_path}, SHA-256 {source_hash}", flush=True)
    print("[网格] HexahedronMesh, 八节点 Q1 单元, p=1", flush=True)
    print(f"[划分] 子结构网格 {n_sub}, 子结构内细单元划分 {n_fine}, 全局细网格 {grid}", flush=True)
    print(f"[尺度] h={FINE_CELL_SIZE}, 计算域尺寸 {domain_size}, 过滤半径 {radius}", flush=True)
    print(f"[输出] {output}, 每轮 VTU + evolution.pvd", flush=True)
    print("[初始化] 创建布局、参考子结构和接口空间", flush=True)
    layout = StructuredSubstructureLayout(
        domain_size=domain_size, n_sub=n_sub, n_fine=n_fine,
        E_base=E0, nu=NU, hypothesis="3D",
    )
    assembler = GlobalAssembler(layout)
    prototype, sub_meshes, positions = build_substructures(
        assembler, integration_order=INTEGRATION_ORDER, penal=penalty, rho_min=emin,
    )
    space = build_interface_space(
        kind=TRACE, assembler=assembler, sub_meshes=sub_meshes, prototype=prototype,
    )
    problem = FullMBBBeam3d(
        domain=domain, E=E0, nu=NU, P=LOAD,
        support=SUPPORT, load_subdivisions=(grid[0], grid[2]),
    )
    print("[初始化] 投影载荷与支承约束", flush=True)
    load, constraints = space.constrained_conditions(problem)
    pattern = space.pattern
    trace = space.trace_basis
    psi = bm.asarray(trace.matrix, **compute_context)
    internal, boundary = prototype.i_dofs, prototype.b_dofs
    internal_local = bm.asarray(internal, **index_context)
    boundary_local = bm.asarray(boundary, **index_context)
    cell2dof_local = bm.asarray(prototype.cell2dof, **index_context)
    ke_unit = bm.asarray(prototype.KE_unit[0], **compute_context)
    count = len(sub_meshes)
    batches = (count + args.chunk_size - 1) // args.chunk_size
    rho = None if analysis_only else bm.full(grid, INITIAL_DENSITY, **compute_context)
    # 等体积网格: W^T 1 同时用于物理体积判断及其设计梯度.
    weights = None
    if not analysis_only:
        weights = apply_structured_density_filter_adjoint(
            gradient=bm.ones(grid, **compute_context), rmin=radius, spacing=spacing,
        )
    volume_gradient = weights
    physical = None
    displacement = None
    history = []
    recent = []
    converged = False
    iterations_dir = output / "iterations"
    iterations_dir.mkdir(exist_ok=True)
    # 索引只包含本次运行已完成的帧, 不扫描旧运行遗留的 VTU.
    pvd_root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    pvd_collection = ET.SubElement(pvd_root, "Collection")
    pvd_tree = ET.ElementTree(pvd_root)
    pvd_temp = output / "evolution.pvd.tmp"
    pvd_tree.write(pvd_temp, encoding="utf-8", xml_declaration=True)
    pvd_temp.replace(output / "evolution.pvd")
    export_mesh = None
    last_vtu = None
    run_started = perf_counter()
    for iteration in range(1, n_analyses + 1):
        started = perf_counter()
        if analysis_only:
            print("[单次分析] 使用输入物理密度, 不再过滤", flush=True)
            physical = input_physical
        else:
            if rho is None:
                raise RuntimeError("设计密度未初始化.")
            print(f"[迭代 {iteration}/{n_analyses}] 密度过滤", flush=True)
            physical = apply_structured_density_filter(density=rho, rmin=radius, spacing=spacing)
        if physical is None:
            raise RuntimeError("物理密度未初始化.")
        local_density = layout.split_global_cell_field(bm.asarray(physical.reshape(-1), **cpu_context))
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
            B = Kib @ psi
            projected = bm.matrix_transpose(psi) @ Kbb @ psi
            print("  刚度分块与迹投影完成", flush=True)
            T = bm.linalg.solve(Kii, -B)
            print("  内部消元完成", flush=True)
            reduced = projected + bm.matrix_transpose(B) @ T
            accumulator.add(start, bm.asarray(reduced, **cpu_context))
            del K, Kii, Kib, Kbb, B, projected, T, reduced
            print(f"  接口刚度散加完成，累计 {end}/{count}，本批 {perf_counter() - stamp:.3f} s", flush=True)
        system = InterfaceSystem(stiffness=accumulator.to_csr(), global_dofs=space.global_dofs)
        assembly_seconds = perf_counter() - started
        stamp = perf_counter()
        print("[接口求解] 开始", flush=True)
        solved = solve_constrained_system(system=system, load=load, constraints=constraints, solver=solver)
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
            ub = q_local @ bm.matrix_transpose(psi)
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
        energy = layout.merge_substructure_cell_field(
            prototype.cell_to_grid_field(energy_local)).reshape(grid)
        if not bm.all(bm.isfinite(energy)) or not bm.all(bm.isfinite(displacement)):
            raise FloatingPointError("恢复位移或单元能量非有限")
        energy_work = float(bm.sum((emin + (1 - emin) * physical**penalty) * energy))
        energy_error = abs(energy_work - compliance) / compliance
        recovery_seconds = perf_counter() - stamp
        dc = None
        if not analysis_only:
            dc_physical = -penalty * (1 - emin) * physical**(penalty - 1) * energy
            dc = apply_structured_density_filter_adjoint(gradient=dc_physical, rmin=radius, spacing=spacing)
            if not bm.all(bm.isfinite(dc)):
                raise FloatingPointError("柔顺度灵敏度非有限")
        symmetry = {}
        fields = [("", physical)]
        if rho is not None:
            fields.append(("design_", rho))
        for prefix, field in fields:
            norm = float(bm.sqrt(bm.sum(field * field)))
            for axis, label in ((0, "x"), (2, "z")):
                difference = field - bm.flip(field, axis=axis)
                symmetry[f"{prefix}symmetry_error_{label}"] = float(bm.max(bm.abs(difference)))
                symmetry[f"{prefix}symmetry_relative_error_{label}"] = (
                    float(bm.sqrt(bm.sum(difference * difference))) / max(norm, 1.0e-30)
                )
        del difference, fields, field
        relative = None if not history else abs(compliance - history[-1]["compliance"]) / compliance
        if relative is not None:
            recent.append(relative)
            recent = recent[-CONVERGENCE_WINDOW:]
        converged = len(recent) == CONVERGENCE_WINDOW and all(value < CONVERGENCE_TOLERANCE for value in recent)
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=float(physical.mean()),
                      relative_change=relative, equilibrium_residual=float(solved.equilibrium_relative_residual),
                      constraint_residual=float(solved.constraint_relative_residual), energy_relative_error=energy_error,
                      assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
                      recovery_seconds=recovery_seconds, analysis_seconds=perf_counter() - started)
        record.update(symmetry)
        # 保存本轮已分析的场, 必须早于 OC 更新及停止判断.
        export_started = perf_counter()
        if export_mesh is None:
            print("[输出] 首次构建全局细网格, 后续迭代复用", flush=True)
            full_mesh = layout.full_mesh
            export_mesh = SimpleNamespace(entity=lambda kind: bm.to_numpy(full_mesh.entity(kind)))
        nodal_u_backend = bm.reshape(displacement, (-1, 3))
        nodal_u = bm.to_numpy(nodal_u_backend)
        u_mag = bm.to_numpy(bm.sqrt(bm.sum(nodal_u_backend * nodal_u_backend, axis=1)))
        cell_data = {"density": bm.to_numpy(physical).reshape(-1)}
        if rho is not None:
            cell_data["design_density"] = bm.to_numpy(rho).reshape(-1)
        relative_vtu = f"iterations/iter_{iteration:04d}.vtu"
        vtu_path = output / relative_vtu
        temporary_vtu = iterations_dir / f"iter_{iteration:04d}.tmp.vtu"
        write_vtu(
            mesh=export_mesh, filepath=str(temporary_vtu.with_suffix("")), cell_data=cell_data,
            point_data={"u_x": nodal_u[:, 0], "u_y": nodal_u[:, 1],
                        "u_z": nodal_u[:, 2], "u_mag": u_mag},
        )
        temporary_vtu.replace(vtu_path)
        # VTU 完整落盘后才发布该帧, 避免读取未完成的文件.
        ET.SubElement(pvd_collection, "DataSet", timestep=str(iteration),
                      group="", part="0", file=relative_vtu)
        pvd_tree.write(pvd_temp, encoding="utf-8", xml_declaration=True)
        pvd_temp.replace(output / "evolution.pvd")
        last_vtu = vtu_path
        record["vtu_file"] = relative_vtu
        record["export_seconds"] = perf_counter() - export_started
        del nodal_u_backend, nodal_u, u_mag, cell_data
        print(f"[输出] 第 {iteration} 轮已保存: {relative_vtu}", flush=True)
        history.append(record)
        print(f"[迭代结果] C={compliance:.10e}，体积分数 {physical.mean():.6f}，相对变化 {relative}，能量相对差 {energy_error:.3e}", flush=True)
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # 停止时保留刚完成分析的密度, 避免最终柔顺度与密度错位.
        if analysis_only or converged or iteration == n_analyses:
            break
        if rho is None or dc is None or volume_gradient is None:
            raise RuntimeError("OC 所需的设计密度或梯度未初始化.")
        print("[OC] 更新设计密度", flush=True)
        updated_density = OCOptimizer.update_design_variable(
            design_variable=rho,
            objective_gradient=dc,
            constraint_gradient=volume_gradient,
            constraint_function=lambda candidate: bm.mean(volume_gradient * candidate) - volume,
            **oc_options,
        )
        rho_new = bm.asarray(updated_density, **compute_context)
        if not bm.all(bm.isfinite(rho_new)) or bm.min(rho_new) < 0 or bm.max(rho_new) > 1:
            raise FloatingPointError("OC 更新得到无效密度")
        candidate_volume = float(bm.mean(volume_gradient * rho_new))
        if candidate_volume > volume + VOLUME_TOLERANCE:
            raise RuntimeError("OC 乘子搜索未满足物理体积约束")
        print(f"[OC] 最大密度变化 {bm.max(bm.abs(rho_new - rho)):.3e}", flush=True)
        rho = rho_new

    if physical is None or displacement is None or not history or last_vtu is None:
        raise RuntimeError("优化未完成任何分析, 无法保存最终结果.")
    if rho is not None:
        np.save(output / "design_density_final.npy", bm.to_numpy(rho))
    np.save(output / "density_final.npy", bm.to_numpy(physical))
    np.save(output / "displacement_final.npy", bm.to_numpy(displacement))
    # 最终文件复用最后一轮已写好的 VTU.
    final_vtu_temp = output / "result_final.tmp.vtu"
    copyfile(last_vtu, final_vtu_temp)
    final_vtu_temp.replace(output / "result_final.vtu")
    summary = dict(history[-1], converged=converged,
                   termination="analysis_only" if analysis_only else ("converged" if converged else "max_iter"),
                   total_seconds=perf_counter() - run_started)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[完成] {summary['termination']}，结果保存到 {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
