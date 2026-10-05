"""三维 MBB 梁的 linear_corner 精确子结构拓扑优化入口.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from math import isfinite

# 固定模型与算法参数. 命令行默认值直接定义在 parse_args 中.
DOMAIN = (0.0, 6.0, 0.0, 1.0, 0.0, 1.0)
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs",
                        help="结果根目录, 每次运行创建独立子目录")
    parser.add_argument(
        "--backend", choices=("numpy", "pytorch"), default="numpy",
        help="全流程张量计算后端 (默认: numpy)",
    )
    parser.add_argument(
        "--device", default="cpu", help="密度与消元等数值计算设备: cpu, cuda 或 cuda:N (默认: cpu)",
    )
    args = parser.parse_args(argv)
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
    domain_size = tuple(DOMAIN[2 * d + 1] - DOMAIN[2 * d] for d in range(3))
    grid = tuple(a * b for a, b in zip(n_sub, n_fine))
    spacing = tuple(a / b for a, b in zip(domain_size, grid))
    radius = FILTER_RADIUS_CELLS * spacing[0]
    volume = VOLUME_FRACTION
    emin, penalty = EMIN / E0, SIMP_PENALTY
    oc_options = dict(OC_OPTIONS)
    output = args.output_dir / datetime.now(timezone.utc).strftime("linear_corner_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(trace=TRACE, domain=DOMAIN, backend=args.backend, device=device,
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
    print(f"[配置] {TRACE}, 网格 {grid}, 支承 {SUPPORT}, 计算后端 {args.backend}, 数值设备 {device}, 输出 {output}", flush=True)
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
        domain=DOMAIN, E=E0, nu=NU, P=LOAD,
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
    rho = bm.full(grid, INITIAL_DENSITY, **compute_context)
    # 等体积网格: W^T 1 同时用于物理体积判断及其设计梯度.
    weights = apply_structured_density_filter_adjoint(
        gradient=bm.ones(grid, **compute_context), rmin=radius, spacing=spacing,
    )
    volume_gradient = weights
    physical = None
    displacement = None
    history = []
    recent = []
    converged = False
    run_started = perf_counter()
    for iteration in range(1, args.max_iter + 1):
        started = perf_counter()
        print(f"[迭代 {iteration}/{args.max_iter}] 密度过滤", flush=True)
        physical = apply_structured_density_filter(density=rho, rmin=radius, spacing=spacing)
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
        dc_physical = -penalty * (1 - emin) * physical**(penalty - 1) * energy
        dc = apply_structured_density_filter_adjoint(gradient=dc_physical, rmin=radius, spacing=spacing)
        if not bm.all(bm.isfinite(dc)):
            raise FloatingPointError("柔顺度灵敏度非有限")
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
            constraint_function=lambda candidate: bm.mean(volume_gradient * candidate) - volume,
            **oc_options,
        )
        rho_new = bm.asarray(updated_density, **compute_context)
        if not bm.all(bm.isfinite(rho_new)) or bm.min(rho_new) < 0 or bm.max(rho_new) > 1:
            raise FloatingPointError("OC 更新得到无效密度")
        candidate_volume = float(bm.mean(weights * rho_new))
        if candidate_volume > volume + VOLUME_TOLERANCE:
            raise RuntimeError("OC 乘子搜索未满足物理体积约束")
        print(f"[OC] 最大密度变化 {bm.max(bm.abs(rho_new - rho)):.3e}", flush=True)
        rho = rho_new

    if physical is None or displacement is None or not history:
        raise RuntimeError("优化未完成任何分析, 无法保存最终结果.")
    np.save(output / "design_density_final.npy", bm.to_numpy(rho))
    np.save(output / "density_final.npy", bm.to_numpy(physical))
    np.save(output / "displacement_final.npy", bm.to_numpy(displacement))
    print("[输出] 构建全局细网格并导出 VTU", flush=True)
    nodal_u = bm.to_numpy(displacement).reshape(-1, 3)
    full_mesh = layout.full_mesh
    export_mesh = SimpleNamespace(entity=lambda kind: bm.to_numpy(full_mesh.entity(kind)))
    write_vtu(
        mesh=export_mesh, filepath=str(output / "result_final"),
        cell_data={"density": bm.to_numpy(physical).reshape(-1), "design_density": bm.to_numpy(rho).reshape(-1)},
        point_data={"u_x": nodal_u[:, 0], "u_y": nodal_u[:, 1],
                    "u_z": nodal_u[:, 2], "u_mag": np.linalg.norm(nodal_u, axis=1)},
    )
    summary = dict(history[-1], converged=converged,
                   termination="converged" if converged else "max_iter",
                   total_seconds=perf_counter() - run_started)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[完成] {summary['termination']}，结果保存到 {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
