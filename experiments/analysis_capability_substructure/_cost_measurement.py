"""二维 FA、full_trace 与 linear_corner 计算成本的独立进程测量协议."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from scipy.linalg import qr
from scipy.sparse import bmat
from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    GlobalAssembler,
    solve_constrained_system,
    solve_interface_system,
)
from examples.substructure_elasticity.verify_linear_corner_consistency import (
    make_fa_analyzer,
)

from experiments.analysis_capability_substructure._density_update import (
    RESIDUAL_TOLERANCE,
    _build_context,
    _conditions,
    _density,
    _free_residual,
    _problem,
    _sha256,
    _validate_retained_conditions,
)
from experiments.analysis_capability_substructure._performance_process import (
    _wait_with_monitor,
    peak_rss_bytes,
    performance_environment,
    timing_statistics,
)


SCHEMA_VERSION = "substructure-route-cost-2d-v2"
ENERGY_REPEAT_TOLERANCE = 1.0e-11
ROUTES = ("fa", "full_trace", "linear_corner")


def cost_measurement_config(route: str | None = None) -> dict[str, Any]:
    """返回写入运行配置的固定测量与验收口径."""
    return {
        "route": route,
        "density": "pattern_a",
        "execution_mode": "fresh_subprocess_per_sample_serial",
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "energy_repeat_tolerance": ENERGY_REPEAT_TOLERANCE,
        "timing_scope": {
            "preparation": (
                "问题、网格、映射、载荷、支承与固定 density=pattern_a 的构造，"
                "在分析计时之前单列"
            ),
            "analysis": (
                "所选路径的刚度装配、缩聚与接口装配（如适用）、"
                "边界处理与求解、完整位移恢复"
            ),
            "excluded": "平衡残差、密度及位移校验、环境采集和 JSON 写入",
        },
        "memory_scope": (
            "独立 Worker 从解释器启动至位移恢复完成的 Linux VmHWM；"
            "在平衡残差和证据计算之前读取"
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """原子更新 JSON 记录，保留失败前最后一个完整阶段."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _phase_start(record: dict[str, Any], record_path: Path, phase: str) -> float:
    """在阶段开始前持久化标识，并返回计时起点."""
    record["phase"] = phase
    _write_json(record_path, record)
    return time.perf_counter()


def _validate_seconds(phases: dict[str, float]) -> None:
    """检查阶段计时均为有限非负数且总时间为正."""
    if any(not math.isfinite(value) or value < 0.0 for value in phases.values()):
        raise AssertionError("阶段计时包含非有限值或负值.")
    if phases.get("analysis_total", 0.0) <= 0.0:
        raise AssertionError("总分析时间必须为正数.")


def _physical_data(
    context: dict[str, Any], density: Any, force: Any, fixed: Any
) -> dict[str, Any]:
    """记录跨路径对照所需的物理设置与输入指纹."""
    pde = context["pde"]
    density_np = np.asarray(bm.to_numpy(density), dtype=np.float64)
    force_np = np.asarray(bm.to_numpy(force), dtype=np.float64).reshape(-1)
    fixed_np = np.asarray(bm.to_numpy(fixed), dtype=np.int64)
    return {
        "problem": type(pde).__name__,
        "domain": [float(value) for value in pde.domain],
        "material": {
            "E": float(pde.E),
            "nu": float(pde.nu),
            "hypothesis": "plane_stress",
        },
        "simp": {"penalty": 3.0, "rho_min": 0.0},
        "load": {
            "P": float(pde.P),
            "resultant": force_np.reshape(-1, 2).sum(axis=0).tolist(),
            "sha256": _sha256(force_np),
        },
        "boundary": {
            "fixed_dofs": int(len(fixed_np)),
            "fixed_dofs_sha256": _sha256(fixed_np),
        },
        "density": {
            "name": "pattern_a",
            "minimum": float(density_np.min()),
            "maximum": float(density_np.max()),
            "mean": float(density_np.mean()),
            "sha256": _sha256(density_np),
        },
    }


def _worker(request_path: Path, record_path: Path) -> int:
    """执行一次全新进程测量，并在异常时保留阶段与回溯."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    route = str(request["route"])
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "phase": "request_loaded",
        "pid": os.getpid(),
        "sample_kind": request["sample_kind"],
        "sample_index": request["sample_index"],
        "route": route,
        "density_name": "pattern_a",
        "solve_method": request["solve_method"],
        "seconds": {},
    }
    _write_json(record_path, record)

    try:
        if request["dim"] != 2:
            raise ValueError("计算成本工况当前仅支持二维.")
        if route not in ROUTES:
            raise ValueError(f"未知分析路径: {route}")
        if request["solve_method"] != "mumps":
            raise ValueError("计算成本工况必须使用 MUMPS.")

        tick = _phase_start(record, record_path, "problem_preparation")
        n_sub = tuple(int(value) for value in request["n_sub"])
        n_fine = tuple(int(value) for value in request["n_fine"])
        total_fine = tuple(int(a * b) for a, b in zip(n_sub, n_fine))
        density = _density(total_fine, n_fine, "pattern_a")

        if route == "fa":
            pde = _problem(2)
            domain = tuple(
                pde.domain[2 * index + 1] - pde.domain[2 * index]
                for index in range(2)
            )
            assembler = GlobalAssembler(
                domain, n_sub, n_fine, E_base=pde.E, nu=pde.nu
            )
            context = {"pde": pde, "assembler": assembler}
            analyzer = make_fa_analyzer(
                assembler.full_mesh,
                pde,
                assembler.material,
                solve_method=request["solve_method"],
            )
            force = analyzer.assemble_external_load()
            prescribed, fixed_mask = analyzer.tensor_space.boundary_interpolate(
                gd=pde.dirichlet_bc,
                threshold=pde.is_dirichlet_boundary(),
                method="interp",
            )
            if np.any(np.asarray(bm.to_numpy(prescribed))):
                raise ValueError("计算成本工况仅支持齐次 Dirichlet 约束.")
            fixed = bm.nonzero(fixed_mask)[0]
        else:
            context = _build_context(2, n_sub, n_fine, route)
            force, fixed = _conditions(context, request["solve_method"])
            _validate_retained_conditions(context, force, fixed)
            assembler = context["assembler"]
        preparation_seconds = time.perf_counter() - tick

        phases: dict[str, float] = {}
        residual = None
        constraint_residual = None
        constraint_matrix = None
        if route == "fa":
            tick = _phase_start(record, record_path, "global_stiffness_assembly")
            from experiments.analysis_capability_substructure._fa_chunked import (
                DEFAULT_CHUNK_SIZE, assemble_chunked_fa,
            )
            stiffness, fa_assembly = assemble_chunked_fa(
                analyzer, density, domain_size=assembler.domain_size,
                total_fine=total_fine, chunk_size=DEFAULT_CHUNK_SIZE,
            )
            record["fa_assembly"] = fa_assembly
            phases["global_stiffness_assembly"] = time.perf_counter() - tick
            record["seconds"] = dict(phases)
            system = SimpleNamespace(
                stiffness=stiffness,
                global_dofs=bm.arange(
                    assembler.total_full_dofs, dtype=bm.int64
                ),
            )
            tick = _phase_start(record, record_path, "conditions_solve")
            displacement = solve_interface_system(
                system, force, fixed, solver=request["solve_method"]
            )
            phases["conditions_solve"] = time.perf_counter() - tick
            record["seconds"] = dict(phases)
            system_dofs = assembler.total_full_dofs
            free_dofs = system_dofs - len(fixed)
        else:
            prototype = context["prototype"]
            sub_meshes = context["sub_meshes"]
            condensor = context["condensor"]
            tick = _phase_start(record, record_path, "local_assembly")
            local_density = assembler.split_global_cell_field(density)
            local_stiffness = prototype.assemble_local_stiffness_batch(
                local_density
            )
            phases["local_assembly"] = time.perf_counter() - tick
            record["seconds"] = dict(phases)

            tick = _phase_start(record, record_path, "condensation")
            condensed, _ = condensor.condense(local_stiffness)
            phases["condensation"] = time.perf_counter() - tick
            record["seconds"] = dict(phases)
            del local_stiffness, local_density

            if route == "full_trace":
                pattern_seconds = 0.0
                pattern_calls = 0
                original_prepare = assembler._prepare_interface_pattern

                def measured_prepare(*args: Any, **kwargs: Any) -> Any:
                    nonlocal pattern_seconds, pattern_calls
                    pattern_calls += 1
                    started = time.perf_counter()
                    try:
                        return original_prepare(*args, **kwargs)
                    finally:
                        pattern_seconds += time.perf_counter() - started

                tick = _phase_start(record, record_path, "interface_assembly")
                assembler._prepare_interface_pattern = measured_prepare
                try:
                    system = assembler.assemble_trace_system(
                        sub_meshes,
                        condensor,
                        trace_basis=context["trace"],
                    )
                finally:
                    assembler._prepare_interface_pattern = original_prepare
                interface_total = time.perf_counter() - tick
                if pattern_calls != 1 or pattern_seconds <= 0.0:
                    raise AssertionError(
                        "首次接口装配必须且只能调用一次 pattern 准备."
                    )
                phases["pattern_prepare"] = pattern_seconds
                phases["interface_mapping_and_numeric_assembly"] = max(
                    0.0, interface_total - pattern_seconds
                )
                record["seconds"] = dict(phases)
                tick = _phase_start(record, record_path, "conditions_solve")
                interface_force = assembler.project_global_vector(system, force)
                interface_fixed = assembler.project_global_dofs(system, fixed)
                interface_u = solve_interface_system(
                    system,
                    interface_force,
                    interface_fixed,
                    solver=request["solve_method"],
                )
                phases["conditions_solve"] = time.perf_counter() - tick
                record["seconds"] = dict(phases)
                full_view = system
                system_dofs = len(system.global_dofs)
                free_dofs = system_dofs - len(interface_fixed)
            else:
                trace = context["trace"]
                projection = context["projection"]
                full_view = context["interface_view"]
                tick = _phase_start(record, record_path, "macro_assembly")
                system = assembler.assemble_trace_system(
                    sub_meshes, condensor, trace_basis=trace
                )
                phases["macro_assembly"] = time.perf_counter() - tick
                record["seconds"] = dict(phases)
                tick = _phase_start(record, record_path, "conditions_solve")
                full_force = np.asarray(
                    bm.to_numpy(
                        assembler.project_global_vector(full_view, force)
                    )
                )
                full_fixed = np.asarray(
                    bm.to_numpy(
                        assembler.project_global_dofs(full_view, fixed)
                    ),
                    dtype=np.int64,
                )
                macro_force = projection.T @ full_force
                constraint_matrix = projection[full_fixed].tocsr()
                active = np.unique(constraint_matrix.indices)
                dense_active = constraint_matrix[:, active].toarray()
                _, triangular, pivots = qr(
                    dense_active.T, mode="economic", pivoting=True
                )
                diagonal = np.abs(np.diag(triangular))
                threshold = (
                    max(constraint_matrix.shape[0], len(active))
                    * np.finfo(float).eps
                    * (diagonal.max() if diagonal.size else 0.0)
                )
                constraint_rank = int(
                    np.count_nonzero(diagonal > threshold)
                )
                independent_rows = np.asarray(
                    pivots[:constraint_rank], dtype=np.int64
                )
                independent = constraint_matrix[independent_rows]
                stiffness = system.stiffness.to_scipy().tocsr()
                saddle = bmat(
                    [[stiffness, independent.T], [independent, None]],
                    format="csc",
                )
                rhs = np.concatenate((
                    np.asarray(macro_force, dtype=np.float64),
                    np.zeros(constraint_rank, dtype=np.float64),
                ))
                from soptx.solvers import create

                linear_solver = create(request["solve_method"])
                try:
                    solved_all, _ = linear_solver.setup(saddle).solve(rhs)
                finally:
                    linear_solver.close()
                solved_np = np.asarray(
                    bm.to_numpy(solved_all), dtype=np.float64
                )
                macro_u = bm.asarray(
                    solved_np[:len(system.global_dofs)], dtype=bm.float64
                )
                multipliers = solved_np[len(system.global_dofs):]
                phases["conditions_solve"] = time.perf_counter() - tick
                record["seconds"] = dict(phases)
                system_dofs = len(system.global_dofs)
                free_dofs = system_dofs - constraint_rank

            tick = _phase_start(record, record_path, "recovery")
            if route == "linear_corner":
                interface_u = bm.asarray(
                    projection @ bm.to_numpy(macro_u), dtype=bm.float64
                )
            displacement = assembler.recover_full_displacement(
                sub_meshes, condensor, full_view, interface_u
            )
            phases["recovery"] = time.perf_counter() - tick
            record["seconds"] = dict(phases)

        phases["analysis_total"] = sum(phases.values())
        _validate_seconds(phases)

        measured_peak_rss = peak_rss_bytes()
        record["seconds"] = dict(phases)
        record.update(
            phase="validation",
            preparation_seconds=preparation_seconds,
            seconds=phases,
            memory_peak_rss_bytes=measured_peak_rss,
        )
        _write_json(record_path, record)

        if route == "fa":
            residual = _free_residual(system, displacement, force, fixed)
        elif route == "full_trace":
            residual = _free_residual(
                system, interface_u, interface_force, interface_fixed
            )
        else:
            macro_u_np = np.asarray(
                bm.to_numpy(macro_u), dtype=np.float64
            )
            stiffness = system.stiffness.to_scipy().tocsr()
            internal_force = stiffness @ macro_u_np
            reaction = independent.T @ multipliers
            equilibrium = internal_force + reaction - macro_force
            equilibrium_scale = max(
                float(np.linalg.norm(macro_force)),
                float(np.linalg.norm(internal_force)),
                float(np.linalg.norm(reaction)),
                np.finfo(float).tiny,
            )
            residual = float(
                np.linalg.norm(equilibrium)
            ) / equilibrium_scale
            constraint_scale = max(
                float(np.linalg.norm(macro_u_np)), np.finfo(float).tiny
            )
            constraint_residual = float(
                np.linalg.norm(constraint_matrix @ macro_u_np)
            ) / constraint_scale
        physical = _physical_data(context, density, force, fixed)
        displacement_np = np.asarray(
            bm.to_numpy(displacement), dtype=np.float64
        ).reshape(-1)
        displacement_hash = _sha256(displacement_np)
        energy = 0.5 * float(bm.dot(force, displacement))
        support = np.asarray(bm.to_numpy(fixed), dtype=np.int64)
        support_error = float(
            np.linalg.norm(displacement_np[support])
        ) / max(
            float(np.linalg.norm(displacement_np)), np.finfo(float).tiny
        )
        if (
            not np.all(np.isfinite(displacement_np))
            or not math.isfinite(energy)
            or energy <= 0.0
        ):
            raise AssertionError("位移或应变能不是有限有效值.")
        if not math.isfinite(residual) or residual > RESIDUAL_TOLERANCE:
            raise AssertionError(
                f"相对平衡残差 {residual:.4e} 超过 "
                f"{RESIDUAL_TOLERANCE:.1e}."
            )
        if (
            constraint_residual is not None
            and (
                not math.isfinite(constraint_residual)
                or constraint_residual > RESIDUAL_TOLERANCE
            )
        ):
            raise AssertionError(
                f"线性约束残差 {constraint_residual:.4e} 超过 "
                f"{RESIDUAL_TOLERANCE:.1e}."
            )
        if not math.isfinite(support_error) or support_error > RESIDUAL_TOLERANCE:
            raise AssertionError(
                f"支承相对误差 {support_error:.4e} 超过 "
                f"{RESIDUAL_TOLERANCE:.1e}."
            )

        artifact = None
        if request.get("save_displacement", False):
            artifact_path = record_path.parent / "displacement.npy"
            np.save(artifact_path, displacement_np, allow_pickle=False)
            artifact = {
                "path": artifact_path.name,
                "shape": list(displacement_np.shape),
                "dtype": str(displacement_np.dtype),
                "sha256": displacement_hash,
            }

        record.update(
            status="PASS",
            phase="complete",
            preparation_seconds=preparation_seconds,
            seconds=phases,
            memory_peak_rss_bytes=measured_peak_rss,
            physical_data=physical,
            problem_data={
                "problem": type(context["pde"]).__name__,
                "dimension": 2,
                "degree": 1,
                "n_sub": list(n_sub),
                "n_fine": list(n_fine),
                "global_fine_grid": list(total_fine),
                "cell_count": int(np.prod(total_fine)),
                "full_dofs": int(assembler.total_full_dofs),
                "system_dofs": int(system_dofs),
                "free_dofs": int(free_dofs),
            },
            validation={
                "equilibrium": "PASS",
                "support": "PASS",
                "constraint": (
                    "NOT_APPLICABLE"
                    if constraint_residual is None
                    else "PASS"
                ),
                "equilibrium_relative_residual": residual,
                "constraint_relative_residual": constraint_residual,
                "support_relative_error": support_error,
                "strain_energy": energy,
                "density_sha256": _sha256(density),
                "displacement_sha256": displacement_hash,
            },
            displacement_artifact=artifact,
            environment=performance_environment(),
        )
        _write_json(record_path, record)
        return 0
    except BaseException as error:
        record.update(
            status="FAILED",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        try:
            _write_json(record_path, record)
        except BaseException:
            pass
        return 1

def _worker_run(
    request: dict[str, Any],
    directory: Path,
    *,
    monitor: bool,
    monitor_interval: float,
    label: str,
) -> dict[str, Any]:
    """启动一个全新 Worker，并保留其请求、结果和标准输出."""
    directory.mkdir(parents=True, exist_ok=False)
    request_path = directory / "request.json"
    record_path = directory / "record.json"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    _write_json(request_path, request)
    command = [
        sys.executable,
        "-m",
        "experiments.analysis_capability_substructure._cost_measurement",
        "--worker",
        str(request_path.resolve()),
        str(record_path.resolve()),
    ]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, \
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
        )
        try:
            if monitor:
                returncode, stats = _wait_with_monitor(
                    process, label, started, monitor_interval
                )
            else:
                process.wait()
                returncode = int(process.returncode or 0)
                stats = None
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    elapsed = time.perf_counter() - started
    record = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path.is_file()
        else {
            "status": "FAILED",
            "phase": "worker_no_record",
            "error": {"type": "MissingRecord", "message": "Worker 未生成记录."},
        }
    )
    record["worker_elapsed_seconds"] = elapsed
    record["worker_returncode"] = returncode
    if returncode != 0 and record.get("status") == "RUNNING":
        record["status"] = "FAILED"
        record["error"] = {
            "type": "WorkerExit",
            "message": f"Worker 在 {record.get('phase')} 阶段退出: {returncode}",
        }
    if stats is not None:
        record["monitor_observed"] = asdict(stats)
    _write_json(record_path, record)
    if returncode != 0 or record.get("status") != "PASS":
        stderr_tail = stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )[-8000:]
        raise RuntimeError(
            f"{label} Worker 失败，exit code={returncode}，"
            f"阶段={record.get('phase')}；证据: {record_path}\n{stderr_tail}"
        )
    return record


def _result_name(
    route: str, n_sub: Sequence[int], n_fine: Sequence[int]
) -> str:
    sub = "x".join(str(value) for value in n_sub)
    fine = "x".join(str(value) for value in n_fine)
    return f"{route}_cost_2d_sub-{sub}_fine-{fine}.json"


def run_route_cost_2d(
    output_dir: str,
    *,
    route: str,
    n_sub: Sequence[int],
    n_fine: Sequence[int],
    warmup: int,
    repeat: int,
    solve_method: str,
    monitor: bool = False,
    monitor_interval: float = 0.5,
    expected_displacement_sha256: str | None = None,
    expected_strain_energy: float | None = None,
) -> dict[str, Any]:
    """按独立进程试运行和正式测量单条二维分析路径."""
    if route not in ROUTES:
        raise ValueError(f"未知分析路径: {route}")
    if solve_method != "mumps":
        raise ValueError("二维计算成本工况必须使用 MUMPS.")
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup 必须非负且 repeat 必须为正整数.")
    peak_rss_bytes()

    output = Path(output_dir)
    result_path = output / _result_name(route, n_sub, n_fine)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "configuration": {
            **cost_measurement_config(route),
            "n_sub": list(n_sub),
            "n_fine": list(n_fine),
            "solve_method": solve_method,
            "warmup": warmup,
            "repeat": repeat,
            "monitor": monitor,
            "monitor_interval": monitor_interval,
            "expected_displacement_sha256": expected_displacement_sha256,
            "expected_strain_energy": expected_strain_energy,
        },
        "warmup_samples": [],
        "samples": [],
    }
    _write_json(result_path, result)

    request_base = {
        "dim": 2,
        "route": route,
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "solve_method": solve_method,
    }
    try:
        for ordinal in range(warmup + repeat):
            is_warmup = ordinal < warmup
            sample_index = ordinal + 1 if is_warmup else ordinal - warmup + 1
            kind = "warmup" if is_warmup else "measurement"
            label = (
                f"试运行 {sample_index}/{warmup} [{route}]"
                if is_warmup
                else f"正式测量 {sample_index}/{repeat} [{route}]"
            )
            print(f"=== {label} ===", flush=True)
            sample = _worker_run(
                {
                    **request_base,
                    "sample_kind": kind,
                    "sample_index": sample_index,
                    "save_displacement": (not is_warmup and sample_index == 1),
                },
                output / f"{kind}_{sample_index:02d}",
                monitor=monitor,
                monitor_interval=monitor_interval,
                label=label,
            )
            key = "warmup_samples" if is_warmup else "samples"
            result[key].append(sample)
            _write_json(result_path, result)
            print(f"--- {label} PASS ---\n", flush=True)

        samples = result["samples"]
        density_hashes = {
            row["validation"]["density_sha256"] for row in samples
        }
        displacement_hashes = {
            row["validation"]["displacement_sha256"] for row in samples
        }
        energies = [row["validation"]["strain_energy"] for row in samples]
        energy_reference = energies[0]
        maximum_energy_difference = max(
            abs(value - energy_reference) / abs(energy_reference)
            for value in energies
        )
        if len(density_hashes) != 1:
            raise AssertionError("正式测量使用的 pattern_a 密度指纹不一致.")
        if len(displacement_hashes) != 1:
            raise AssertionError("正式测量的位移指纹不一致.")
        if maximum_energy_difference > ENERGY_REPEAT_TOLERANCE:
            raise AssertionError(
                f"正式测量应变能相对差 {maximum_energy_difference:.4e} 超过 "
                f"{ENERGY_REPEAT_TOLERANCE:.1e}."
            )
        if (
            expected_displacement_sha256 is not None
            and next(iter(displacement_hashes))
            != expected_displacement_sha256
        ):
            raise AssertionError("位移指纹与指定参考结果不一致.")
        expected_energy_error = None
        if expected_strain_energy is not None:
            expected_energy_error = abs(
                energy_reference - expected_strain_energy
            ) / abs(expected_strain_energy)
            if expected_energy_error > ENERGY_REPEAT_TOLERANCE:
                raise AssertionError(
                    f"应变能相对参考差 {expected_energy_error:.4e} 超过 "
                    f"{ENERGY_REPEAT_TOLERANCE:.1e}."
                )

        phase_names = tuple(samples[0]["seconds"])
        result.update(
            status="PASS",
            physical_data=samples[0]["physical_data"],
            problem_data=samples[0]["problem_data"],
            environment=samples[0]["environment"],
            representative_displacement=samples[0]["displacement_artifact"],
            statistics_seconds={
                name: timing_statistics(
                    [row["seconds"][name] for row in samples]
                )
                for name in phase_names
            },
            statistics_preparation_seconds=timing_statistics(
                [row["preparation_seconds"] for row in samples]
            ),
            statistics_peak_rss_bytes=timing_statistics(
                [row["memory_peak_rss_bytes"] for row in samples]
            ),
            validation={
                "equilibrium": "PASS",
                "support": "PASS",
                "constraint": (
                    "PASS" if route == "linear_corner" else "NOT_APPLICABLE"
                ),
                "density_sha256_consistency": "PASS",
                "displacement_sha256_consistency": "PASS",
                "strain_energy_repeat_consistency": "PASS",
                "density_sha256": next(iter(density_hashes)),
                "displacement_sha256": next(iter(displacement_hashes)),
                "strain_energy": energy_reference,
                "maximum_equilibrium_relative_residual": max(
                    row["validation"]["equilibrium_relative_residual"]
                    for row in samples
                ),
                "maximum_constraint_relative_residual": max(
                    (
                        row["validation"]["constraint_relative_residual"]
                        for row in samples
                        if row["validation"]["constraint_relative_residual"]
                        is not None
                    ),
                    default=None,
                ),
                "maximum_support_relative_error": max(
                    row["validation"]["support_relative_error"]
                    for row in samples
                ),
                "maximum_strain_energy_relative_difference": (
                    maximum_energy_difference
                ),
                "expected_displacement_sha256": (
                    "NOT_REQUESTED"
                    if expected_displacement_sha256 is None
                    else "PASS"
                ),
                "expected_strain_energy": (
                    "NOT_REQUESTED"
                    if expected_strain_energy is None
                    else "PASS"
                ),
                "expected_strain_energy_relative_error": expected_energy_error,
            },
        )
        _write_json(result_path, result)
    except BaseException as error:
        result.update(
            status="FAILED",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        _write_json(result_path, result)
        raise

    print(
        f"PASS：{route} 的平衡、支承、密度、位移和应变能检查通过."
    )
    print(f"结果：{result_path}")
    return result


def _main(arguments: Sequence[str]) -> int:
    if len(arguments) != 3 or arguments[0] != "--worker":
        raise ValueError("本模块仅供 run.py 启动内部 Worker.")
    return _worker(Path(arguments[1]), Path(arguments[2]))


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))