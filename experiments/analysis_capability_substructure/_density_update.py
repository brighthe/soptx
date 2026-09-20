"""验证变密度子结构分析中复用几何对象与重新构建的一致性.

Notes
-----
本目录的静力缩聚只作用于刚度, 不缩聚载荷, 恢复关系固定为 u_i^j = N^j u_b^j. 该设定
与 Huang 2023 式 (6) 假设内部自由度不受外载一致, 不是实现缺口. 由此集中载荷与面载荷
必须作用在接口自由度上; 体力 (自重, 热载) 使 f_i^j != 0, 需补缩聚载荷
f_s^j = f_b^j - K_bi^j (K_ii^j)^{-1} f_i^j 与恢复式中的 (K_ii^j)^{-1} f_i^j, 论文与
本实现均未覆盖. 本模块的 _build_context 与 _conditions 据此构造载荷和支承,
_cost_measurement.py 与 _corner_convergence.py 共用同一前提.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Sequence, cast

import numpy as np
from scipy.sparse import csr_matrix
from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    FEAStaticCondensation,
    FullTraceBasis,
    GlobalAssembler,
    LinearCornerTraceBasis,
    build_substructures,
    solve_constrained_system,
    solve_interface_system,
)
from soptx.problems.elasticity import CantileverCorner2d, FullMBBBeam3d
from examples.substructure_elasticity.verify_linear_corner_consistency import (
    build_corner_projection,
    make_fa_analyzer,
)
from experiments.analysis_capability_substructure._performance_process import (
    _wait_with_monitor,
    peak_rss_bytes,
)

DENSITY_SEQUENCE = ("uniform", "pattern_a", "pattern_b", "uniform")
RELATIVE_TOLERANCE = 1.0e-11
RESIDUAL_TOLERANCE = 1.0e-9
_CHUNK_BYTES = 64 * 2**20


def density_update_config() -> dict[str, Any]:
    """返回写入运行证据的固定密度序列与验收参数."""
    return {
        "sequence": list(DENSITY_SEQUENCE),
        "uniform_value": 0.7,
        "pattern_amplitude": 0.2,
        "density_bounds": [0.5, 0.9],
        "relative_tolerance": RELATIVE_TOLERANCE,
        "residual_tolerance": RESIDUAL_TOLERANCE,
    }


def _problem(dim: int) -> Any:
    if dim == 2:
        return CantileverCorner2d(
            domain=(0.0, 2.0, 0.0, 1.0), P=-1.0, E=1.0, nu=0.3,
        )
    if dim == 3:
        return FullMBBBeam3d(
            domain=(0.0, 6.0, 0.0, 1.0, 0.0, 1.0),
            P=-1.0, E=1.0, nu=0.3,
        )
    raise ValueError("dim 必须为 2 或 3.")


def _density(shape: Sequence[int], n_fine: Sequence[int], name: str) -> np.ndarray:
    """生成能改变块内相对分布的确定性密度场."""
    if name == "uniform":
        return np.full(tuple(shape), 0.7, dtype=np.float64)
    local_axes = [
        ((np.arange(count, dtype=np.float64) % fine) + 0.5) / fine
        for count, fine in zip(shape, n_fine)
    ]
    global_axes = [
        (np.arange(count, dtype=np.float64) + 0.5) / count
        for count in shape
    ]
    local = [
        axis.reshape((1,) * index + (-1,) + (1,) * (len(shape) - index - 1))
        for index, axis in enumerate(local_axes)
    ]
    global_ = [
        axis.reshape((1,) * index + (-1,) + (1,) * (len(shape) - index - 1))
        for index, axis in enumerate(global_axes)
    ]
    if name == "pattern_a":
        modulation = (
            np.sin(2.0 * np.pi * local[0])
            * np.cos(np.pi * local[1])
            * (0.75 + 0.25 * np.sin(2.0 * np.pi * global_[0]))
        )
        if len(shape) == 3:
            modulation = modulation * np.cos(np.pi * local[2])
    elif name == "pattern_b":
        modulation = (
            np.cos(np.pi * local[0])
            * np.sin(2.0 * np.pi * local[1])
            * (0.75 + 0.25 * np.cos(2.0 * np.pi * global_[1]))
        )
        if len(shape) == 3:
            modulation = modulation * np.sin(2.0 * np.pi * local[2])
    else:
        raise ValueError(f"未知密度场: {name}")
    return np.asarray(0.7 + 0.2 * modulation, dtype=np.float64, order="C")


def _sha256(array: Any) -> str:
    value = np.asarray(bm.to_numpy(array))
    digest = hashlib.sha256()
    flat = value.reshape(-1)
    count = max(1, _CHUNK_BYTES // value.dtype.itemsize)
    for start in range(0, flat.size, count):
        block = np.ascontiguousarray(flat[start:start + count])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _save_dense(path: Path, array: Any) -> dict[str, Any]:
    value = np.asarray(bm.to_numpy(array))
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=value.dtype, shape=value.shape
    )
    source, saved = value.reshape(-1), target.reshape(-1)
    count = max(1, _CHUNK_BYTES // value.dtype.itemsize)
    for start in range(0, source.size, count):
        saved[start:start + count] = source[start:start + count]
    target.flush()
    del target
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha256(value),
    }


def _compare_dense(path: Path, array: Any) -> float:
    reference = np.asarray(bm.to_numpy(array))
    saved = np.load(path, mmap_mode="r", allow_pickle=False)
    if saved.shape != reference.shape or saved.dtype != reference.dtype:
        return 1.0
    left, right = saved.reshape(-1), reference.reshape(-1)
    count = max(1, _CHUNK_BYTES // reference.dtype.itemsize)
    numerator = denominator = 0.0
    for start in range(0, right.size, count):
        stop = min(start + count, right.size)
        diff = np.asarray(left[start:stop]) - right[start:stop]
        numerator += float(np.dot(diff, diff))
        denominator += float(np.dot(right[start:stop], right[start:stop]))
    return float(np.sqrt(numerator) / max(
        np.sqrt(denominator), np.finfo(float).tiny
    ))


def _canonical_csr(matrix: Any) -> csr_matrix:
    value = (
        matrix.to_scipy().tocsr()
        if hasattr(matrix, "to_scipy")
        else csr_matrix(matrix)
    )
    value.sum_duplicates()
    value.sort_indices()
    return value


def _save_sparse(directory: Path, name: str, matrix: Any) -> dict[str, Any]:
    value = _canonical_csr(matrix)
    metadata: dict[str, Any] = {"shape": list(value.shape), "nnz": int(value.nnz)}
    for field in ("indptr", "indices", "data"):
        metadata[field] = _save_dense(
            directory / f"{name}_{field}.npy", getattr(value, field)
        )
    return metadata


def _compare_sparse(directory: Path, name: str, matrix: Any) -> dict[str, Any]:
    value = _canonical_csr(matrix)
    indptr = np.load(
        directory / f"{name}_indptr.npy", mmap_mode="r", allow_pickle=False
    )
    indices = np.load(
        directory / f"{name}_indices.npy", mmap_mode="r", allow_pickle=False
    )
    structure = bool(
        indptr.shape == value.indptr.shape
        and indices.shape == value.indices.shape
        and np.array_equal(indptr, value.indptr)
        and np.array_equal(indices, value.indices)
    )
    return {
        "structure_match": structure,
        "data_relative_error": (
            _compare_dense(directory / f"{name}_data.npy", value.data)
            if structure else 1.0
        ),
    }


def _build_context(
    dim: int,
    n_sub: Sequence[int],
    n_fine: Sequence[int],
    route: str,
) -> dict[str, Any]:
    pde = _problem(dim)
    domain = tuple(
        pde.domain[2 * index + 1] - pde.domain[2 * index]
        for index in range(dim)
    )
    assembler = GlobalAssembler(
        domain, tuple(n_sub), tuple(n_fine), E_base=pde.E, nu=pde.nu
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    context: dict[str, Any] = {
        "pde": pde,
        "assembler": assembler,
        "prototype": prototype,
        "sub_meshes": sub_meshes,
        "condensor": condensor,
    }
    if route in ("full_trace", "linear_corner"):
        trace = (
            FullTraceBasis.from_prototype(prototype)
            if route == "full_trace"
            else LinearCornerTraceBasis.from_prototype(prototype)
        )
        context["trace"] = trace
    if route == "linear_corner":
        interface_dofs = assembler.build_interface_dofs(sub_meshes)
        interface_view = SimpleNamespace(global_dofs=interface_dofs)
        context.update(
            interface_view=interface_view,
            projection=build_corner_projection(
                assembler, sub_meshes, interface_dofs, trace
            ),
        )
    return context


def _conditions(context: dict[str, Any], solve_method: str) -> tuple[Any, Any]:
    assembler, pde = context["assembler"], context["pde"]
    analyzer = make_fa_analyzer(
        assembler.full_mesh, pde, assembler.material, solve_method=solve_method
    )
    force = analyzer.assemble_external_load()
    prescribed, fixed_mask = analyzer.tensor_space.boundary_interpolate(
        gd=pde.dirichlet_bc,
        threshold=cast(Any, pde.is_dirichlet_boundary()),
        method="interp",
    )
    if np.any(np.asarray(bm.to_numpy(prescribed))):
        raise ValueError("密度更新验证仅支持齐次 Dirichlet 约束.")
    fixed = bm.nonzero(fixed_mask)[0]
    return force, fixed


def _validate_retained_conditions(
    context: dict[str, Any],
    force: Any,
    fixed: Any,
) -> None:
    """确认载荷与支承均位于缩聚保留接口上."""
    assembler = context["assembler"]
    interface_dofs = np.asarray(
        bm.to_numpy(
            context.get("interface_view", SimpleNamespace(
                global_dofs=assembler.build_interface_dofs(
                    context["sub_meshes"]
                )
            )).global_dofs
        ),
        dtype=np.int64,
    )
    force_np = np.asarray(bm.to_numpy(force)).reshape(-1)
    active_load = np.flatnonzero(force_np)
    fixed_np = np.asarray(bm.to_numpy(fixed), dtype=np.int64)
    if (
        not np.all(np.isin(active_load, interface_dofs))
        or not np.all(np.isin(fixed_np, interface_dofs))
    ):
        raise ValueError("载荷或支承包含子结构内部自由度，不能执行静力缩聚.")

def _free_residual(system: Any, displacement: Any, force: Any, fixed: Any) -> float:
    matrix = _canonical_csr(system.stiffness)
    value = np.asarray(bm.to_numpy(displacement)).reshape(-1)
    load = np.asarray(bm.to_numpy(force)).reshape(-1)
    fixed_np = np.asarray(bm.to_numpy(fixed), dtype=np.int64)
    free = np.setdiff1d(np.arange(len(value)), fixed_np)
    residual = (matrix @ value - load)[free]
    return float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(load[free])), np.finfo(float).tiny
    )


def _record_dense(
    directory: Path,
    name: str,
    value: Any,
    compare: bool,
    record: dict[str, Any],
) -> None:
    if compare:
        record[name] = _compare_dense(directory / f"{name}.npy", value)
    else:
        record[name] = _save_dense(directory / f"{name}.npy", value)


def _analyze(
    context: dict[str, Any],
    density: np.ndarray,
    route: str,
    force: Any,
    fixed: Any,
    artifact_dir: Path,
    *,
    compare: bool,
    solve_method: str,
) -> dict[str, Any]:
    assembler = context["assembler"]
    prototype = context["prototype"]
    sub_meshes = context["sub_meshes"]
    condensor = context["condensor"]
    comparison: dict[str, Any] = {}
    timings: dict[str, float] = {}
    started = time.perf_counter()

    tick = time.perf_counter()
    local_density = assembler.split_global_cell_field(density)
    local_stiffness = prototype.assemble_local_stiffness_batch(local_density)
    timings["local_assembly"] = time.perf_counter() - tick
    _record_dense(
        artifact_dir, "local_stiffness", local_stiffness, compare, comparison
    )

    tick = time.perf_counter()
    condensed, recovery = condensor.condense(local_stiffness)
    timings["condensation"] = time.perf_counter() - tick
    _record_dense(
        artifact_dir, "condensed_stiffness", condensed, compare, comparison
    )
    _record_dense(
        artifact_dir, "recovery_matrix", recovery, compare, comparison
    )
    del local_stiffness, local_density

    tick = time.perf_counter()
    if route == "full_trace":
        system = assembler.assemble_trace_system(
            sub_meshes, condensor, trace_basis=context["trace"]
        )
        interface_force = assembler.project_global_vector(system, force)
        interface_fixed = assembler.project_global_dofs(system, fixed)
        interface_u = solve_interface_system(
            system, interface_force, interface_fixed, solver=solve_method
        )
        residual = _free_residual(
            system, interface_u, interface_force, interface_fixed
        )
        full_view = system
    elif route == "linear_corner":
        trace = context["trace"]
        projection = context["projection"]
        full_view = context["interface_view"]
        system = assembler.assemble_trace_system(
            sub_meshes, condensor, trace_basis=trace
        )
        full_force = np.asarray(
            bm.to_numpy(assembler.project_global_vector(full_view, force))
        )
        full_fixed = np.asarray(
            bm.to_numpy(assembler.project_global_dofs(full_view, fixed)),
            dtype=np.int64,
        )
        macro_force = projection.T @ full_force
        constraints = projection[full_fixed]
        solved = solve_constrained_system(
            system, macro_force, constraints, solver=solve_method
        )
        interface_u = bm.asarray(
            projection @ bm.to_numpy(solved.displacement), dtype=bm.float64
        )
        corner_residuals = (
            solved.equilibrium_relative_residual,
            solved.constraint_relative_residual,
        )
        residual = (
            max(corner_residuals)
            if all(np.isfinite(value) for value in corner_residuals)
            else float("nan")
        )
    else:
        raise ValueError(f"未知接口: {route}")
    timings["interface_assembly_and_solve"] = time.perf_counter() - tick
    comparison["interface_stiffness"] = (
        _compare_sparse(artifact_dir, "interface_stiffness", system.stiffness)
        if compare
        else _save_sparse(artifact_dir, "interface_stiffness", system.stiffness)
    )

    tick = time.perf_counter()
    displacement = assembler.recover_full_displacement(
        sub_meshes, condensor, full_view, interface_u
    )
    timings["recovery"] = time.perf_counter() - tick
    _record_dense(
        artifact_dir, "displacement", displacement, compare, comparison
    )
    energy = 0.5 * float(bm.dot(force, displacement))
    displacement_sha256 = _sha256(displacement)
    timings["total"] = time.perf_counter() - started

    # 每步重新计算全部数值状态；结束后清空 K_s/N，仅复用几何对象和缩聚器实例。
    condensor.K_s = None
    condensor.N = None
    return {
        "comparison": comparison,
        "strain_energy": energy,
        "displacement_sha256": displacement_sha256,
        "residual": residual,
        "seconds": timings,
    }


def _validate(comparison: dict[str, Any], route: str) -> None:
    for name in (
        "local_stiffness",
        "condensed_stiffness",
        "recovery_matrix",
        "displacement",
    ):
        value = comparison[name]
        if not np.isfinite(value) or value > RELATIVE_TOLERANCE:
            raise AssertionError(
                f"{route} {name} 相对差 {value:.4e} 超过 "
                f"{RELATIVE_TOLERANCE:.1e}."
            )
    sparse = comparison["interface_stiffness"]
    if not sparse["structure_match"]:
        raise AssertionError(f"{route} 接口刚度稀疏结构不一致.")
    value = sparse["data_relative_error"]
    if not np.isfinite(value) or value > RELATIVE_TOLERANCE:
        raise AssertionError(
            f"{route} 接口刚度相对差 {value:.4e} 超过 "
            f"{RELATIVE_TOLERANCE:.1e}."
        )



def _write_record(path: Path, result: dict[str, Any]) -> None:
    """原子保存进度; 非有限诊断值写为 null, 验收仍判失败."""
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [sanitize(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _worker(request_path: Path, record_path: Path, scratch: Path) -> int:
    """逐步保存更新与重建对照, 失败时保留已经得到的诊断."""
    bm.set_backend("numpy")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    dim = int(request["dim"])
    n_sub, n_fine = tuple(request["n_sub"]), tuple(request["n_fine"])
    route, solve_method = request["route"], request["solve_method"]
    total_fine = tuple(sub * fine for sub, fine in zip(n_sub, n_fine))
    result: dict[str, Any] = {
        "schema_version": "substructure-density-update-consistency-v1",
        "status": "RUNNING", "passed": False,
        "route": route, "dimension": dim,
        "interface_assembly_method": "pattern" if route == "full_trace" else "coalesce",
        "n_sub": list(n_sub), "n_fine": list(n_fine),
        "total_fine": list(total_fine), "degree": 1, "solver": solve_method,
        "density_update": density_update_config(),
        "strain_energy_formula": "0.5 * dot(full_force, displacement), homogeneous Dirichlet equilibrium",
        "reused_objects": [
            "GlobalAssembler", "SubstructurePrototype", "SubstructureMesh mappings",
            "trace/projection", "FEAStaticCondensation instance",
        ],
        "numeric_state_policy": "每步重新计算数值矩阵; 更新分析结束后清空 K_s/N, 再创建独立重建对照. 不验证旧数值缓存的自动失效.",
        "steps": [], "phase": "build_context",
    }
    _write_record(record_path, result)
    first_displacement = scratch / "initial_displacement.npy"
    first_energy = None
    try:
        persistent = _build_context(dim, n_sub, n_fine, route)
        result.update(
            problem=type(persistent["pde"]).__name__,
            domain=list(persistent["pde"].domain),
            material={
                "E": persistent["pde"].E, "nu": persistent["pde"].nu,
                "penal": persistent["prototype"].penal,
                "rho_min": persistent["prototype"].rho_min,
            },
        )
        force, fixed = _conditions(persistent, solve_method)
        _validate_retained_conditions(persistent, force, fixed)
        initial_density_sha = None
        sequence = DENSITY_SEQUENCE
        for index, name in enumerate(sequence):
            result["phase"] = f"step_{index + 1}_density"
            _write_record(record_path, result)
            step_dir = scratch / f"step_{index}"
            step_dir.mkdir(parents=True, exist_ok=False)
            density = _density(total_fine, n_fine, name)
            step: dict[str, Any] = {
                "index": index + 1, "validation": "RUNNING",
                "density": {
                    "name": name, "minimum": float(density.min()),
                    "maximum": float(density.max()), "mean": float(density.mean()),
                    "sha256": _sha256(density),
                },
            }
            density_path = record_path.parent / f"density_{name}.npy"
            if index < len(DENSITY_SEQUENCE) - 1:
                _save_dense(density_path, density)
            step["density"].update(
                file=density_path.name, shape=list(density.shape), dtype=str(density.dtype),
            )
            result["steps"].append(step)
            result["phase"] = f"step_{index + 1}_update"
            _write_record(record_path, result)
            print(f"[density] {index + 1}/{len(sequence)} {name}: update", flush=True)
            update = _analyze(
                persistent, density, route, force, fixed, step_dir,
                compare=False, solve_method=solve_method,
            )
            step["updated"] = update
            result["phase"] = f"step_{index + 1}_rebuild"
            _write_record(record_path, result)
            if index == 0:
                shutil.copyfile(step_dir / "displacement.npy", first_displacement)
                first_energy = update["strain_energy"]
                initial_density_sha = step["density"]["sha256"]
            elif index == len(DENSITY_SEQUENCE) - 1:
                displacement = np.load(step_dir / "displacement.npy", mmap_mode="r", allow_pickle=False)
                roundtrip = {
                    "density_sha256_match": initial_density_sha == step["density"]["sha256"],
                    "displacement_relative_error": _compare_dense(first_displacement, displacement),
                    "strain_energy_relative_error": abs(update["strain_energy"] - first_energy)
                    / max(abs(first_energy), np.finfo(float).tiny),
                }
                del displacement
                result["roundtrip"] = roundtrip
                _write_record(record_path, result)
            gc.collect()
            print(f"[density] {index + 1}/{len(sequence)} {name}: rebuild", flush=True)
            rebuilt = _build_context(dim, n_sub, n_fine, route)
            reference_force, reference_fixed = _conditions(rebuilt, solve_method)
            _validate_retained_conditions(rebuilt, reference_force, reference_fixed)
            input_match = (
                np.array_equal(bm.to_numpy(force), bm.to_numpy(reference_force))
                and np.array_equal(bm.to_numpy(fixed), bm.to_numpy(reference_fixed))
            )
            step["load_and_support_match"] = bool(input_match)
            if not input_match:
                raise AssertionError("更新与重建的载荷或支承不一致.")
            reference = _analyze(
                rebuilt, density, route, reference_force, reference_fixed, step_dir,
                compare=True, solve_method=solve_method,
            )
            del rebuilt, reference_force, reference_fixed
            gc.collect()
            step["rebuilt"] = reference
            comparison = reference["comparison"]
            comparison["strain_energy"] = abs(update["strain_energy"] - reference["strain_energy"]) / max(
                abs(reference["strain_energy"]), np.finfo(float).tiny,
            )
            step["relative_errors"] = comparison
            result["phase"] = f"step_{index + 1}_validation"
            _write_record(record_path, result)
            _validate(comparison, route)
            scalars = (
                comparison["strain_energy"], update["strain_energy"],
                reference["strain_energy"], update["residual"], reference["residual"],
            )
            if (
                not all(np.isfinite(value) for value in scalars)
                or comparison["strain_energy"] > RELATIVE_TOLERANCE
                or update["residual"] > RESIDUAL_TOLERANCE
                or reference["residual"] > RESIDUAL_TOLERANCE
            ):
                raise AssertionError(f"{route} 第 {index + 1} 次切换的能量或残差验收失败.")
            step["validation"] = "PASS"
            result["memory_peak_rss_bytes"] = peak_rss_bytes()
            _write_record(record_path, result)
            print(f"[density] {index + 1}/{len(sequence)} {name}: PASS", flush=True)
            shutil.rmtree(step_dir)
            del density
            gc.collect()
        roundtrip = result["roundtrip"]
        if not roundtrip["density_sha256_match"] or any(
            not np.isfinite(roundtrip[key]) or roundtrip[key] > RELATIVE_TOLERANCE
            for key in ("displacement_relative_error", "strain_energy_relative_error")
        ):
            raise AssertionError("恢复初始均匀密度后的回切一致性验收失败.")
        result.update(status="PASS", passed=True, phase="complete")
        result["validation"] = {"update_vs_rebuild": "PASS", "uniform_roundtrip": "PASS"}
        result["memory_peak_rss_bytes"] = peak_rss_bytes()
        _write_record(record_path, result)
        return 0
    except BaseException as error:
        result.update(status="FAILED", passed=False, error=f"{type(error).__name__}: {error}")
        if result["steps"] and result["steps"][-1]["validation"] == "RUNNING":
            result["steps"][-1]["validation"] = "FAILED"
        result["memory_peak_rss_bytes"] = peak_rss_bytes()
        _write_record(record_path, result)
        raise


def _result_name(
    route: str,
    dim: int,
    n_sub: Sequence[int],
    n_fine: Sequence[int],
) -> str:
    sub = "x".join(map(str, n_sub))
    fine = "x".join(map(str, n_fine))
    return f"{route}_density_update_{dim}d_sub-{sub}_fine-{fine}.json"


def _log_tail(path: Path, limit: int = 8000) -> str:
    """仅读取日志末尾, 避免失败处理加载整个运行日志."""
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode("utf-8", errors="replace")


def run_density_update_consistency(
    dim: int,
    output_dir: str,
    *,
    n_sub: Sequence[int],
    n_fine: Sequence[int],
    route: str,
    solve_method: str = "scipy",
    monitor: bool = False,
    monitor_interval: float = 0.5,
) -> dict[str, Any]:
    """在独立 Worker 中执行密度更新验证，并保留失败证据."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / _result_name(route, dim, n_sub, n_fine)
    scratch = output / ".density_update_scratch"
    request_path = output / "density_update_request.json"
    worker_record = output / "density_update_worker_result.json"
    stdout_path = output / "worker_stdout.log"
    stderr_path = output / "worker_stderr.log"
    request = {
        "dim": dim,
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "route": route,
        "solve_method": solve_method,
        "monitor_interval": monitor_interval,
        "density_update": density_update_config(),
        "strain_energy_formula": "0.5 * dot(full_force, displacement)",
    }
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    initial = {
        "schema_version": "substructure-density-update-consistency-v1",
        "status": "RUNNING",
        "request": request,
    }
    target.write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scratch.mkdir(exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "experiments.analysis_capability_substructure._density_update",
        "--worker",
        str(request_path.resolve()),
        str(worker_record.resolve()),
        str(scratch.resolve()),
    ]
    observed = None
    started = time.perf_counter()
    try:
        with stdout_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stdout, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
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
                        process,
                        f"density update [{route}]",
                        time.perf_counter(),
                        monitor_interval,
                    )
                    observed = {
                        "samples": stats.samples,
                        "max_rss_bytes": (
                            None if stats.max_rss_kib is None
                            else stats.max_rss_kib * 1024
                        ),
                        "max_vm_hwm_bytes": (
                            None if stats.max_peak_kib is None
                            else stats.max_peak_kib * 1024
                        ),
                        "min_system_available_bytes": (
                            None if stats.min_available_kib is None
                            else stats.min_available_kib * 1024
                        ),
                    }
                else:
                    process.wait()
                    returncode = int(process.returncode or 0)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        stdout_text = _log_tail(stdout_path)
        stderr_text = _log_tail(stderr_path)
        if returncode != 0:
            hint = (
                " (SIGKILL，可能内存不足；需查内核日志确认)"
                if returncode in (-9, 137) else ""
            )
            raise RuntimeError(
                f"density update Worker 失败，exit code={returncode}{hint}.\n"
                f"{stderr_text[-8000:]}\n{stdout_text[-8000:]}"
            )
        result = json.loads(worker_record.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or result.get("route") != route:
            raise RuntimeError("Worker 结果状态或接口类型与请求不一致.")
        result["runtime"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "monitor_enabled": monitor,
            "monitor_interval": monitor_interval,
            "observed": observed,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        }
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"[证据] 密度更新一致性通过: {target}")
        return result
    except BaseException as error:
        partial: dict[str, Any] = {}
        if worker_record.is_file():
            try:
                partial = json.loads(worker_record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        failure = {
            **initial,
            **partial,
            "passed": False,
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
            "monitor_observed": observed,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        }
        target.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _main(arguments: Sequence[str]) -> int:
    if len(arguments) != 4 or arguments[0] != "--worker":
        raise ValueError("本模块仅供 run.py 启动内部 Worker.")
    return _worker(Path(arguments[1]), Path(arguments[2]), Path(arguments[3]))


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
