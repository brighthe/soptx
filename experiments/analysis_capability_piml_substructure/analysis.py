"""独立分量 PIML 网络的固定悬臂结构求解验证."""

from __future__ import annotations

import json
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from fealpy.backend import backend_manager as bm
from soptx.fem.substructure import (
    ExactSchurReduction,
    GlobalAssembler,
    ShapeFunctionCondensation,
    build_substructures,
    solve_constrained_system,
    solve_interface_system,
)


def _json_value(value: Any) -> Any:
    """转换为严格 JSON 值, 不写入 NaN 或 Infinity."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _relative_error(
    value: np.ndarray, reference: np.ndarray
) -> tuple[float | None, str | None]:
    """计算相对误差, 并显式记录零参照范数."""
    difference = float(np.linalg.norm(value - reference))
    scale = float(np.linalg.norm(reference))
    if not np.isfinite(difference) or not np.isfinite(scale):
        raise ValueError("误差评价遇到非有限数组.")
    if scale == 0.0:
        return (
            (0.0, "reference_norm_zero_and_values_equal")
            if difference == 0.0
            else (None, "reference_norm_zero")
        )
    return difference / scale, None


def _predict(model: torch.nn.Module, modulus: np.ndarray, route: str) -> np.ndarray:
    """执行 CPU float64 推理, 返回尚未约束补全的独立矩阵条目."""
    parameters = tuple(model.parameters())
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise ValueError(f"{route} 网络必须位于 CPU.")
    if any(parameter.dtype != torch.float64 for parameter in parameters):
        raise ValueError(f"{route} 网络必须使用 torch.float64.")
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(modulus).to(dtype=torch.float64))
    result = np.asarray(prediction.detach().cpu().numpy(), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != modulus.shape[0]:
        raise ValueError(f"{route} 网络输出形状 {result.shape} 与 batch 不一致.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{route} 网络输出包含非有限值.")
    return result


def _conditions(assembler: GlobalAssembler) -> tuple[np.ndarray, np.ndarray]:
    """构造左端固支、右上角单位向下集中力."""
    nodes = np.asarray(bm.to_numpy(assembler.full_mesh.node), dtype=np.float64)
    tolerance = 100.0 * np.finfo(np.float64).eps * max(
        1.0, max(assembler.domain_size)
    )
    fixed_nodes = np.flatnonzero(
        np.isclose(nodes[:, 0], 0.0, atol=tolerance, rtol=0.0)
    )
    fixed = (
        assembler.dim * fixed_nodes[:, None]
        + np.arange(assembler.dim, dtype=np.int64)[None, :]
    ).reshape(-1)
    target = np.asarray(assembler.domain_size, dtype=np.float64)
    distances = np.linalg.norm(nodes - target[None, :], axis=1)
    load_node = int(np.argmin(distances))
    if distances[load_node] > tolerance:
        raise ValueError("未找到悬臂自由端角节点.")
    load = np.zeros(assembler.total_full_dofs, dtype=np.float64)
    load[assembler.dim * load_node + assembler.dim - 1] = -1.0
    return load, np.asarray(fixed, dtype=np.int64)


def _free_residual(
    system: Any, displacement: np.ndarray, load: np.ndarray, fixed: np.ndarray
) -> float:
    matrix = (
        system.stiffness.to_scipy().tocsr()
        if hasattr(system.stiffness, "to_scipy")
        else system.stiffness.tocsr()
    )
    free = np.setdiff1d(np.arange(len(displacement)), fixed)
    residual = np.asarray(matrix @ displacement - load)[free]
    scale = max(float(np.linalg.norm(load[free])), np.finfo(np.float64).tiny)
    result = float(np.linalg.norm(residual)) / scale
    if not np.isfinite(result):
        raise ValueError("接口求解产生非有限平衡残差.")
    return result


class _StiffnessView:
    """让完整接口装配器消费已在 full_trace 上的刚度."""

    def __init__(self, stiffness: np.ndarray) -> None:
        self.K_s = bm.asarray(stiffness, dtype=bm.float64)

    def recover(self, boundary_displacement: Any) -> Any:
        raise RuntimeError("刚度视图不承担位移恢复.")


def _assemble(
    assembler: GlobalAssembler,
    sub_meshes: Sequence[Any],
    trace_kind: str,
    stiffness: np.ndarray,
) -> Any:
    if trace_kind == "linear_corner":
        return assembler.assemble_macro_system(list(sub_meshes), stiffness)
    return assembler.assemble_interface_system(
        list(sub_meshes), _StiffnessView(stiffness)
    )


def _solve(
    assembler: GlobalAssembler,
    sub_meshes: Sequence[Any],
    trace: Any,
    trace_kind: str,
    system: Any,
    load: np.ndarray,
    fixed: np.ndarray,
) -> dict[str, Any]:
    """在所选迹空间施加同一全局载荷和约束."""
    if trace_kind == "full_trace":
        force = np.asarray(
            bm.to_numpy(assembler.project_global_vector(system, load)),
            dtype=np.float64,
        )
        retained_fixed = np.asarray(
            bm.to_numpy(assembler.project_global_dofs(system, fixed)),
            dtype=np.int64,
        )
        displacement = np.asarray(
            bm.to_numpy(
                solve_interface_system(
                    system,
                    bm.asarray(force, dtype=bm.float64),
                    bm.asarray(retained_fixed, dtype=bm.int64),
                    solver="scipy",
                )
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(displacement)):
            raise ValueError("full_trace 求解产生非有限位移.")
        return {
            "trace": displacement,
            "boundary": displacement,
            "local_indices": np.asarray(
                bm.to_numpy(
                    assembler.interface_indices(sub_meshes, system.global_dofs)
                ),
                dtype=np.int64,
            ),
            "equilibrium_relative_residual": _free_residual(
                system, displacement, force, retained_fixed
            ),
            "constraint_relative_residual": 0.0,
        }

    interface_dofs = assembler.build_interface_dofs(list(sub_meshes))
    interface_view = SimpleNamespace(global_dofs=interface_dofs)
    projection = assembler.build_linear_corner_projection(
        sub_meshes, interface_view, trace
    )
    full_force = np.asarray(
        bm.to_numpy(assembler.project_global_vector(interface_view, load)),
        dtype=np.float64,
    )
    full_fixed = np.asarray(
        bm.to_numpy(assembler.project_global_dofs(interface_view, fixed)),
        dtype=np.int64,
    )
    solved = solve_constrained_system(
        system,
        bm.asarray(projection.T @ full_force, dtype=bm.float64),
        projection[full_fixed],
        solver="scipy",
    )
    displacement = np.asarray(bm.to_numpy(solved.displacement), dtype=np.float64)
    boundary = np.asarray(projection @ displacement, dtype=np.float64)
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(boundary)):
        raise ValueError("linear_corner 求解产生非有限位移.")
    return {
        "trace": displacement,
        "boundary": boundary,
        "local_indices": np.asarray(
            bm.to_numpy(assembler.macro_corner_indices(sub_meshes)),
            dtype=np.int64,
        ),
        "equilibrium_relative_residual": float(
            solved.equilibrium_relative_residual
        ),
        "constraint_relative_residual": float(
            solved.constraint_relative_residual
        ),
    }


def _recover(
    assembler: GlobalAssembler,
    sub_meshes: Sequence[Any],
    positions: Sequence[Sequence[int]],
    trace_matrix: np.ndarray,
    recovery: np.ndarray,
    solution: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """恢复各块内部位移并散射为完整全局位移."""
    q_local = np.asarray(solution["trace"], dtype=np.float64)[
        np.asarray(solution["local_indices"], dtype=np.int64)
    ]
    internal = np.einsum("bij,bj->bi", recovery, q_local)
    boundary = np.einsum("ij,bj->bi", trace_matrix, q_local)
    full = np.full(assembler.total_full_dofs, np.nan, dtype=np.float64)
    i_dofs = np.asarray(bm.to_numpy(sub_meshes[0].i_dofs), dtype=np.int64)
    b_dofs = np.asarray(bm.to_numpy(sub_meshes[0].b_dofs), dtype=np.int64)
    for index, (position, sub_mesh) in enumerate(zip(positions, sub_meshes)):
        global_dofs = np.asarray(
            bm.to_numpy(
                assembler.get_substructure_global_dofs(position, sub_mesh)
            ),
            dtype=np.int64,
        )
        local = np.empty(len(global_dofs), dtype=np.float64)
        local[i_dofs] = internal[index]
        local[b_dofs] = boundary[index]
        assigned = np.isfinite(full[global_dofs])
        if np.any(assigned):
            mismatch = np.abs(full[global_dofs][assigned] - local[assigned])
            if np.any(mismatch > 1.0e-10 * np.maximum(1.0, np.abs(local[assigned]))):
                raise ValueError("相邻子结构在共享接口上的恢复位移不一致.")
        full[global_dofs] = local
    if not np.all(np.isfinite(internal)) or not np.all(np.isfinite(full)):
        raise ValueError("位移恢复产生非有限值或未覆盖全部自由度.")
    return internal, full


def _stiffness_diagnostics(
    stiffness: np.ndarray, deformation_basis: np.ndarray
) -> dict[str, float]:
    reduced = np.einsum(
        "qi,bij,jk->bqk",
        deformation_basis.T,
        stiffness,
        deformation_basis,
    )
    eigenvalues = np.linalg.eigvalsh(reduced)
    minimum = float(np.min(eigenvalues))
    maximum = float(np.max(eigenvalues))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("局部刚度的特征值包含非有限值.")
    return {
        "deformation_minimum_eigenvalue": minimum,
        "deformation_maximum_eigenvalue": maximum,
        "deformation_reciprocal_condition": (
            minimum / maximum if maximum > 0.0 else 0.0
        ),
    }


def run_analysis(
    provider: Any,
    networks: Mapping[str, torch.nn.Module],
    output_dir: Path | str,
    *,
    n_sub: Sequence[int],
    seed: int,
    routes: Sequence[str],
    checkpoint_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """执行同迹空间 exact、shape 与 stiffness 的固定悬臂对照.

    Parameters
    ----------
    provider : IndependentTargetProvider
        定义子结构原型、迹空间以及独立条目补全规则.
    networks : mapping of str to torch.nn.Module
        已加载最佳权重的 CPU float64 网络. stiffness 路线还必须提供 shape
        网络, 用于恢复内部位移.
    output_dir : pathlib.Path or str
        本次分析的全新结果目录, 已存在时拒绝覆盖.
    n_sub : sequence of int
        各方向子结构数量, 长度须等于空间维数.
    seed : int
        材料样本随机种子.
    routes : sequence of str
        待验证路线, 可包含 shape 和 stiffness.
    checkpoint_sources : mapping
        权重路径、摘要等可追溯信息, 原样写入配置快照.

    Returns
    -------
    dict
        与 summary.json 一致的结果摘要. status 为 FAILED 时, 各失败路线保留
        error_type 和 error, 且不执行精确回退.

    Notes
    -----
    低求解残差只说明给定预测刚度系统被正确求解, 不代表代理精度; 代理精度由
    相对 exact 基线的刚度、位移和柔度误差分别给出.
    """
    if bm.backend_name != "numpy":
        raise RuntimeError("结构求解验证要求 FEALPy numpy 后端.")
    metadata = provider.metadata()
    dim = int(metadata["spatial_dimension"])
    n_sub = tuple(int(value) for value in n_sub)
    if len(n_sub) != dim or any(value <= 0 for value in n_sub):
        raise ValueError(f"n_sub 必须包含 {dim} 个正整数.")
    routes = tuple(routes)
    if not routes or any(route not in ("shape", "stiffness") for route in routes):
        raise ValueError("routes 必须是 shape/stiffness 的非空子集.")
    if len(set(routes)) != len(routes):
        raise ValueError("routes 不得重复.")
    required = set(routes) | ({"shape"} if "stiffness" in routes else set())
    missing = sorted(required.difference(networks))
    if missing:
        raise ValueError("缺少结构求解所需网络: " + ", ".join(missing))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    _write_json(
        target / "analysis_config.json",
        {
            "schema_version": "independent-structure-analysis-v1",
            "backend": "numpy",
            "network_device": "cpu",
            "network_dtype": "float64",
            "solver": "scipy",
            "case": "fixed_cantilever_corner_load",
            "seed": int(seed),
            "sampling": {
                "distribution": "independent_uniform",
                "minimum_inclusive": 1.0e-6,
                "maximum_exclusive": 1.0,
            },
            "n_sub": list(n_sub),
            "routes": list(routes),
            "provider": metadata,
            "checkpoint_sources": checkpoint_sources,
        },
    )

    cell_size = tuple(float(value) for value in metadata["cell_size"])
    domain = tuple(cell_size[index] * n_sub[index] for index in range(dim))
    assembler = GlobalAssembler(
        domain,
        n_sub,
        tuple(int(value) for value in metadata["n_fine"]),
        E_base=1.0,
        nu=float(metadata["poisson_ratio"]),
    )
    _, sub_meshes, positions = build_substructures(assembler)
    modulus = np.random.default_rng(seed).uniform(
        1.0e-6,
        1.0,
        size=(prod(n_sub), int(metadata["n_cells"])),
    )
    local_stiffness = np.asarray(
        bm.to_numpy(
            provider.prototype.assemble_local_stiffness_batch(
                bm.asarray(modulus, dtype=bm.float64),
                chunk_size=provider.chunk_size,
            )
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(local_stiffness)):
        raise ValueError("局部刚度包含非有限值.")
    load, fixed = _conditions(assembler)
    np.save(target / "material_modulus.npy", modulus)
    np.save(target / "load.npy", load)
    np.save(target / "fixed_dofs.npy", fixed)

    trace_matrix = np.asarray(bm.to_numpy(provider.trace.matrix), dtype=np.float64)
    exact = ExactSchurReduction(
        provider.prototype.i_dofs, provider.prototype.b_dofs
    ).reduce_many(bm.asarray(local_stiffness, dtype=bm.float64))
    full_stiffness = np.asarray(bm.to_numpy(exact.stiffness), dtype=np.float64)
    full_recovery = np.asarray(bm.to_numpy(exact.recovery), dtype=np.float64)
    exact_stiffness = np.einsum(
        "qi,bij,jk->bqk", trace_matrix.T, full_stiffness, trace_matrix
    )
    exact_recovery = np.einsum(
        "bij,jk->bik", full_recovery, trace_matrix
    )
    rigid, deformation, rigid_interior = (
        provider.prototype.trace_interface_bases(provider.trace)
    )
    deformation = np.asarray(bm.to_numpy(deformation), dtype=np.float64)
    exact_system = _assemble(
        assembler, sub_meshes, provider.trace_kind, exact_stiffness
    )
    exact_solution = _solve(
        assembler,
        sub_meshes,
        provider.trace,
        provider.trace_kind,
        exact_system,
        load,
        fixed,
    )
    exact_internal, exact_full = _recover(
        assembler,
        sub_meshes,
        positions,
        trace_matrix,
        exact_recovery,
        exact_solution,
    )
    exact_compliance = float(load @ exact_full)
    np.save(target / "exact_displacement.npy", exact_full)
    np.save(target / "exact_trace_displacement.npy", exact_solution["trace"])
    np.save(target / "exact_internal_displacement.npy", exact_internal)

    # 两条路线均使用预测的内部延拓恢复内部位移.
    shape_prediction = _predict(
        networks["shape"], modulus, "shape"
    )
    # 根据刚体约束补全内部延拓矩阵.
    shape_recovery = provider.shape_codec.decode(shape_prediction)
    # 将补全结果转换为 float64 NumPy 数组.
    shape_recovery = np.asarray(shape_recovery, dtype=np.float64)

    # 仅构造所选路线的局部缩聚刚度.
    predicted = {}

    if "shape" in routes:
        # 形函数路线: 由预测延拓与细网格刚度进行变分构造.
        builder = ShapeFunctionCondensation(
            provider.prototype.i_dofs,
            provider.prototype.b_dofs,
            rigid_basis=rigid,
            deformation_basis=bm.asarray(deformation, dtype=bm.float64),
            rigid_interior=rigid_interior,
            trace=provider.trace,
        )

        local_stiffness_backend = bm.asarray(
            local_stiffness, dtype=bm.float64
        )
        shape_recovery_backend = bm.asarray(
            shape_recovery, dtype=bm.float64
        )
        shape_stiffness = builder.assemble_reduced_stiffness(
            local_stiffness_backend, shape_recovery_backend
        )
        predicted["shape"] = np.asarray(
            bm.to_numpy(shape_stiffness), dtype=np.float64
        )

    if "stiffness" in routes:
        # 直接刚度路线: 预测独立条目, 再补全缩聚刚度矩阵.
        stiffness_prediction = _predict(
            networks["stiffness"], modulus, "stiffness"
        )
        stiffness_matrix = provider.stiffness_codec.decode(
            stiffness_prediction
        )
        predicted["stiffness"] = np.asarray(
            stiffness_matrix, dtype=np.float64
        )

    summary: dict[str, Any] = {
        "schema_version": "independent-structure-analysis-v1",
        "status": "RUNNING",
        "trace": provider.trace_kind,
        "dimension": dim,
        "n_sub": list(n_sub),
        "n_fine": list(metadata["n_fine"]),
        "exact": {
            "compliance": exact_compliance,
            "equilibrium_relative_residual": exact_solution[
                "equilibrium_relative_residual"
            ],
            "constraint_relative_residual": exact_solution[
                "constraint_relative_residual"
            ],
            **_stiffness_diagnostics(exact_stiffness, deformation),
        },
        "routes": {},
    }
    for route in routes:
        record: dict[str, Any] = {
            "status": "RUNNING",
            "fallback_used": False,
        }
        summary["routes"][route] = record
        stiffness = predicted[route]
        np.save(target / f"{route}_local_trace_stiffness.npy", stiffness)
        try:
            stiffness_error, stiffness_note = _relative_error(
                stiffness, exact_stiffness
            )
            record["local_trace_stiffness_relative_error"] = stiffness_error
            if stiffness_note is not None:
                record["local_trace_stiffness_zero_reference_note"] = (
                    stiffness_note
                )
            diagnostics = _stiffness_diagnostics(stiffness, deformation)
            record.update(diagnostics)
            if diagnostics["deformation_minimum_eigenvalue"] <= 0.0:
                raise ValueError(
                    "预测局部刚度在变形子空间上不是正定矩阵."
                )
            system = _assemble(
                assembler, sub_meshes, provider.trace_kind, stiffness
            )
            solution = _solve(
                assembler,
                sub_meshes,
                provider.trace,
                provider.trace_kind,
                system,
                load,
                fixed,
            )
            internal, full = _recover(
                assembler,
                sub_meshes,
                positions,
                trace_matrix,
                shape_recovery,
                solution,
            )
            compliance = float(load @ full)
            comparisons = {
                "trace_displacement": _relative_error(
                    np.asarray(solution["trace"]),
                    np.asarray(exact_solution["trace"]),
                ),
                "full_interface_displacement": _relative_error(
                    np.asarray(solution["boundary"]),
                    np.asarray(exact_solution["boundary"]),
                ),
                "internal_displacement": _relative_error(
                    internal, exact_internal
                ),
                "full_displacement": _relative_error(full, exact_full),
                "compliance": _relative_error(
                    np.asarray([compliance]),
                    np.asarray([exact_compliance]),
                ),
            }
            record.update(
                status="COMPLETED",
                compliance=compliance,
                equilibrium_relative_residual=solution[
                    "equilibrium_relative_residual"
                ],
                constraint_relative_residual=solution[
                    "constraint_relative_residual"
                ],
                **{
                    f"{name}_relative_error": result[0]
                    for name, result in comparisons.items()
                },
                zero_reference_notes={
                    name: result[1]
                    for name, result in comparisons.items()
                    if result[1] is not None
                },
            )
            np.save(target / f"{route}_displacement.npy", full)
            np.save(
                target / f"{route}_trace_displacement.npy",
                solution["trace"],
            )
            np.save(
                target / f"{route}_internal_displacement.npy",
                internal,
            )
        except Exception as error:
            record.update(
                status="FAILED",
                error_type=type(error).__name__,
                error=str(error),
            )
        finally:
            _write_json(target / "summary.json", summary)

    summary["status"] = (
        "COMPLETED"
        if all(
            record["status"] == "COMPLETED"
            for record in summary["routes"].values()
        )
        else "FAILED"
    )
    _write_json(target / "summary.json", summary)
    return summary
