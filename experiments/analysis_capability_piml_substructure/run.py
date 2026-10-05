"""二维/三维 PIML 子结构: 样本生成、监督训练、局部预测与结构求解验证."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import isfinite, prod
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parents[1] / "src"))


def _json_value(value: Any) -> Any:
    """转换为严格 JSON 值, 不写入 NaN 或 Infinity."""
    import numpy as np
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
    import numpy as np
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


def _conditions(assembler: GlobalAssembler) -> tuple[np.ndarray, np.ndarray]:
    """构造左端固支、右上角单位向下集中力."""
    import numpy as np
    from soptx.backend import backend_manager as bm
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
    import numpy as np

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
        from soptx.backend import backend_manager as bm

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
    import numpy as np
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure import (
        InterfaceDofsView, solve_constrained_system, solve_interface_system,
    )
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
    interface_view = InterfaceDofsView(global_dofs=interface_dofs)
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
    import numpy as np
    from soptx.backend import backend_manager as bm
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
    import numpy as np

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
    import numpy as np
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure import (
        ExactSchurReduction, GlobalAssembler, ShapeFunctionCondensation,
        build_substructures,
    )
    from soptx.ml.substructure.inference import predict_independent_outputs
    if bm.backend_name != "numpy":
        raise RuntimeError("结构求解验证要求 numpy 后端.")
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
        E_base=provider.prototype.E_base,
        nu=float(metadata["poisson_ratio"]),
        hypothesis=metadata.get("material_hypothesis"),
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
    shape_prediction = predict_independent_outputs(
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
        stiffness_prediction = predict_independent_outputs(
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


def parse_args():
    """解析命令行参数并检查参数组合."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--all", dest="stage", action="store_const", const="all",
                         help="构建网络、生成样本并训练")
    actions.add_argument("--generate-samples", dest="stage", action="store_const",
                         const="samples", help="仅生成样本")
    actions.add_argument("--train", dest="stage", action="store_const",
                         const="train", help="使用已有样本训练")
    actions.add_argument("--analyze", dest="stage", action="store_const",
                         const="analyze", help="加载最佳权重并与同接口精确缩聚比较")
    actions.add_argument("--validate-local", dest="stage", action="store_const",
                         const="local", help="加载单路线权重并执行局部预测验证")
    parser.add_argument("--checkpoint", type=Path, help="--validate-local 使用的权重文件")
    parser.add_argument("--n-test", type=int, default=1000, help="局部验证样本数")
    parser.add_argument("--test-batch-size", type=int, default=32, help="局部验证计算批量")
    parser.add_argument("--checkpoint-dir", type=Path, help="--analyze 使用的最佳权重目录")
    parser.add_argument("--n-sub", type=int, nargs="+",
                        help="--analyze 的各方向子结构数, 默认 2 1 或 2 1 1")
    parser.add_argument("--dim", type=int, choices=(2, 3), help="空间维数, 样本生成入口默认 3")
    parser.add_argument("--hypothesis", choices=("plane_stress", "plane_strain"),
                        help="二维材料假设, 默认 plane_stress; 三维不接受此参数")
    parser.add_argument("--cell-size", type=float, nargs="+",
                        help="子结构各方向尺寸, 数量须与 --dim 一致, 默认各方向为 1")
    parser.add_argument("--n-fine", type=int, help="每个方向的细单元数, 样本生成入口默认 5, 至少为 2")
    parser.add_argument("--trace-kind", choices=("linear_corner", "full_trace"),
                        help="接口空间, 样本生成入口默认 linear_corner")
    parser.add_argument("--dataset", type=Path, help="已有数据集目录")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "outputs")
    parser.add_argument("--n-train", type=int, default=400_000)
    parser.add_argument("--n-validation", type=int, default=40_000)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--min-modulus", type=float, default=1e-6)
    parser.add_argument("--route", choices=("shape", "stiffness"),
                        help="训练与整体分析的路线, 默认 shape; 两条路线须分别训练; 局部验证从权重读取")
    parser.add_argument("--num-networks", type=int,
                        help="所选路线的输出拆分网络数量, 默认 4")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD 动量, 非零值仅适用于 --optimizer sgd")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage in ("train", "local", "analyze"):
        geometry_options = ("dim", "cell_size", "n_fine", "trace_kind", "hypothesis")
        if any(getattr(args, name) is not None for name in geometry_options):
            parser.error("--train / --validate-local / --analyze 从数据集或权重恢复子结构配置, "
                         "不接受 --dim、--cell-size、--n-fine、--trace-kind 或 --hypothesis")
    if args.stage in ("all", "samples"):
        args.dim = 3 if args.dim is None else args.dim
        args.n_fine = 5 if args.n_fine is None else args.n_fine
        args.trace_kind = "linear_corner" if args.trace_kind is None else args.trace_kind
        if args.dim == 3 and args.hypothesis is not None:
            parser.error("--hypothesis 仅用于 --dim 2")
        if args.dim == 2 and args.hypothesis is None:
            args.hypothesis = "plane_stress"
        if args.cell_size is None:
            args.cell_size = [1.0] * args.dim
        if len(args.cell_size) != args.dim or any(
            not isfinite(size) or size <= 0 for size in args.cell_size
        ):
            parser.error("--cell-size 须包含与 --dim 同数量的有限正数")
    if (args.stage == "train") != (args.dataset is not None):
        parser.error("--train 必须指定 --dataset, 其他入口不接受 --dataset")
    if (args.stage == "analyze") != (args.checkpoint_dir is not None):
        parser.error("--analyze 必须指定 --checkpoint-dir, 其他入口不接受该参数")
    if (args.stage == "local") != (args.checkpoint is not None):
        parser.error("--validate-local 必须指定 --checkpoint, 其他入口不接受该参数")
    if args.stage == "local":
        if args.route is not None:
            parser.error("--validate-local 从权重读取路线, 不接受 --route")
        if not args.checkpoint.is_file():
            parser.error(f"权重文件不存在: {args.checkpoint}")
        if args.num_networks is not None:
            parser.error("--validate-local 从权重读取网络数量, 不接受 --num-networks")
        if args.device not in ("cpu", "cuda"):
            parser.error("--validate-local 的 device 须为 cpu 或 cuda")
    if args.n_test <= 0 or args.test_batch_size <= 0:
        parser.error("--n-test 与 --test-batch-size 必须为正整数")
    if args.n_sub is not None and args.stage != "analyze":
        parser.error("--n-sub 仅用于 --analyze")
    if args.stage == "analyze":
        if args.device != "cpu":
            parser.error("--analyze 当前仅支持 --device cpu")
        if args.num_networks is not None:
            parser.error("--analyze 从权重读取网络数量, 不接受 --num-networks")
        if args.n_sub is not None and min(args.n_sub) <= 0:
            parser.error("--n-sub 须为正整数列表")
    if min(args.n_train, args.n_validation, args.generation_batch_size,
           args.epochs, args.batch_size) <= 0:
        parser.error("样本数、批量大小与训练轮数必须为正数")
    if args.num_networks is not None and args.num_networks <= 0:
        parser.error("--num-networks 必须为正整数")
    if args.stage in ("all", "samples") and args.n_fine < 2:
        parser.error("--n-fine 至少为 2, 以保留内部自由度")
    if not 0 < args.min_modulus < 1:
        parser.error("--min-modulus 必须位于 (0, 1)")
    if not isfinite(args.lr) or args.lr < 1e-6 or args.patience < 0 or args.seed < 0:
        parser.error("lr 不得低于 1e-6, patience 与 seed 不能为负")
    if not isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay 必须为有限非负数")
    if not isfinite(args.momentum) or args.momentum < 0:
        parser.error("--momentum 必须为有限非负数")
    if args.optimizer != "sgd" and args.momentum != 0:
        parser.error("非零 --momentum 仅适用于 --optimizer sgd")
    if args.stage != "local" and args.route is None:
        args.route = "shape"
    return args


def resolve_analysis_n_sub(n_sub, spatial_dimension):
    """按权重恢复的空间维数确定并校验整体结构排列."""
    if spatial_dimension not in (2, 3):
        raise ValueError("权重中的空间维数必须为 2 或 3")
    values = ([2] + [1] * (spatial_dimension - 1)
              if n_sub is None else list(n_sub))
    if len(values) != spatial_dimension or any(value <= 0 for value in values):
        raise ValueError(f"--n-sub 须包含 {spatial_dimension} 个正整数")
    return tuple(values)


def main():
    """分派局部验证、结构求解或网络构建与样本训练流程."""
    args = parse_args()
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider
    from soptx.ml.substructure.independent_contract import build_network
    from soptx.ml.substructure.independent_data import prepare_training_data
    from soptx.ml.substructure.independent_training import train_network
    from soptx.ml.substructure.training import TrainingConfig

    output_root = args.output_dir / "independent_15_layer"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    if args.stage in ("analyze", "local"):
        from soptx.backend import backend_manager as bm
        bm.set_backend("numpy")

    if args.stage in ("train", "local"):
        from soptx.ml.substructure.independent_contract import SCHEMA, provider_metadata_matches

        if args.stage == "train":
            manifest_path = args.dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            import torch

            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
                    or payload.get("route") not in ("shape", "stiffness")):
                raise ValueError("权重格式或预测路线不匹配")
            args.route = payload["route"]
            manifest = payload.get("dataset")
            del payload
        if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
                or manifest.get("complete") is not True
                or manifest.get("input_quantity") != "normalized_young_modulus"):
            raise ValueError("保存的数据集记录未完成或格式不支持")
        saved = manifest.get("provider")
        required = {"cell_size", "n_fine", "poisson_ratio", "trace", "spatial_dimension"}
        if not isinstance(saved, dict) or not required.issubset(saved):
            raise ValueError("保存的元数据缺少子结构配置")
        if saved["spatial_dimension"] == 2 and "material_hypothesis" not in saved:
            raise ValueError("二维数据集缺少 material_hypothesis")
        provider = IndependentTargetProvider(
            cell_size=tuple(saved["cell_size"]), n_fine=tuple(saved["n_fine"]),
            nu=saved["poisson_ratio"], trace_kind=saved["trace"],
            hypothesis=saved.get("material_hypothesis"),
        )
        if not provider_metadata_matches(saved, provider.metadata()):
            raise ValueError("恢复的子结构配置或独立分量编号与数据集不一致")
        print(f"恢复配置: trace={saved['trace']}, "
              f"cell_size={saved['cell_size']}, n_fine={saved['n_fine']}", flush=True)
    elif args.stage == "analyze":
        from soptx.ml.substructure.independent_checkpoints import load_analysis_provider

        provider = load_analysis_provider(args.checkpoint_dir)
        saved = provider.metadata()
        print(f"恢复配置: trace={saved['trace']}, "
              f"cell_size={saved['cell_size']}, n_fine={saved['n_fine']}", flush=True)
    else:
        provider = IndependentTargetProvider(
            cell_size=tuple(args.cell_size), n_fine=(args.n_fine,) * args.dim,
            trace_kind=args.trace_kind, hypothesis=args.hypothesis,
        )
    metadata = provider.metadata()

    if args.stage == "analyze":
        from soptx.ml.substructure.independent_checkpoints import load_analysis_networks

        n_sub = resolve_analysis_n_sub(args.n_sub, metadata["spatial_dimension"])
        networks, sources = load_analysis_networks(
            args.checkpoint_dir, metadata, route=args.route,
        )
        routes = (args.route,)
        output = output_root / "analysis" / stamp
        summary = run_analysis(
            provider, networks, output, n_sub=n_sub,
            seed=args.seed, routes=routes, checkpoint_sources=sources,
        )
        print(f"结构求解结果: {output}")
        if summary["status"] != "COMPLETED":
            raise SystemExit("结构求解未全部完成, 请检查输出目录中的 summary.json")
        return

    if args.stage == "local":
        from soptx.ml.substructure.validation import load_local_model
        from local_validation import evaluate_local_predictions

        output = output_root / "local_validation" / f"{stamp}_{args.route}"
        print(f"加载权重: {args.checkpoint.resolve()}", flush=True)
        network = load_local_model(args.checkpoint, provider, args.route, args.device)
        print(f"局部预测: route={args.route}, n_test={args.n_test}, "
              f"batch_size={args.test_batch_size}", flush=True)
        print(f"结果目录: {output.resolve()}", flush=True)
        summary = evaluate_local_predictions(
            network, provider, args.route, n_test=args.n_test,
            min_modulus=args.min_modulus, batch_size=args.test_batch_size,
            seed=args.seed, device=args.device, output_dir=output,
        )
        print(f"状态: {summary['status']}, 已完成样本数: {summary['n_completed']}")
        print(f"汇总: {output.resolve() / 'summary.json'}")
        return

    # 1. 构建网络.
    if args.stage != "samples":
        options = {} if args.num_networks is None else {"num_networks": args.num_networks}
        network = build_network(metadata, route=args.route, seed=args.seed, **options)

    # 2. 生成训练与验证样本, 或使用已有数据集.
    dataset = args.dataset
    if args.stage != "train":
        dataset = prepare_training_data(
            provider, output_root / "samples" / stamp,
            n_train=args.n_train, n_validation=args.n_validation,
            batch_size=args.generation_batch_size,
            min_modulus=args.min_modulus, seed=args.seed,
        )
        print(f"数据集: {dataset}")
    if args.stage == "samples":
        return

    # 3. 训练网络并保存最佳权重.
    optimizer_params = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        optimizer_params["momentum"] = args.momentum
    config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, patience=args.patience,
        optimizer=args.optimizer, optimizer_params=optimizer_params,
    )
    output = output_root / "training" / args.route / stamp
    train_network(
        dataset, output, route=args.route, network=network,
        provider=provider, device=args.device, config=config,
    )
    print(f"模型与训练记录: {output}")


if __name__ == "__main__":
    main()
