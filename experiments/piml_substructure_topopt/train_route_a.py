#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线训练 PIML Route A 形函数代理并保存带签名 checkpoint。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fealpy.backend import backend_manager as bm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soptx.fem.substructure import (
    ExactSchurReduction,
    GlobalAssembler,
    PIMLShapeReduction,
    SubstructurePrototype,
    build_substructures,
)
from soptx.ml import ShapeFunctionSurrogateNet
from soptx.ml.substructure import (
    DensitySamplingConfig,
    ModelSignature,
    SamplingFractions,
    TrainingConfig,
    sample_density_fields,
    save_checkpoint,
    train_surrogate,
)
from soptx.ml.substructure.sampling import SAMPLER_VERSION

from config import OUTPUT_DIR, TopOptCase, load


def build_context(
    case: TopOptCase,
) -> tuple[SubstructurePrototype, GlobalAssembler]:
    """按拓扑工况构造唯一子结构原型及其全局装配器。"""
    domain_size = tuple(
        case.domain[2 * axis + 1] - case.domain[2 * axis]
        for axis in range(case.dim)
    )
    assembler = GlobalAssembler(
        domain_size,
        case.n_sub,
        case.n_fine,
        E_base=case.emax,
        nu=case.nu,
    )
    prototype, _, _ = build_substructures(assembler)
    prototype.rho_min = case.emin
    prototype.penal = case.simp_penalty
    return prototype, assembler


def model_signature(prototype: SubstructurePrototype) -> ModelSignature:
    n_fine = tuple(int(value) for value in prototype.n_fine)
    n_interior = int(len(prototype.i_dofs))
    n_reduced = int(prototype.deformation_basis.shape[1])
    return ModelSignature(
        n_fine=n_fine,
        input_dim=int(np.prod(n_fine)),
        output_dim=n_interior * n_reduced,
        n_interior_dofs=n_interior,
        n_reduced=n_reduced,
        sampler_version=SAMPLER_VERSION,
    )


def exact_shape_function_targets(
    prototype: SubstructurePrototype,
    density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    r"""用 Exact Schur 生成 $M=N R_{\perp}$ 标签及精确降阶刚度。"""
    density_bm = bm.asarray(density, dtype=bm.float64)
    density_cell = prototype.grid_to_cell_field(density_bm)
    local_stiffness = prototype.assemble_local_stiffness_batch(density_cell)
    exact = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    result = exact.reduce_many(local_stiffness, density_bm)
    target = bm.einsum(
        "bij,jk->bik",
        result.recovery,
        prototype.deformation_basis,
    )
    x = density.reshape(len(density), -1).astype(np.float32)
    y = bm.to_numpy(target).reshape(len(density), -1).astype(np.float32)
    return x, y, local_stiffness, result.stiffness


def _scalar(data: np.lib.npyio.NpzFile, name: str) -> str:
    return str(np.asarray(data[name]).item())


def load_exact_trajectory(
    path: Path,
    case: TopOptCase,
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取一条完整 Exact 轨迹；不在轨迹内部切分 train/validation。"""
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    required = {
        "schema_version",
        "solver_mode",
        "trajectory_id",
        "density_role",
        "volfrac",
        "n_sub",
        "n_fine",
        "rho_sub",
    }
    with np.load(resolved, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"轨迹 {resolved} 缺少字段: {missing}")
        if _scalar(data, "schema_version") != "substructure-density-trajectory-v2":
            raise ValueError(f"轨迹 {resolved} 的 schema_version 不兼容。")
        if _scalar(data, "solver_mode") != "fea_baseline":
            raise ValueError(f"轨迹 {resolved} 不是 fea_baseline 生成。")
        if _scalar(data, "density_role") != "training_candidate":
            raise ValueError(f"轨迹 {resolved} 未声明为 training_candidate。")
        n_sub = tuple(int(value) for value in np.asarray(data["n_sub"]))
        n_fine = tuple(int(value) for value in np.asarray(data["n_fine"]))
        if n_sub != case.n_sub or n_fine != case.n_fine:
            raise ValueError(
                f"轨迹 {resolved} 的划分 {(n_sub, n_fine)} 与工况不一致。"
            )
        rho_sub = np.asarray(data["rho_sub"], dtype=np.float64)
        expected_tail = (int(np.prod(case.n_sub)), *case.n_fine)
        if rho_sub.ndim != len(expected_tail) + 1 or tuple(rho_sub.shape[1:]) != expected_tail:
            raise ValueError(
                f"轨迹 {resolved} 的 rho_sub 形状 {rho_sub.shape} 不符合 "
                f"(n_iter, {expected_tail})。"
            )
        values = rho_sub.reshape((-1, *case.n_fine))
        metadata = {
            "path": str(resolved),
            "sha256": digest,
            "trajectory_id": _scalar(data, "trajectory_id"),
            "volfrac": float(np.asarray(data["volfrac"]).item()),
            "iteration_count": int(rho_sub.shape[0]),
            "local_sample_count": int(values.shape[0]),
        }
    return values, metadata


def _raw_route_a_stiffness(
    model: torch.nn.Module,
    prototype: SubstructurePrototype,
    density: np.ndarray,
    local_stiffness: Any,
) -> Any:
    r"""由 raw 网络输出按式 (17) 构造 $\widehat K$，不执行 Exact fallback。"""
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(density.reshape(len(density), -1), dtype=torch.float32)
        ).cpu().numpy()
    deformation = bm.reshape(
        bm.asarray(prediction, dtype=bm.float64),
        (len(density), len(prototype.i_dofs), prototype.deformation_basis.shape[1]),
    )
    rigid = (
        prototype.rigid_interior_modes
        @ bm.matrix_transpose(prototype.rigid_basis)
    )
    recovery = (
        rigid[None, :, :]
        + deformation @ bm.matrix_transpose(prototype.deformation_basis)
    )
    i_dofs = prototype.i_dofs
    b_dofs = prototype.b_dofs
    K_ii = local_stiffness[:, i_dofs[:, None], i_dofs]
    K_ib = local_stiffness[:, i_dofs[:, None], b_dofs]
    K_bb = local_stiffness[:, b_dofs[:, None], b_dofs]
    cross = bm.transpose(K_ib, (0, 2, 1)) @ recovery
    return (
        K_bb
        + cross
        + bm.transpose(cross, (0, 2, 1))
        + bm.transpose(recovery, (0, 2, 1)) @ K_ii @ recovery
    )


def make_physics_evaluator(
    prototype: SubstructurePrototype,
    validation_density: np.ndarray,
    validation_sources: tuple[str, ...],
    validation_stiffness: Any,
    exact_reduced_stiffness: Any,
    solid_stiffness_norm: float,
    gate_weight: float,
):
    """构造分来源 gate 与 raw 式 (17) 刚度误差评价器。"""
    density_bm = bm.asarray(validation_density, dtype=bm.float64)
    source_array = np.asarray(validation_sources)

    def evaluate(model: torch.nn.Module) -> dict[str, Any]:
        raw_stiffness = _raw_route_a_stiffness(
            model,
            prototype,
            validation_density,
            validation_stiffness,
        )
        raw_np = bm.to_numpy(raw_stiffness)
        exact_np = bm.to_numpy(exact_reduced_stiffness)
        numerator = np.linalg.norm(raw_np - exact_np, axis=(1, 2))
        exact_norm = np.linalg.norm(exact_np, axis=(1, 2))
        denominator = np.maximum(exact_norm, 1.0e-12 * solid_stiffness_norm)
        relative_error = numerator / denominator

        reduction = PIMLShapeReduction(
            prototype.i_dofs,
            prototype.b_dofs,
            model=model,
            rigid_basis=prototype.rigid_basis,
            deformation_basis=prototype.deformation_basis,
            rigid_interior=prototype.rigid_interior_modes,
        )
        gated = reduction.reduce_many(validation_stiffness, density_bm)
        fallback = np.asarray(
            [item.used_fallback for item in gated.diagnostics], dtype=bool
        )
        spread = np.max(validation_density, axis=tuple(range(1, validation_density.ndim)))
        mean = np.mean(validation_density, axis=tuple(range(1, validation_density.ndim)))
        deployment_eligible = (spread - mean) >= 1.0e-4

        by_source: dict[str, Any] = {}
        worst_p95 = 0.0
        minimum_gate_pass = 1.0
        trajectory_p95: list[float] = []
        trajectory_gate_pass: list[float] = []
        for source in sorted(set(validation_sources)):
            indices = np.flatnonzero(source_array == source)
            eligible = indices[deployment_eligible[indices]]
            deployment_indices = eligible if len(eligible) else indices
            errors = relative_error[deployment_indices]
            gate_pass_rate = 1.0 - float(np.mean(fallback[deployment_indices]))
            p95 = float(np.quantile(errors, 0.95))
            by_source[source] = {
                "count": int(len(indices)),
                "deployment_eligible_count": int(len(eligible)),
                "gate_pass_rate": gate_pass_rate,
                "eq17_relative_error_median": float(np.median(errors)),
                "eq17_relative_error_p95": p95,
                "eq17_relative_error_max": float(np.max(errors)),
            }
            worst_p95 = max(worst_p95, p95)
            minimum_gate_pass = min(minimum_gate_pass, gate_pass_rate)
            if source.startswith("trajectory:"):
                trajectory_p95.append(p95)
                trajectory_gate_pass.append(gate_pass_rate)

        if not trajectory_p95:
            raise ValueError("physics validation 必须包含独立 trajectory 来源。")
        trajectory_worst_p95 = max(trajectory_p95)
        trajectory_minimum_gate_pass = min(trajectory_gate_pass)
        selection_score = trajectory_worst_p95 + gate_weight * (
            1.0 - trajectory_minimum_gate_pass
        )
        reasons = Counter(
            item.fallback_reason or "unspecified"
            for item in gated.diagnostics
            if item.used_fallback
        )
        return {
            "selection_score": float(selection_score),
            "selection_definition": (
                "validation_trajectory_worst_eq17_p95 + gate_weight * "
                "(1 - validation_trajectory_worst_gate_pass_rate)"
            ),
            "selection_gate_weight": float(gate_weight),
            "worst_source_eq17_relative_error_p95": float(worst_p95),
            "worst_source_gate_pass_rate": float(minimum_gate_pass),
            "trajectory_worst_eq17_relative_error_p95": float(
                trajectory_worst_p95
            ),
            "trajectory_worst_gate_pass_rate": float(
                trajectory_minimum_gate_pass
            ),
            "evaluated_count": int(len(gated)),
            "fallback_count": int(np.sum(fallback)),
            "fallback_reasons": dict(reasons),
            "by_source": by_source,
        }

    return evaluate


def _replace_trajectory_source(
    sources: tuple[str, ...],
    trajectory_id: str,
) -> tuple[str, ...]:
    return tuple(
        f"trajectory:{trajectory_id}" if source == "trajectory" else source
        for source in sources
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="离线训练 PIML Route A checkpoint")
    parser.add_argument("--case", default="mbb_piml_route_a")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--trajectory-fraction", type=float, default=0.2)
    parser.add_argument("--training-trajectory", type=Path, required=True)
    parser.add_argument("--validation-trajectory", type=Path, required=True)
    parser.add_argument("--design-min", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="连续多少次 physics 检查未改善后停止",
    )
    parser.add_argument("--physics-eval-interval", type=int, default=25)
    parser.add_argument("--selection-gate-weight", type=float, default=1.0)
    parser.add_argument(
        "--refit-all-trajectories",
        action="store_true",
        help="按已选 best_epoch 用 training+validation 两条轨迹从头固定轮数重训",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    _, cases = load()
    matches = [case for case in cases if case.id == args.case]
    if len(matches) != 1:
        parser.error(f"未找到唯一工况: {args.case}")
    case = matches[0]
    if case.solver_mode != "piml_route_a":
        parser.error("train_route_a.py 只接受 piml_route_a 工况")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction 必须位于 (0, 1)")
    if not 0.0 < args.trajectory_fraction < 1.0:
        parser.error("--trajectory-fraction 必须位于 (0, 1)")
    if args.physics_eval_interval <= 0:
        parser.error("--physics-eval-interval 必须为正整数")
    if args.selection_gate_weight < 0.0:
        parser.error("--selection-gate-weight 不能为负数")

    prototype, _ = build_context(case)
    signature = model_signature(prototype)
    training_trajectory, training_meta = load_exact_trajectory(
        args.training_trajectory,
        case,
    )
    validation_trajectory, validation_meta = load_exact_trajectory(
        args.validation_trajectory,
        case,
    )
    if training_meta["sha256"] == validation_meta["sha256"]:
        parser.error("training 与 validation 轨迹文件 hash 相同，存在数据泄漏。")
    if training_meta["trajectory_id"] == validation_meta["trajectory_id"]:
        parser.error("training 与 validation 的 trajectory_id 相同，存在数据泄漏。")

    synthetic_fraction = (1.0 - args.trajectory_fraction) / 4.0
    fractions = SamplingFractions(
        continuous=synthetic_fraction,
        low_density=synthetic_fraction,
        near_binary=synthetic_fraction,
        correlated=synthetic_fraction,
        trajectory=args.trajectory_fraction,
    )
    validation_count = max(1, int(round(args.samples * args.validation_fraction)))
    training_count = args.samples - validation_count
    if training_count <= 0:
        parser.error("训练样本数必须大于验证样本数")

    training_sampling_config = DensitySamplingConfig(
        shape=signature.n_fine,
        design_min=args.design_min,
        seed=args.seed,
        fractions=fractions,
    )
    validation_sampling_config = DensitySamplingConfig(
        shape=signature.n_fine,
        design_min=args.design_min,
        seed=args.seed + 1,
        fractions=fractions,
    )
    training_samples = sample_density_fields(
        training_count,
        training_sampling_config,
        trajectory=training_trajectory,
    )
    validation_samples = sample_density_fields(
        validation_count,
        validation_sampling_config,
        trajectory=validation_trajectory,
    )
    validation_sources = _replace_trajectory_source(
        validation_samples.sources,
        validation_meta["trajectory_id"],
    )

    x_train, y_train, _, _ = exact_shape_function_targets(
        prototype,
        training_samples.values,
    )
    x_validation, y_validation, validation_stiffness, exact_validation = (
        exact_shape_function_targets(prototype, validation_samples.values)
    )
    _, _, _, solid_reduced = exact_shape_function_targets(
        prototype,
        np.ones((1, *signature.n_fine), dtype=np.float64),
    )
    solid_stiffness_norm = float(
        np.linalg.norm(bm.to_numpy(solid_reduced)[0])
    )

    torch.manual_seed(args.seed)
    model = ShapeFunctionSurrogateNet(signature.input_dim, signature.output_dim)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        patience=args.patience,
        physics_eval_interval=args.physics_eval_interval,
    )
    evaluator = make_physics_evaluator(
        prototype,
        validation_samples.values,
        validation_sources,
        validation_stiffness,
        exact_validation,
        solid_stiffness_norm,
        args.selection_gate_weight,
    )
    result = train_surrogate(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        training_config,
        evaluator=evaluator,
    )
    refit_summary: dict[str, Any] | None = None
    if args.refit_all_trajectories:
        combined_trajectory = np.concatenate(
            (training_trajectory, validation_trajectory), axis=0
        )
        refit_sampling_config = DensitySamplingConfig(
            shape=signature.n_fine,
            design_min=args.design_min,
            seed=args.seed + 2,
            fractions=fractions,
        )
        refit_samples = sample_density_fields(
            args.samples,
            refit_sampling_config,
            trajectory=combined_trajectory,
        )
        x_refit, y_refit, _, _ = exact_shape_function_targets(
            prototype,
            refit_samples.values,
        )
        torch.manual_seed(args.seed)
        model = ShapeFunctionSurrogateNet(signature.input_dim, signature.output_dim)
        refit_config = TrainingConfig(
            epochs=result.best_epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            patience=0,
            select_final_state=True,
        )
        refit_result = train_surrogate(
            model,
            x_refit,
            y_refit,
            x_validation,
            y_validation,
            refit_config,
        )
        if refit_result.best_epoch != result.best_epoch:
            raise RuntimeError("fixed-epoch refit 未停在预注册的 best_epoch。")
        refit_summary = {
            "epochs": refit_result.best_epoch,
            "sampling_config": asdict(refit_sampling_config),
            "source_counts": dict(refit_samples.source_counts),
            "train_size": len(refit_samples.values),
            "final_training_loss": refit_result.final_training_loss,
            "monitor_validation_loss": refit_result.best_validation_loss,
            "trajectory_sha256": [
                training_meta["sha256"],
                validation_meta["sha256"],
            ],
        }
    destination = args.output_path or OUTPUT_DIR / f"{case.id}_checkpoint.pt"
    summary = {
        "case_id": case.id,
        "model_stage": "fixed_epoch_refit" if refit_summary else "selection",
        "sampler_version": SAMPLER_VERSION,
        "training_sampling_config": asdict(training_sampling_config),
        "validation_sampling_config": asdict(validation_sampling_config),
        "training_source_counts": dict(training_samples.source_counts),
        "validation_source_counts": dict(Counter(validation_sources)),
        "dataset_roles": {
            "training_trajectory": training_meta,
            "validation_trajectory": validation_meta,
            "final_refit_trajectories": (
                [training_meta, validation_meta] if refit_summary else []
            ),
            "canonical_regression": "not consumed by training",
        },
        "training_config": asdict(training_config),
        "train_size": len(training_samples.values),
        "validation_size": len(validation_samples.values),
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "best_selection_score": result.best_selection_score,
        "final_training_loss": (
            refit_summary["final_training_loss"]
            if refit_summary
            else result.final_training_loss
        ),
        "evaluation": dict(result.evaluation),
        "physics_history": [dict(item) for item in result.physics_history],
        "refit": refit_summary,
    }
    save_checkpoint(destination, model, signature, summary)
    print(f"[+] Route A checkpoint 已保存: {destination}")
    print(f"[+] best physics validation: {summary['evaluation']}")


if __name__ == "__main__":
    main()
