"""Run the unified 2D/3D linear-elasticity PINN baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from fealpy.backend import bm
from fealpy.ml.torch_mapping import activations, optimizers

import contract
from cases import ElasticityCase, create_case
from postprocess import show_training_history
import report
from solve import (
    PreparedPINNProblem,
    TrainingResult,
    prepare_problem,
    train_prepared_problem,
)


@dataclass
class ExecutionResult:
    prepared: PreparedPINNProblem
    training: TrainingResult
    payload: dict


def parse_arguments(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a 2D plane-strain or 3D isotropic elasticity problem "
            "using a strong-form PINN."
        )
    )
    parser.add_argument(
        "--dim",
        default=contract.DEFAULT_DIMENSION,
        type=int,
        choices=contract.SUPPORTED_DIMENSIONS,
    )
    parser.add_argument(
        "--pde",
        default="soptx-default",
        choices=("soptx-default",),
        help="Dimension-specific SOPT-X manufactured problem.",
    )
    parser.add_argument(
        "--mesh-size",
        "--mesh_size",
        dest="mesh_size",
        default=None,
        type=int,
        help=(
            "Nodes per coordinate direction for L2 diagnostics; "
            "defaults to 30 in 2D and 8 in 3D."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        "--sampling_mode",
        dest="sampling_mode",
        default=contract.RunConfig.sampling_mode,
        choices=contract.SUPPORTED_SAMPLING_MODES,
        help=(
            "'random' resamples per update; 'linspace' reuses a tensor grid "
            "whose argument is the number of points per free axis."
        ),
    )
    parser.add_argument("--npde", default=contract.RunConfig.npde, type=int)
    parser.add_argument("--nbc", default=contract.RunConfig.nbc, type=int)
    parser.add_argument(
        "--nval-pde",
        "--nval_pde",
        dest="nval_pde",
        default=contract.RunConfig.nval_pde,
        type=int,
    )
    parser.add_argument(
        "--nval-bc",
        "--nval_bc",
        dest="nval_bc",
        default=contract.RunConfig.nval_bc,
        type=int,
    )
    parser.add_argument(
        "--weights",
        default=contract.RunConfig.weights,
        nargs=2,
        type=float,
        metavar=("W_EQ", "W_D"),
    )
    parser.add_argument(
        "--hidden-size",
        "--hidden_size",
        dest="hidden_size",
        default=contract.RunConfig.hidden_size,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--optimizer",
        default=contract.RunConfig.optimizer,
        choices=tuple(optimizers),
    )
    parser.add_argument(
        "--activation",
        default=contract.RunConfig.activation,
        choices=tuple(activations),
    )
    parser.add_argument("--lr", default=contract.RunConfig.lr, type=float)
    parser.add_argument(
        "--step-size",
        "--step_size",
        dest="step_size",
        default=contract.RunConfig.step_size,
        type=int,
        help="Learning-rate decay period; zero disables StepLR.",
    )
    parser.add_argument(
        "--gamma",
        default=contract.RunConfig.gamma,
        type=float,
    )
    parser.add_argument(
        "--epochs",
        default=contract.RunConfig.epochs,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=contract.RunConfig.seed,
        type=int,
    )
    parser.add_argument(
        "--device",
        default=contract.RunConfig.device,
        type=str,
    )
    parser.add_argument(
        "--dtype",
        default=contract.RunConfig.dtype,
        choices=contract.SUPPORTED_DTYPES,
    )
    parser.add_argument(
        "--log-interval",
        "--log_interval",
        dest="log_interval",
        default=contract.RunConfig.log_interval,
        type=int,
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        default=None,
        type=Path,
    )
    parser.add_argument("--summary", default=None, type=Path)
    parser.add_argument(
        "--pbar-log",
        "--pbar_log",
        dest="pbar_log",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        "--log_level",
        dest="log_level",
        default=contract.RunConfig.log_level,
        type=str,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive training-diagnostic figure.",
    )
    return parser.parse_args(args)


def config_from_arguments(arguments: argparse.Namespace) -> contract.RunConfig:
    return contract.RunConfig(
        hidden_size=tuple(arguments.hidden_size),
        activation=arguments.activation,
        optimizer=arguments.optimizer,
        lr=arguments.lr,
        step_size=arguments.step_size,
        gamma=arguments.gamma,
        epochs=arguments.epochs,
        seed=arguments.seed,
        device=arguments.device,
        dtype=arguments.dtype,
        sampling_mode=arguments.sampling_mode,
        npde=arguments.npde,
        nbc=arguments.nbc,
        nval_pde=arguments.nval_pde,
        nval_bc=arguments.nval_bc,
        weights=tuple(arguments.weights),
        log_interval=arguments.log_interval,
        checkpoint_dir=arguments.checkpoint_dir,
        summary_path=arguments.summary,
        diagnostic_mesh_size=arguments.mesh_size,
        pbar_log=arguments.pbar_log,
        log_level=arguments.log_level,
    )


def execute(
    case: ElasticityCase,
    config: contract.RunConfig,
    *,
    command: list[str] | None = None,
) -> ExecutionResult:
    prepared = prepare_problem(case, config)
    training = train_prepared_problem(prepared)
    gates = report.local_gates(training, config)
    payload = report.build_run_payload(
        case,
        config,
        training,
        gates,
        command=command,
        environment=prepared.environment,
    )
    if config.summary_path is not None:
        report.write_json(config.summary_path, payload)
    return ExecutionResult(
        prepared=prepared,
        training=training,
        payload=payload,
    )


def main(args: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(args)
    bm.set_backend("pytorch")
    case = create_case(arguments.dim)
    config = config_from_arguments(arguments)
    result = execute(case, config)
    report.print_run_summary(result.payload)
    if not arguments.no_show:
        show_training_history(result.training.history)
    return 0 if result.payload["local_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
