"""Correctness and training-baseline validation for elasticity PINNs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Sequence

import torch

from fealpy.backend import bm

import contract
from cases import ElasticityCase, create_case
import layout
from postprocess import relative_l2_metrics
from references import (
    fixed_diagnostic_points,
    make_exact_operator,
    manufactured_validation_points,
)
import report
from run import execute
from solve import (
    prepare_problem,
    restore_best_state,
    train_prepared_problem,
)


def positive_epochs(value: str) -> int:
    epochs = int(value)
    if epochs < 1:
        raise argparse.ArgumentTypeError("--epochs must be positive")
    return epochs


def parse_arguments(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the unified 2D/3D linear-elasticity PINN baseline."
    )
    parser.add_argument(
        "--dim",
        choices=("2", "3", "all"),
        default="all",
    )
    parser.add_argument(
        "--epochs",
        type=positive_epochs,
        default=None,
        help=(
            "Positive development override for parameter updates. "
            "Only the default count is eligible for formal evidence."
        ),
    )
    return parser.parse_args(args)


def selected_dimensions(value: str) -> tuple[int, ...]:
    if value == "all":
        return contract.SUPPORTED_DIMENSIONS
    return (int(value),)


def add_check(
    result: dict[str, Any],
    name: str,
    passed: bool,
    *,
    metrics: dict[str, Any] | None = None,
    detail: str = "",
) -> None:
    result["checks"][name] = {
        "passed": bool(passed),
        "metrics": {} if metrics is None else metrics,
        "detail": detail,
    }
    if not passed:
        result["failures"].append(name if not detail else f"{name}: {detail}")


def expect_exception(
    action: Callable[[], Any],
    exception_type: type[BaseException],
    message_fragment: str,
) -> tuple[bool, str]:
    try:
        action()
    except exception_type as error:
        message = str(error)
        return message_fragment in message, message
    except BaseException as error:  # noqa: BLE001 - report wrong type.
        return False, f"{type(error).__name__}: {error}"
    return False, "No exception was raised."


def smoke_config(
    case: ElasticityCase,
    **overrides: Any,
) -> contract.RunConfig:
    config = contract.RunConfig(
        epochs=1,
        hidden_size=(4,),
        npde=4,
        nbc=2,
        nval_pde=4,
        nval_bc=2,
        log_interval=1,
        seed=7,
        device="cpu",
        dtype="float64",
        diagnostic_mesh_size=4 if case.dimension == 2 else 3,
        log_level="WARNING",
    )
    return replace(config, **overrides)


def validate_constructor(
    case: ElasticityCase,
    result: dict[str, Any],
) -> None:
    prepared = prepare_problem(case, smoke_config(case))
    points = torch.zeros(
        (3, case.dimension),
        dtype=torch.float64,
    )
    prediction = prepared.operator.predict(points)
    passed = (
        prepared.case.dimension == case.dimension
        and tuple(prediction.shape) == (3, case.dimension)
        and len(prepared.case.domain) == 2 * case.dimension
    )
    add_check(
        result,
        "constructor_and_shape",
        passed,
        metrics={
            "dimension": prepared.case.dimension,
            "case": prepared.case.name,
            "pde_class": prepared.case.problem.__class__.__name__,
            "prediction_shape": list(prediction.shape),
            "material": prepared.case.material.as_dict(),
        },
    )


def validate_exact_solution(
    case: ElasticityCase,
    result: dict[str, Any],
) -> None:
    device = torch.device("cpu")
    operator = make_exact_operator(
        case,
        dtype=torch.float64,
        device=device,
    )
    interior, boundary = manufactured_validation_points(
        case,
        dtype=torch.float64,
        device=device,
    )
    computed_gradient = operator.displacement_gradient(interior)
    exact_gradient = case.exact_displacement_gradient(interior)
    strain = operator.strain(interior)
    equilibrium = operator.equilibrium_residual(
        interior,
        create_graph=False,
    )
    boundary_residual = operator.dirichlet_residual(boundary)

    gradient_max = float(
        torch.max(torch.abs(computed_gradient - exact_gradient)).item()
    )
    symmetry_max = float(
        torch.max(torch.abs(strain - strain.transpose(-1, -2))).item()
    )
    component_max = torch.max(
        torch.abs(equilibrium),
        dim=0,
    ).values
    equilibrium_max = float(torch.max(torch.abs(equilibrium)).item())
    boundary_max = float(torch.max(torch.abs(boundary_residual)).item())
    passed = (
        symmetry_max <= contract.EXACT_STRAIN_SYMMETRY_MAX_ABS
        and gradient_max <= contract.EXACT_GRADIENT_MAX_ABS
        and equilibrium_max <= contract.EXACT_EQUILIBRIUM_MAX_ABS
        and boundary_max <= contract.EXACT_BOUNDARY_MAX_ABS
    )
    add_check(
        result,
        "manufactured_solution_consistency",
        passed,
        metrics={
            "interior_points": int(interior.shape[0]),
            "boundary_points": int(boundary.shape[0]),
            "strain_symmetry_max_abs": symmetry_max,
            "displacement_gradient_max_abs": gradient_max,
            "equilibrium_component_max_abs": [
                float(value) for value in component_max.tolist()
            ],
            "equilibrium_residual_max_abs": equilibrium_max,
            "dirichlet_residual_max_abs": boundary_max,
        },
    )


class MixedBoundaryProblem:
    """Delegate manufactured data but make one component non-Dirichlet."""

    def __init__(self, problem) -> None:
        self._problem = problem

    def __getattr__(self, name: str):
        return getattr(self._problem, name)

    def is_dirichlet_boundary(self):
        predicates = list(self._problem.is_dirichlet_boundary())

        def never(points):
            return torch.zeros(
                points.shape[:-1],
                dtype=torch.bool,
                device=points.device,
            )

        predicates[0] = never
        return tuple(predicates)


def validate_failure_guards(
    case: ElasticityCase,
    result: dict[str, Any],
) -> None:
    guards: dict[str, dict[str, Any]] = {}

    bm.set_backend("numpy")
    try:
        passed, detail = expect_exception(
            lambda: prepare_problem(case, smoke_config(case)),
            RuntimeError,
            "PyTorch backend",
        )
    finally:
        bm.set_backend("pytorch")
    guards["non_pytorch_backend"] = {"passed": passed, "detail": detail}

    passed, detail = expect_exception(
        lambda: create_case(4),
        ValueError,
        "2 or 3",
    )
    guards["unsupported_dimension"] = {"passed": passed, "detail": detail}

    if case.dimension == 2:
        plane_stress_problem = create_case(2).problem
        plane_stress_problem.plane_type = "plane_stress"
        plane_stress_case = replace(case, problem=plane_stress_problem)
        passed, detail = expect_exception(
            lambda: prepare_problem(
                plane_stress_case,
                smoke_config(plane_stress_case),
            ),
            ValueError,
            "plane_type='plane_strain'",
        )
        guards["plane_stress_problem"] = {
            "passed": passed,
            "detail": detail,
        }
    else:
        wrong_hypothesis = replace(
            case,
            material=replace(case.material, hypothesis="plane_strain"),
        )
        passed, detail = expect_exception(
            lambda: prepare_problem(
                wrong_hypothesis,
                smoke_config(wrong_hypothesis),
            ),
            ValueError,
            "hypothesis='3D'",
        )
        guards["non_3d_material_hypothesis"] = {
            "passed": passed,
            "detail": detail,
        }

    mixed_case = replace(
        case,
        problem=MixedBoundaryProblem(case.problem),
    )
    mixed_prepared = prepare_problem(
        mixed_case,
        smoke_config(mixed_case),
    )
    _, boundary = manufactured_validation_points(
        case,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    passed, detail = expect_exception(
        lambda: mixed_prepared.operator.dirichlet_residual(boundary),
        ValueError,
        "requires every sampled boundary point",
    )
    guards["mixed_boundary_problem"] = {
        "passed": passed,
        "detail": detail,
    }

    add_check(
        result,
        "unsupported_problem_guards",
        all(record["passed"] for record in guards.values()),
        metrics=guards,
    )


def validate_one_update(
    case: ElasticityCase,
    result: dict[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f"soptx-pinn-{case.dimension}d-smoke-"
    ) as directory:
        config = smoke_config(
            case,
            checkpoint_dir=Path(directory),
        )
        prepared = prepare_problem(case, config)
        before = [
            parameter.detach().clone()
            for parameter in prepared.operator.network.parameters()
        ]
        training = train_prepared_problem(prepared)
        after = list(prepared.operator.network.parameters())
        changed = any(
            not torch.equal(left, right.detach())
            for left, right in zip(before, after)
        )

        best_path = Path(directory) / "best.pt"
        last_path = Path(directory) / "last.pt"
        best_payload = torch.load(
            best_path,
            map_location=prepared.device,
            weights_only=False,
        )
        last_payload = torch.load(
            last_path,
            map_location=prepared.device,
            weights_only=False,
        )
        required = {
            "schema_version",
            "stage",
            "dimension",
            "case",
            "domain",
            "material",
            "environment",
            "rng_state",
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "options",
            "history",
            "metrics",
        }
        fields_present = required <= set(best_payload)
        probe = torch.full(
            (2, case.dimension),
            0.25,
            dtype=prepared.dtype,
            device=prepared.device,
        )
        expected = prepared.operator.network(probe).detach().clone()
        reloaded = prepare_problem(
            case,
            replace(config, checkpoint_dir=None),
        )
        reloaded.operator.network.load_state_dict(
            best_payload["model_state_dict"]
        )
        actual = reloaded.operator.network(probe).detach()
        prediction_matches = bool(
            torch.allclose(expected, actual, rtol=0.0, atol=0.0)
        )
        passed = (
            training.history["epoch"] == [1]
            and changed
            and best_path.is_file()
            and last_path.is_file()
            and fields_present
            and best_payload["schema_version"] == contract.SCHEMA_VERSION
            and best_payload["stage"] == contract.STAGE
            and best_payload["dimension"] == case.dimension
            and best_payload["case"] == case.name
            and last_payload["schema_version"] == contract.SCHEMA_VERSION
            and last_payload["stage"] == contract.STAGE
            and last_payload["epoch"] == 1
            and prediction_matches
            and report.history_is_finite(training.history)
        )
        add_check(
            result,
            "one_update_and_checkpoints",
            passed,
            metrics={
                "recorded_epochs": training.history["epoch"],
                "parameters_changed": changed,
                "best_checkpoint_created": best_path.is_file(),
                "last_checkpoint_created": last_path.is_file(),
                "checkpoint_fields_present": fields_present,
                "checkpoint_schema_version": best_payload["schema_version"],
                "checkpoint_prediction_matches": prediction_matches,
                "finite_losses": report.history_is_finite(training.history),
            },
        )


def validate_cli_smoke(
    case: ElasticityCase,
    result: dict[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f"soptx-pinn-{case.dimension}d-cli-"
    ) as directory:
        summary = Path(directory) / "summary.json"
        mesh_size = "4" if case.dimension == 2 else "3"
        command = [
            sys.executable,
            str(layout.RUN_SCRIPT),
            "--dim",
            str(case.dimension),
            "--epochs",
            "1",
            "--hidden-size",
            "4",
            "--npde",
            "4",
            "--nbc",
            "2",
            "--nval-pde",
            "4",
            "--nval-bc",
            "2",
            "--log-interval",
            "1",
            "--mesh-size",
            mesh_size,
            "--summary",
            str(summary),
            "--log-level",
            "WARNING",
            "--no-show",
        ]
        completed = subprocess.run(
            command,
            cwd=layout.REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        payload = (
            json.loads(summary.read_text(encoding="utf-8"))
            if summary.is_file()
            else {}
        )
        passed = (
            completed.returncode == 0
            and payload.get("schema_version") == contract.SCHEMA_VERSION
            and payload.get("stage") == contract.STAGE
            and payload.get("local_passed") is True
        )
        add_check(
            result,
            "cli_smoke",
            passed,
            metrics={
                "returncode": completed.returncode,
                "summary_created": summary.is_file(),
                "schema_version": payload.get("schema_version"),
                "local_passed": payload.get("local_passed"),
            },
            detail=(
                ""
                if passed
                else (completed.stderr.strip() or completed.stdout.strip())
            ),
        )


def validate_training_baseline(
    case: ElasticityCase,
    result: dict[str, Any],
    *,
    epochs: int,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f"soptx-pinn-{case.dimension}d-baseline-"
    ) as directory:
        config = replace(
            contract.RunConfig(),
            epochs=epochs,
            checkpoint_dir=Path(directory),
            device="cpu",
            dtype="float64",
        )
        result["training_config"] = config.evidence_dict()
        result["official_baseline"] = contract.is_official_baseline(config)

        start = time.perf_counter()
        execution = execute(
            case,
            config,
            command=["validate.py", "--dim", str(case.dimension)],
        )
        elapsed = time.perf_counter() - start
        prepared = execution.prepared
        training = execution.training

        restore_best_state(prepared, training)
        l2 = relative_l2_metrics(
            prepared.operator.network,
            case,
            prepared.diagnostic_mesh,
            dtype=prepared.dtype,
            device=prepared.device,
        )
        interior, boundary = fixed_diagnostic_points(
            case,
            dtype=prepared.dtype,
            device=prepared.device,
        )
        equilibrium = prepared.operator.equilibrium_residual(
            interior,
            create_graph=False,
        )
        with torch.no_grad():
            boundary_error = (
                prepared.operator.predict(boundary)
                - case.problem.dirichlet_bc(boundary)
            )

        history = training.history
        equilibrium_rms = float(
            torch.sqrt(torch.mean(equilibrium**2)).item()
        )
        boundary_max = float(torch.max(torch.abs(boundary_error)).item())
        first_validation = float(history["validation_loss"][0])
        best_validation = float(training.best_validation_loss)
        final_validation = float(history["validation_loss"][-1])
        all_finite = report.history_is_finite(history)
        improved = best_validation < first_validation
        if case.dimension == 2:
            validation_passed = (
                best_validation
                <= contract.BEST_VALIDATION_LOSS_MAX_2D
                and improved
            )
            accuracy_passed = (
                l2["relative_combined"]
                <= contract.RELATIVE_DISPLACEMENT_L2_MAX_2D
            )
        else:
            validation_passed = improved
            accuracy_passed = (
                l2["relative_combined"]
                < contract.RELATIVE_DISPLACEMENT_L2_MAX_3D
            )

        passed = (
            all_finite
            and validation_passed
            and accuracy_passed
            and execution.payload["local_passed"]
        )
        add_check(
            result,
            "training_baseline",
            passed,
            metrics={
                "performed": True,
                "elapsed_seconds": elapsed,
                "recorded_updates": history["epoch"],
                "first_validation_loss": first_validation,
                "best_validation_loss": best_validation,
                "final_validation_loss": final_validation,
                "validation_improved": improved,
                "validation_gate_passed": validation_passed,
                "relative_l2_gate_passed": accuracy_passed,
                "best_checkpoint_epoch": training.best_epoch,
                "best_checkpoint_displacement_l2": l2,
                "fixed_equilibrium_residual_rms": equilibrium_rms,
                "fixed_boundary_error_max_abs": boundary_max,
                "all_history_values_finite": all_finite,
                "run_local_gates": execution.payload["local_gates"],
                "history": history,
            },
        )


def dimension_result(
    dimension: int,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dimension": dimension,
        "status": "running",
        "thresholds": contract.validation_thresholds(dimension),
        "checks": {},
        "failures": [],
    }
    case = create_case(dimension)
    result["case"] = {
        "name": case.name,
        "pde_class": case.problem.__class__.__name__,
        "domain": list(case.domain),
        "material": case.material.as_dict(),
    }
    validate_constructor(case, result)
    validate_exact_solution(case, result)
    validate_failure_guards(case, result)
    validate_one_update(case, result)
    validate_cli_smoke(case, result)
    epochs = (
        contract.RunConfig.epochs
        if arguments.epochs is None
        else arguments.epochs
    )
    validate_training_baseline(
        case,
        result,
        epochs=epochs,
    )
    result["status"] = "failed" if result["failures"] else "passed"
    return result


def dimension_payload(
    record: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "environment": environment,
        **record,
    }


def run_validation(arguments: argparse.Namespace) -> dict[str, Any]:
    bm.set_backend("pytorch")
    environment = report.environment_record()
    result: dict[str, Any] = {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "scope": "2d-plane-strain-and-3d-all-dirichlet-elasticity-pinn",
        "status": "running",
        "environment": environment,
        "selected_dimensions": list(selected_dimensions(arguments.dim)),
        "dimensions": {},
        "failures": [],
    }
    for dimension in selected_dimensions(arguments.dim):
        try:
            record = dimension_result(dimension, arguments)
        except BaseException as error:  # noqa: BLE001 - preserve other dims.
            record = {
                "dimension": dimension,
                "status": "error",
                "checks": {},
                "failures": [f"{type(error).__name__}: {error}"],
                "traceback": traceback.format_exc(),
            }
        result["dimensions"][str(dimension)] = record
        result["failures"].extend(
            f"{dimension}D: {failure}"
            for failure in record.get("failures", [])
        )
        report.write_json(
            layout.validation_summary_path(dimension),
            dimension_payload(record, environment),
        )

    statuses = {
        record["status"] for record in result["dimensions"].values()
    }
    result["status"] = (
        "passed"
        if statuses == {"passed"} and not result["failures"]
        else "failed"
    )
    if arguments.dim == "all":
        report.write_json(
            layout.aggregate_validation_path(arguments.dim),
            result,
        )
    return result


def print_result(result: dict[str, Any]) -> None:
    environment = result.get("environment", {})
    print(
        "environment: "
        f"python={environment.get('python')}, "
        f"torch={environment.get('torch')}, "
        f"cuda_available={environment.get('cuda_available')}, "
        f"git={environment.get('git_revision')}, "
        f"dirty={environment.get('git_dirty')}"
    )
    for dimension, record in result.get("dimensions", {}).items():
        print(f"{dimension}D validation gates:")
        for name, check in record.get("checks", {}).items():
            marker = "PASS" if check.get("passed") else "FAIL"
            print(f"  [{marker}] {name}")
            metrics = check.get("metrics", {})
            displacement_l2 = metrics.get(
                "best_checkpoint_displacement_l2",
                {},
            )
            for key in (
                "displacement_gradient_max_abs",
                "equilibrium_residual_max_abs",
                "dirichlet_residual_max_abs",
                "first_validation_loss",
                "best_validation_loss",
                "fixed_equilibrium_residual_rms",
                "fixed_boundary_error_max_abs",
                "elapsed_seconds",
            ):
                if key in metrics:
                    print(f"    {key}: {metrics[key]}")
            if "relative_combined" in displacement_l2:
                print(
                    "    best_checkpoint_relative_l2: "
                    f"{displacement_l2['relative_combined']}"
                )
        print(f"  status: {record.get('status')}")
        for failure in record.get("failures", []):
            print(f"  failure: {failure}")
    print(f"validation status: {result['status']}")
    print(f"outputs written below {layout.OUTPUT_DIR}")


def main(args: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(args)
    try:
        result = run_validation(arguments)
    except BaseException as error:  # noqa: BLE001 - always report failure.
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "stage": contract.STAGE,
            "status": "error",
            "environment": report.environment_record(),
            "dimensions": {},
            "failures": [f"{type(error).__name__}: {error}"],
            "traceback": traceback.format_exc(),
        }
        try:
            report.write_json(
                layout.aggregate_validation_path(arguments.dim),
                result,
            )
        except OSError:
            pass
    print_result(result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
