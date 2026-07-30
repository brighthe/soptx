from types import SimpleNamespace

import contract
import report
from run import config_from_arguments, parse_arguments


def test_hyphen_and_underscore_cli_aliases_match():
    canonical = parse_arguments(
        [
            "--mesh-size",
            "4",
            "--sampling-mode",
            "linspace",
            "--hidden-size",
            "8",
            "--checkpoint-dir",
            "checkpoints",
            "--no-show",
        ]
    )
    compatible = parse_arguments(
        [
            "--mesh_size",
            "4",
            "--sampling_mode",
            "linspace",
            "--hidden_size",
            "8",
            "--checkpoint_dir",
            "checkpoints",
            "--no-show",
        ]
    )
    assert config_from_arguments(canonical) == config_from_arguments(compatible)


def test_local_gates_reject_nonfinite_history():
    training = SimpleNamespace(
        history={"epoch": [1], "validation_loss": [float("nan")]},
        best_epoch=1,
        best_model_state_dict={"weight": object()},
    )
    gates = report.local_gates(training, contract.RunConfig())
    assert not gates["history_finite"]
    assert not all(gates.values())


def test_run_payload_uses_shared_schema_and_stage():
    case = SimpleNamespace(
        dimension=2,
        name="case",
        domain=(0.0, 1.0, 0.0, 1.0),
        material=SimpleNamespace(as_dict=lambda: {"hypothesis": "test"}),
    )
    training = SimpleNamespace(
        history={"epoch": [1], "validation_loss": [1.0]},
        best_epoch=1,
        best_validation_loss=1.0,
        best_metrics={"validation_loss": 1.0},
        last_metrics={"validation_loss": 1.0},
    )
    payload = report.build_run_payload(
        case,
        contract.RunConfig(),
        training,
        {"gate": True},
        command=["python", "run.py"],
        environment={"git_dirty": False},
    )
    assert payload["schema_version"] == contract.SCHEMA_VERSION
    assert payload["stage"] == contract.STAGE
    assert payload["local_passed"] is True
