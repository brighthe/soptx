from dataclasses import replace
from pathlib import Path

import pytest

import contract


def test_default_config_is_the_official_baseline():
    config = contract.RunConfig()
    config.validate()
    assert contract.is_official_baseline(config)
    assert contract.DEFAULT_DIMENSION == 2
    assert contract.SUPPORTED_DIMENSIONS == (2, 3)


def test_artifact_only_options_do_not_change_baseline_identity(tmp_path):
    config = replace(
        contract.RunConfig(),
        checkpoint_dir=tmp_path / "checkpoints",
        summary_path=tmp_path / "summary.json",
        pbar_log=True,
        log_level="WARNING",
    )
    assert contract.is_official_baseline(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"epochs": 0}, "epochs"),
        ({"lr": 0.0}, "lr"),
        ({"weights": (0.0, 0.0)}, "weights"),
        ({"sampling_mode": "invalid"}, "sampling_mode"),
        ({"diagnostic_mesh_size": 1}, "diagnostic_mesh_size"),
        ({"log_level": "invalid"}, "logging level"),
    ],
)
def test_invalid_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        replace(contract.RunConfig(), **overrides).validate()


def test_paths_are_serialized_as_strings():
    config = replace(
        contract.RunConfig(),
        checkpoint_dir=Path("checkpoints"),
        summary_path=Path("summary.json"),
    )
    payload = config.as_dict()
    assert payload["checkpoint_dir"] == "checkpoints"
    assert payload["summary_path"] == "summary.json"
    assert "checkpoint_dir" not in config.evidence_dict()
    assert "summary_path" not in config.evidence_dict()
