import json

import contract
import layout
import sync_results


def validation_payload(dimension):
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "environment": {
            "git_revision": "a" * 40,
            "git_dirty": False,
            "python": "3.12",
            "platform": "test",
            "torch": "test",
            "fealpy": "test",
            "soptx": "test",
        },
        "dimension": dimension,
        "status": "passed",
        "official_baseline": True,
        "case": {"name": f"case-{dimension}d"},
        "training_config": contract.RunConfig().evidence_dict(),
        "thresholds": contract.validation_thresholds(dimension),
        "checks": {
            "constructor_and_shape": {"passed": True, "metrics": {}},
            "unsupported_problem_guards": {"passed": True, "metrics": {}},
            "one_update_and_checkpoints": {"passed": True, "metrics": {}},
            "cli_smoke": {"passed": True, "metrics": {}},
            "manufactured_solution_consistency": {
                "passed": True,
                "metrics": {
                    "displacement_gradient_max_abs": 0.0,
                    "equilibrium_residual_max_abs": 1.0e-15,
                    "dirichlet_residual_max_abs": 0.0,
                },
            },
            "training_baseline": {
                "passed": True,
                "metrics": {
                    "first_validation_loss": 2.0,
                    "best_validation_loss": 1.0,
                    "final_validation_loss": 1.0,
                    "best_checkpoint_epoch": 100,
                    "best_checkpoint_displacement_l2": {
                        "relative_combined": 0.5,
                    },
                    "fixed_equilibrium_residual_rms": 0.1,
                    "fixed_boundary_error_max_abs": 0.1,
                    "elapsed_seconds": 1.0,
                },
            },
        },
    }


def test_build_evidence_requires_clean_baseline(tmp_path, monkeypatch):
    source = tmp_path / "validation.json"
    source.write_text(
        json.dumps(validation_payload(2)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        layout,
        "validation_summary_path",
        lambda dimension: source,
    )
    monkeypatch.setattr(
        layout,
        "relative_to_example",
        lambda path: path.name,
    )
    evidence = sync_results.build_evidence(2)
    assert evidence["dimension"] == 2
    assert evidence["environment"]["git_dirty"] is False
    assert len(evidence["source_artifact"]["sha256"]) == 64


def test_dirty_source_is_rejected(tmp_path, monkeypatch):
    payload = validation_payload(3)
    payload["environment"]["git_dirty"] = True
    source = tmp_path / "validation.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        layout,
        "validation_summary_path",
        lambda dimension: source,
    )
    try:
        sync_results.build_evidence(3)
    except sync_results.EvidenceError as error:
        assert "git_dirty=false" in str(error)
    else:
        raise AssertionError("dirty evidence source was accepted")


def test_generated_readme_block_uses_stable_markers():
    evidence = {
        "environment": {"git_revision": "a" * 40},
        "manufactured_solution": {
            "displacement_gradient_max_abs": 0.0,
            "equilibrium_residual_max_abs": 0.0,
            "dirichlet_residual_max_abs": 0.0,
        },
        "training": {
            "first_validation_loss": 2.0,
            "best_validation_loss": 1.0,
            "best_relative_displacement_l2": 0.5,
            "fixed_equilibrium_residual_rms": 0.1,
            "fixed_boundary_error_max_abs": 0.1,
            "elapsed_seconds": 1.0,
        },
    }
    block = sync_results.render_readme_block(2, evidence)
    begin, end = layout.readme_markers(2)
    assert block.startswith(begin)
    assert block.endswith(end)


def test_check_file_detects_drift(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("old\n", encoding="utf-8")
    assert not sync_results.check_file(path, "new\n")
    path.write_text("new\n", encoding="utf-8")
    assert sync_results.check_file(path, "new\n")
