from dataclasses import replace

import torch

from fealpy.backend import bm

import contract
from cases import create_case
import report
import solve
from solve import (
    prepare_problem,
    restore_best_state,
    train_prepared_problem,
)


def small_config(tmp_path=None):
    return contract.RunConfig(
        epochs=1,
        hidden_size=(4,),
        npde=4,
        nbc=2,
        nval_pde=4,
        nval_bc=2,
        log_interval=1,
        seed=7,
        diagnostic_mesh_size=3,
        checkpoint_dir=tmp_path,
        log_level="WARNING",
    )


def test_one_update_records_post_update_best_and_final_state(tmp_path):
    bm.set_backend("pytorch")
    case = create_case(2)
    prepared = prepare_problem(case, small_config(tmp_path))
    before = {
        name: value.detach().clone()
        for name, value in prepared.operator.network.state_dict().items()
    }
    result = train_prepared_problem(prepared)
    final = prepared.operator.network.state_dict()

    assert result.history["epoch"] == [1]
    assert result.best_epoch == 1
    assert report.history_is_finite(result.history)
    assert any(not torch.equal(before[name], final[name]) for name in final)
    assert all(
        torch.equal(final[name].cpu(), result.best_model_state_dict[name])
        for name in final
    )
    assert result.last_metrics == result.best_metrics


def test_diagnostics_are_recorded_after_the_parameter_update(
    tmp_path,
    monkeypatch,
):
    bm.set_backend("pytorch")
    case = create_case(2)
    config = replace(
        small_config(tmp_path),
        step_size=1,
        gamma=0.5,
    )
    prepared = prepare_problem(case, config)
    before = {
        name: value.detach().clone()
        for name, value in prepared.operator.network.state_dict().items()
    }
    observed = {}
    original = solve._record_diagnostics

    def record_state(problem, *args, **kwargs):
        observed.update(
            {
                name: value.detach().clone()
                for name, value in problem.operator.network.state_dict().items()
            }
        )
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(solve, "_record_diagnostics", record_state)
    result = train_prepared_problem(prepared)
    final = prepared.operator.network.state_dict()

    assert any(not torch.equal(before[name], observed[name]) for name in before)
    assert all(torch.equal(observed[name], final[name]) for name in final)
    assert result.history["learning_rate"] == [config.lr]
    assert prepared.optimizer.param_groups[0]["lr"] == config.lr * config.gamma


def test_restore_best_state_is_explicit(tmp_path):
    bm.set_backend("pytorch")
    case = create_case(2)
    prepared = prepare_problem(case, small_config(tmp_path))
    result = train_prepared_problem(prepared)
    with torch.no_grad():
        for parameter in prepared.operator.network.parameters():
            parameter.add_(1.0)
    restore_best_state(prepared, result)
    assert all(
        torch.equal(
            value.detach().cpu(),
            result.best_model_state_dict[name],
        )
        for name, value in prepared.operator.network.state_dict().items()
    )


def test_checkpoint_v3_contains_full_contract(tmp_path):
    bm.set_backend("pytorch")
    case = create_case(2)
    prepared = prepare_problem(case, small_config(tmp_path))
    train_prepared_problem(prepared)
    payload = torch.load(
        tmp_path / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    last = torch.load(
        tmp_path / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema_version"] == contract.SCHEMA_VERSION
    assert payload["stage"] == contract.STAGE
    assert payload["dimension"] == 2
    assert payload["domain"] == list(case.domain)
    assert payload["material"] == case.material.as_dict()
    assert "environment" in payload
    assert "rng_state" in payload
    assert last["schema_version"] == contract.SCHEMA_VERSION
    assert last["stage"] == contract.STAGE


def test_nonbaseline_epoch_count_is_not_official():
    assert not contract.is_official_baseline(
        replace(contract.RunConfig(), epochs=1)
    )
