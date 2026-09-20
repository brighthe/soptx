"""应力优化终态诊断产物的无求解器测试."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "paper_topopt_huzhang"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
from driver import _write_stress_optimizer_state


def _pipeline():
    row = {"outer_index": 2, "accepted": False, "inner_steps": 5}
    return SimpleNamespace(
        al_objective=SimpleNamespace(lamb=np.array([[2.], [3.]]), mu=50.),
        optimizer=SimpleNamespace(
            final_design_variable=np.array([.2, .8]),
            _filter=SimpleNamespace(beta=10.),
            options=SimpleNamespace(inner_stop_rule="projected_gradient",
                inner_relative_tolerance=.1, inner_absolute_tolerance=1e-6,
                mma_iters_per_al=5),
            last_inner_diagnostics=row, outer_history=[row],
        ),
    )


def test_failed_inner_state_roundtrips_without_claiming_exact_restart(tmp_path):
    pipe = _pipeline()
    density = np.array([.1, .9])
    meta = _write_stress_optimizer_state(tmp_path, pipe, density)
    assert meta["exact_restart_supported"] is False
    assert meta["last_inner_diagnostics"]["accepted"] is False
    assert meta["inner_stop_rule"] == "projected_gradient"
    with np.load(tmp_path / meta["file"], allow_pickle=False) as state:
        np.testing.assert_array_equal(state["design"], pipe.optimizer.final_design_variable)
        np.testing.assert_array_equal(state["density"], density)
        np.testing.assert_array_equal(state["lamb"], pipe.al_objective.lamb)
        assert float(state["mu"]) == 50.
    assert json.loads((tmp_path / "outer_history.json").read_text()) == pipe.optimizer.outer_history


def test_non_al_pipeline_does_not_export_fake_state(tmp_path):
    assert _write_stress_optimizer_state(tmp_path, SimpleNamespace(), np.ones(2)) is None
    assert not list(tmp_path.iterdir())


def test_nonfinite_state_is_rejected(tmp_path):
    pipe = _pipeline()
    pipe.al_objective.lamb[0, 0] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        _write_stress_optimizer_state(tmp_path, pipe, np.ones(2))
    assert not list(tmp_path.iterdir())
