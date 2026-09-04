"""PIML Route A 局部恢复与灵敏度审计的纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "piml_substructure_topopt"
)
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from sensitivity_audit import (  # noqa: E402
    audit_accepted_local_responses,
    filtered_sensitivity_metrics,
    select_audit_positions,
    summarize_audit_iterations,
)


def _audit(predicted_interior: float, exact_interior: float):
    return audit_accepted_local_responses(
        substructure_ids=[7],
        boundary_displacement=np.asarray([[1.0]]),
        predicted_interior_displacement=np.asarray([[predicted_interior]]),
        exact_interior_displacement=np.asarray([[exact_interior]]),
        density_cell=np.asarray([[0.5]]),
        interior_dofs=np.asarray([0]),
        boundary_dofs=np.asarray([1]),
        cell_to_dof=np.asarray([[0, 1]]),
        unit_cell_stiffness=np.eye(2),
        simp_penalty=3.0,
        rho_min=0.1,
        gate_metrics=[{"excess_ratio": 0.01}],
    )


def test_identical_recovery_has_zero_error_and_unit_cosine() -> None:
    exact_energy, records = _audit(1.0, 1.0)

    np.testing.assert_allclose(exact_energy, [[2.0]])
    record = records[0]
    assert record["interior_displacement_relative_l2"] == pytest.approx(0.0)
    assert record["cell_energy_relative_l2"] == pytest.approx(0.0)
    assert record["sensitivity_relative_l2"] == pytest.approx(0.0)
    assert record["sensitivity_cosine_similarity"] == pytest.approx(1.0)
    assert record["density_min"] == pytest.approx(0.5)
    assert record["density_mean"] == pytest.approx(0.5)
    assert record["density_max"] == pytest.approx(0.5)
    assert record["density_std"] == pytest.approx(0.0)


def test_perturbed_recovery_matches_quadratic_form_and_simp_derivative() -> None:
    exact_energy, records = _audit(2.0, 1.0)

    np.testing.assert_allclose(exact_energy, [[2.0]])
    # q_piml = 2^2 + 1^2 = 5, q_exact = 1^2 + 1^2 = 2.
    # dE/drho = 3 * (1 - 0.1) * 0.5^2 = 0.675.
    record = records[0]
    assert record["cell_energy_relative_l2"] == pytest.approx(1.5)
    assert record["sensitivity_relative_l2"] == pytest.approx(1.5)
    assert record["sensitivity_max_abs_error"] == pytest.approx(2.025)
    assert record["gate_metrics"] == {"excess_ratio": 0.01}


def test_audit_input_contract_rejects_inconsistent_partition() -> None:
    with pytest.raises(ValueError, match="完整不重叠分区"):
        audit_accepted_local_responses(
            substructure_ids=[0],
            boundary_displacement=np.asarray([[1.0]]),
            predicted_interior_displacement=np.asarray([[1.0]]),
            exact_interior_displacement=np.asarray([[1.0]]),
            density_cell=np.asarray([[0.5]]),
            interior_dofs=np.asarray([0]),
            boundary_dofs=np.asarray([0]),
            cell_to_dof=np.asarray([[0, 1]]),
            unit_cell_stiffness=np.eye(2),
            simp_penalty=3.0,
            rho_min=0.1,
            gate_metrics=[{}],
        )


def test_deterministic_audit_positions_respect_limit() -> None:
    np.testing.assert_array_equal(select_audit_positions(3, 5), [0, 1, 2])
    selected = select_audit_positions(10, 4)
    np.testing.assert_array_equal(selected, [0, 3, 6, 9])
    with pytest.raises(ValueError, match="正整数"):
        select_audit_positions(10, 0)


def test_filtered_and_iteration_summaries_preserve_scope() -> None:
    metrics = filtered_sensitivity_metrics(
        np.asarray([-1.0, -2.0]),
        np.asarray([-1.0, -2.0]),
    )
    assert metrics["relative_l2"] == pytest.approx(0.0)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)

    _, records = _audit(1.0, 1.0)
    summary = summarize_audit_iterations(
        [
            {
                "substructures": records,
                "filtered_sensitivity": metrics,
            },
            {
                "substructures": [],
                "filtered_sensitivity": None,
            },
        ]
    )
    assert summary["iteration_count"] == 2
    assert summary["filtered_full_reference_iteration_count"] == 1
    assert summary["sensitivity_relative_l2"]["max"] == pytest.approx(0.0)
    assert summary["sensitivity_cosine_similarity"]["min"] == pytest.approx(1.0)
