"""Guards for the summary structure shared by run/report/validate/sync.

``report.build_payload`` writes the summary, ``validate.check_case`` and
``sync_results`` read it back. Before :mod:`schema` existed each side knew the
field names independently, so renaming a key failed silently on the other two
while ``SCHEMA_VERSION`` still claimed the format was unchanged. These tests
pin the three sides to one definition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import report
import schema
from cases import create_case
from contract import RunConfig
from schema import RunResult


def make_config(*, benchmark: bool = False) -> RunConfig:
    summary = Path("outputs") / "unit-test.json"
    return RunConfig(
        dimension=2,
        degree=1,
        resolution=(4, 4),
        operator_level="ea",
        benchmark=benchmark,
        max_iterations=100,
        rtol=1.0e-10,
        atol=1.0e-12,
        output_path=summary.with_suffix(".vtu"),
        summary_path=summary,
        solution_path=summary.with_suffix(".npy"),
    )


def make_result(*, referenced: bool = True) -> RunResult:
    """A result whose every gate passes, with or without serial references."""

    return RunResult(
        global_cells=32,
        global_dofs=50,
        operator={"level": "ea", "storage": "cached-element-matrices"},
        partition={"strategy": "all-cells", "ranks": []},
        solver={
            "converged": True,
            "true_absolute_residual": 1.0e-14,
            "rhs_norm": 1.0,
            "boundary_absolute_error": 0.0,
        },
        timing=None,
        error={"l2_absolute": 1.0e-3, "l2_relative": 1.0e-2},
        matvec_reference={
            "raw_relative_error": 1.0e-16,
            "dirichlet_relative_error": 1.0e-16,
            "symmetry_relative_error": 1.0e-16,
            "random_vector_energy": 1.0,
        }
        if referenced
        else None,
        explicit_solution_reference={"relative_error": 1.0e-12}
        if referenced
        else None,
    )


def test_result_fields_match_the_dataclass():
    assert set(make_result().as_payload_fields()) == set(
        schema.RESULT_FIELDS
    )


def test_payload_carries_exactly_the_declared_fields():
    """The whole point of the module: one definition, three consumers."""

    config = make_config()
    result = make_result()
    payload = report.build_payload(
        result,
        config,
        create_case(2),
        1,
        report.local_gates(result, config, 1),
    )
    assert set(payload) == set(schema.SUMMARY_TOP_LEVEL_FIELDS)
    assert schema.missing_summary_fields(payload) == []
    assert payload["schema_version"] == schema.SCHEMA_VERSION


def test_missing_summary_fields_reports_every_absent_key():
    payload = {"solver": {}}
    missing = schema.missing_summary_fields(payload)
    assert "solver" not in missing
    assert "local_gates" in missing
    assert set(missing) == set(schema.SUMMARY_TOP_LEVEL_FIELDS) - {
        "solver"
    }


def test_full_run_executes_every_gate():
    config = make_config()
    gates = report.local_gates(make_result(), config, 1)
    assert set(gates.values()) == {schema.GATE_PASSED}
    assert schema.gates_passed(gates)


def test_benchmark_run_skips_the_reference_gates():
    """A benchmark run must not claim the comparisons it never made."""

    config = make_config(benchmark=True)
    gates = report.local_gates(
        make_result(referenced=False),
        config,
        1,
    )
    skipped = {
        name
        for name, status in gates.items()
        if status == schema.GATE_SKIPPED
    }
    assert skipped == {
        "raw_matvec",
        "dirichlet_matvec",
        "operator_symmetry",
        "explicit_solution",
    }
    # The gates that did run still decide the verdict.
    assert schema.gates_passed(gates)


def test_multi_rank_run_skips_the_reference_gates():
    gates = report.local_gates(
        make_result(referenced=False),
        make_config(),
        2,
    )
    assert gates["raw_matvec"] == schema.GATE_SKIPPED
    assert gates["converged"] == schema.GATE_PASSED


def test_failed_gate_fails_the_run():
    result = make_result()
    broken = RunResult(
        **{
            **result.as_payload_fields(),
            "solver": {**result.solver, "converged": False},
        }
    )
    gates = report.local_gates(broken, make_config(), 1)
    assert gates["converged"] == schema.GATE_FAILED
    assert not schema.gates_passed(gates)


def test_all_skipped_does_not_count_as_passing():
    gates = {name: schema.GATE_SKIPPED for name in ("a", "b")}
    assert not schema.gates_passed(gates)


def test_empty_gates_do_not_count_as_passing():
    assert not schema.gates_passed({})


def test_unknown_gate_status_is_rejected():
    with pytest.raises(ValueError, match="unknown gate status"):
        schema.gates_passed({"converged": True})
