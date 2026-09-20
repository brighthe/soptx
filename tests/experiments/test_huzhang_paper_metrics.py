# -*- coding: utf-8 -*-
"""Hu--Zhang 论文实验梯度诊断的纯判定测试."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "paper_topopt_huzhang"

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from metrics import _has_adjacent_acceptable_errors  # noqa: E402


def test_gradient_scan_accepts_two_adjacent_accurate_steps():
    assert _has_adjacent_acceptable_errors([8.0e-5, 9.0e-5, 2.0e-4])
    assert _has_adjacent_acceptable_errors([2.0e-4, 9.0e-5, 8.0e-5])


def test_gradient_scan_rejects_single_or_nonadjacent_accidental_hits():
    assert not _has_adjacent_acceptable_errors([8.0e-5, 2.0e-4, 3.0e-4])
    assert not _has_adjacent_acceptable_errors([8.0e-5, 2.0e-4, 9.0e-5])
    assert not _has_adjacent_acceptable_errors([np.inf, 8.0e-5, 2.0e-4])


def test_gradient_parameters_follow_registered_case(monkeypatch):
    import metrics

    registered = {"nx": 80, "ny": 40, "interpolation_method": "simp",
                  "penalty_factor": 3.5, "mu_max": 100000.0}
    monkeypatch.setattr(metrics, "case_parameters", lambda: registered)
    assert metrics.make_params(40, 20) == {**registered, "nx": 40, "ny": 20}
    assert registered["nx"] == 80


def test_protocol_mismatch_flags_missing_safeguard_fields(tmp_path):
    """2026-09-18 之前的产物缺 lambda_max / acceptance_solid_threshold, 必须被冻结评估排除."""
    import json

    from metrics import _SUMMARY_PROTOCOL_FIELDS, _protocol_mismatches

    assert _SUMMARY_PROTOCOL_FIELDS["lambda_max"] == "lambda_max"
    assert (_SUMMARY_PROTOCOL_FIELDS["acceptance_solid_threshold"]
            == "acceptance_solid_threshold")
    parameters = {"lambda_max": 3000.0, "acceptance_solid_threshold": 0.5}

    (tmp_path / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    mismatches = _protocol_mismatches(tmp_path, parameters)
    assert "lambda_max: 缺失 != 3000.0" in mismatches
    assert "acceptance_solid_threshold: 缺失 != 0.5" in mismatches

    (tmp_path / "summary.json").write_text(
        json.dumps({"lambda_max": 3000.0, "acceptance_solid_threshold": 0.5}),
        encoding="utf-8")
    assert _protocol_mismatches(tmp_path, parameters) == []

    (tmp_path / "summary.json").write_text(
        json.dumps({"lambda_max": None, "acceptance_solid_threshold": 0.5}),
        encoding="utf-8")
    assert _protocol_mismatches(tmp_path, parameters) == ["lambda_max: None != 3000.0"]


def test_run_label_alias_round_trips_between_driver_and_metrics():
    """driver 用 solid_thr 别名写目录名, metrics 反解回 acceptance_solid_threshold."""
    import driver as paper_run
    import metrics

    assert paper_run._TAG_ALIASES == {
        alias_target: alias for alias, alias_target in metrics._TAG_ALIASES.items()
    }
    label = "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-4__solid_thr-0.5"
    known = {"load_pad_radius", "acceptance_solid_threshold", "stress_tolerance"} | set(metrics._FIXED_TAGS)
    tags = metrics._parse_run_label(label, known)
    assert tags == {
        "analyzer": "huzhang", "lfem_constraint": "apparent", "load_pad_radius": "1.5",
        "order": "4", "acceptance_solid_threshold": "0.5",
    }
    # 完整字段名写成的目录名 (手工命名) 也仍能反解.
    assert metrics._parse_run_label("order-4__acceptance_solid_threshold-0.5", known) == {
        "order": "4", "acceptance_solid_threshold": "0.5"}
