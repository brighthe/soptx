# -*- coding: utf-8 -*-
"""Hu--Zhang 投稿实验配置层的回归测试.

只覆盖 ``experiments/huzhang_topopt_paper`` 里不启动有限元求解的纯配置逻辑:
``cases.toml`` 的校验、算例筛选、参数扁平化、方法/阶次展开与 ``pipeline.ASSEMBLERS``
模型注册表.
真实求解精度由 ``tests/unit`` 下的制造解与角点松弛用例负责, 这里不重复.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "paper_topopt_huzhang"

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import config as paper_config  # noqa: E402


@pytest.fixture(scope="module")
def cases():
    return paper_config.load_cases(paper_config.CASES_FILE)


def _arguments(**overrides) -> argparse.Namespace:
    defaults = {"method": "all", "order": None, "check_only": False, "full": False}
    return argparse.Namespace(**{**defaults, **overrides})


def test_cases_file_passes_validation(cases):
    identifiers = [case["id"] for case in cases]
    assert identifiers, "cases.toml 必须至少配置一个算例"
    assert len(set(identifiers)) == len(identifiers), "case id 必须唯一"
    for case in cases:
        assert case["status"] in ("ready", "planned")
        assert set(case["methods"]) <= {"lfem", "huzhang"}
        assert case["discretization"]["comparison_orders"]


def test_load_cases_rejects_duplicate_identifier(cases, tmp_path):
    text = paper_config.CASES_FILE.read_text(encoding="utf-8")
    duplicated = tmp_path / "cases.toml"
    duplicated.write_text(f'{text}\n[[cases]]\nid = "{cases[0]["id"]}"\n', encoding="utf-8")
    with pytest.raises(paper_config.ConfigurationError, match="重复"):
        paper_config.load_cases(duplicated)


def test_select_cases_requires_explicit_ready_case(cases):
    with pytest.raises(paper_config.ConfigurationError, match="--case"):
        paper_config.select_cases(cases, None)
    with pytest.raises(paper_config.ConfigurationError, match="未知 case id"):
        paper_config.select_cases(cases, ["no-such-case"])
    ready = [case["id"] for case in cases if case["status"] == "ready"]
    assert [case["id"] for case in paper_config.select_cases(cases, ["all"])] == ready


def test_flatten_parameters_merges_three_sections(cases):
    case = cases[0]
    flattened = paper_config.flatten_parameters(case)
    for section in ("discretization", "optimization"):
        assert case[section].items() <= flattened.items()
    assert case["model"]["parameters"].items() <= flattened.items()


def test_resolve_runs_defaults_to_one_combination(cases):
    case = next(c for c in cases if set(c["methods"]) == {"lfem", "huzhang"})
    orders = tuple(int(o) for o in case["discretization"]["comparison_orders"])
    smallest = min(orders)

    # 缺省 = 一个方法 x 一个阶次, 裸跑一条 case 就是一次运行
    expected_method = paper_config.default_method(tuple(case["methods"]))
    assert paper_config.resolve_runs(case, _arguments(method=None)) == [(expected_method, smallest)]
    # 显式 --method all 只放开方法, 阶次仍是缺省的单值
    assert paper_config.resolve_runs(case, _arguments()) == [
        ("lfem", smallest),
        ("huzhang", smallest),
    ]

    # --full 才展开成论文的完整对比组
    full_runs = paper_config.resolve_runs(case, _arguments(method=None, full=True))
    assert full_runs == [("lfem", o) for o in orders] + [("huzhang", o) for o in orders]

    single = paper_config.resolve_runs(case, _arguments(method="huzhang", order=[orders[0]]))
    assert single == [("huzhang", orders[0])]
    with pytest.raises(paper_config.ConfigurationError, match="未配置方法"):
        paper_config.resolve_runs(case, _arguments(method="nosuchmethod"))
    with pytest.raises(paper_config.ConfigurationError, match="比较阶次"):
        paper_config.resolve_runs(case, _arguments(order=[max(orders) + 10]))


def test_supplementary_orders_are_whitelisted_but_never_swept(cases):
    """补充专题阶次 (如半域梁的 k=1) 可显式点名跑, 但不该混进缺省与 --full."""
    case = next(
        (c for c in cases if c["discretization"].get("supplementary_orders")),
        None,
    )
    if case is None:
        pytest.skip("当前注册表没有声明 supplementary_orders 的算例")
    extra = int(case["discretization"]["supplementary_orders"][0])

    swept = {order for _, order in paper_config.resolve_runs(case, _arguments(full=True))}
    assert extra not in swept
    assert extra not in {order for _, order in paper_config.resolve_runs(case, _arguments())}
    assert paper_config.resolve_runs(case, _arguments(method="huzhang", order=[extra])) == [
        ("huzhang", extra)
    ]


# 该模型只由 convergence.py 驱动, 不经 driver.py 的模型注册表
STANDALONE_MODELS = {"MixedBoundarySinusoidalElasticity2D"}


def test_unregistered_model_is_rejected(cases):
    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    ready_models = {case["model"]["name"] for case in cases if case["status"] == "ready"}
    assert ready_models - STANDALONE_MODELS <= set(paper_pipeline.ASSEMBLERS)
    case = copy.deepcopy(cases[0])
    case["model"]["name"] = "NotARegisteredModel2d"
    with pytest.raises(paper_config.UnsupportedModelError, match="尚未注册"):
        paper_run.build_model_pipeline(case, "lfem", 2, _arguments())


def test_stress_protocol_defaults_to_unified_apparent_and_has_distinct_labels(cases):
    """统一约束与 legacy LFEM 约束必须写入不同运行目录."""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    assert config.stress_constraint_formulation == "apparent"
    assert config.kkt_diagnostics_enabled is True
    assert config.kkt_acceptance_enabled is False
    assert config.asymptote_min_distance == pytest.approx(1.0e-4)
    # 2026-09-15: epsilon 提到 1e-3 以消除滤波尾部单元的慢尾 (见 cases.toml 注)
    assert config.epsilon == pytest.approx(1.0e-3)
    # 2026-09-15: mu_max 退回论文 / PolyStress 的 1e4 (09-11 的 1e5 是对慢尾症状的误判)
    assert config.mu_max == pytest.approx(1.0e4)
    assert config.inner_stop_rule == "legacy"
    assert config.inner_relative_tolerance == pytest.approx(0.1)
    assert config.inner_absolute_tolerance == pytest.approx(1e-6)
    # 2026-09-16: 载荷引入垫片 (实体保留 + 应力豁免), 取 l/4 (见 cases.toml 注)
    assert config.load_pad_radius == pytest.approx(1.5)
    # 2026-09-18: 乘子安全阈与 C2 实体验收子集 (见 cases.toml 注)
    assert config.lambda_max == pytest.approx(3000.0)
    assert config.acceptance_solid_threshold == pytest.approx(0.5)

    apparent_label = paper_run._run_label("lfem", 2, config, {})
    vanishing_label = paper_run._run_label(
        "lfem",
        2,
        replace(config, stress_constraint_formulation="vanishing"),
        {"stress_constraint_formulation": "vanishing"},
    )
    assert "lfem_constraint-apparent" in apparent_label
    assert "lfem_constraint-vanishing" in vanishing_label
    assert apparent_label != vanishing_label


def test_load_pad_radius_tags_run_directory_and_reverts_on_zero(cases):
    """垫片半径同时改变验收区域与设计域, 非零取值必须进产物目录名, 取 0 时复原旧目录名."""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))

    changes = paper_run._override_changes(config, {"load_pad_radius": "0"}, {})
    disabled = replace(config, **changes)
    assert disabled.load_pad_radius == pytest.approx(0.0)

    pad_label = paper_run._run_label("lfem", 2, config, {})
    legacy_label = paper_run._run_label("lfem", 2, disabled, changes)
    assert "load_pad_radius-1.5" in pad_label
    assert "load_pad_radius" not in legacy_label
    # 2026-09-18 起 acceptance_solid_threshold 同样恒进目录名 (别名 solid_thr, 控制
    # Windows 路径长度), 旧名只在它也取 None 时复原
    assert legacy_label == (
        "analyzer-lfem__lfem_constraint-apparent__order-2__solid_thr-0.5")
    fully_legacy = paper_run._run_label(
        "lfem", 2, replace(disabled, acceptance_solid_threshold=None),
        {**changes, "acceptance_solid_threshold": None})
    assert fully_legacy == "analyzer-lfem__lfem_constraint-apparent__order-2"


def test_stress_config_rejects_negative_load_pad_radius(cases):
    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = {
        **paper_config.flatten_parameters(case),
        "load_pad_radius": -1.0,
    }
    with pytest.raises(ValueError, match="有限非负数"):
        paper_pipeline.build_stress_config(parameters)


def test_stress_asymptote_min_distance_flows_through_override_and_al_options(cases):
    """控制参数经 generic --override 进入 AL-MMA, 并保留在 summary 的 changes 中."""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    changes = paper_run._override_changes(
        config, {"asymptote_min_distance": "0.001"}, {}
    )
    overridden = replace(config, **changes)
    options = paper_pipeline._build_al_options(overridden)

    assert changes["asymptote_min_distance"] == pytest.approx(0.001)
    assert {name: str(value) for name, value in changes.items()} == {
        "asymptote_min_distance": "0.001"
    }
    assert options.asymptote_min_distance == pytest.approx(0.001)


def test_stress_inner_stop_controls_flow_through_override_and_al_options(cases):
    """新内层协议可经 generic --override 进入 AL-MMA，默认配置仍是 legacy。"""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    changes = paper_run._override_changes(
        config,
        {
            "inner_stop_rule": "projected_gradient",
            "inner_relative_tolerance": "0.2",
            "inner_absolute_tolerance": "2e-6",
        },
        {},
    )
    options = paper_pipeline._build_al_options(replace(config, **changes))

    assert changes == {
        "inner_stop_rule": "projected_gradient",
        "inner_relative_tolerance": pytest.approx(0.2),
        "inner_absolute_tolerance": pytest.approx(2e-6),
    }
    assert options.inner_stop_rule == "projected_gradient"
    assert options.inner_relative_tolerance == pytest.approx(0.2)
    assert options.inner_absolute_tolerance == pytest.approx(2e-6)


@pytest.mark.parametrize(
    ("method", "protocol", "expected"),
    [
        ("lfem", "apparent", "apparent"),
        ("huzhang", "apparent", "apparent"),
        ("lfem", "vanishing", "vanishing"),
        ("huzhang", "vanishing", "apparent"),
    ],
)
def test_stress_model_resolution_separates_method_from_lfem_protocol(
    cases, method, protocol, expected,
):
    """历史 LFEM 协议不应被误报为 Hu--Zhang 实际采用的模型."""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    config = replace(config, stress_constraint_formulation=protocol)
    assert paper_pipeline.resolve_stress_constraint_formulation(config, method) == expected


def test_stress_model_resolution_rejects_unknown_method_before_assembly(cases):
    """未登记方法在有限元装配前报错, 不隐式采用 Hu--Zhang 约束."""
    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = paper_config.flatten_parameters(case)
    config = paper_pipeline.build_stress_config(parameters)
    with pytest.raises(ValueError, match="不支持分析方法"):
        paper_pipeline.build_stress_analysis_pipeline(config, parameters, "unknown", 2)


def test_stress_protocol_rejects_unknown_formulation(cases):
    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = paper_config.flatten_parameters(case)
    parameters["stress_constraint_formulation"] = "unknown"
    with pytest.raises(ValueError, match="应力约束形式"):
        paper_pipeline.build_stress_config(parameters)


def test_stress_config_rejects_kkt_acceptance_without_explicit_tolerances(cases):
    """check-only 使用的 config 构造阶段也要拒绝不完整的 KKT 验收配置."""
    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = paper_config.flatten_parameters(case)
    parameters.update({
        "kkt_diagnostics_enabled": True,
        "kkt_acceptance_enabled": True,
    })
    with pytest.raises(ValueError, match="三个正的 KKT 容差"):
        paper_pipeline.build_stress_config(parameters)


@pytest.mark.parametrize(
    ("metadata", "expected", "source"),
    [
        (
            {
                "stress_constraint_formulation": "apparent",
                "lfem_stress_constraint_formulation": "vanishing",
            },
            "apparent",
            "summary.stress_constraint_formulation",
        ),
        (
            {"lfem_stress_constraint_formulation": "vanishing"},
            "vanishing",
            "summary.lfem_stress_constraint_formulation",
        ),
        (
            {"stress_constraint_type": "ApparentStressConstraint"},
            "apparent",
            "summary.stress_constraint_type",
        ),
    ],
)
def test_stress_metadata_prefers_actual_model_and_reads_legacy(
    tmp_path, metadata, expected, source,
):
    """实际模型元数据优先于对照协议, 同时保留旧产物读取能力."""
    import json

    paper_config.bootstrap_source_path()
    import metrics as paper_metrics

    (tmp_path / "summary.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert paper_metrics._summary_constraint_formulation(tmp_path) == (expected, source)


def test_stress_config_keeps_mesh_type_and_rejects_unknown_or_odd_sizes(cases):
    """应力配置不再丢弃 mesh_type: 取 cases.toml 的值, 非法值与奇数剖分当场拒绝."""
    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = paper_config.flatten_parameters(case)
    config = paper_pipeline.build_stress_config(parameters)
    assert parameters["mesh_type"] == "triangle-checkerboard"
    assert config.mesh_type == parameters["mesh_type"]

    with pytest.raises(ValueError, match="不支持的网格类型"):
        paper_pipeline.build_stress_config({**parameters, "mesh_type": "quad"})
    with pytest.raises(ValueError, match="为偶数"):
        paper_pipeline.build_stress_config({**parameters, "nx": 81})


def test_algorithm_note_reports_step_family_for_stress_case(cases):
    import pipeline as paper_pipeline
    import driver as paper_driver

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    note = paper_driver._algorithm_note(config)

    assert config.move_limit_decay == pytest.approx(1.0)
    assert "move_limit=0.15, 渐近线下限 asymptote_min_distance=0.0001" in note
    assert "衰减" not in note

    decayed = paper_pipeline.build_stress_config(
        {**paper_config.flatten_parameters(case), "move_limit_decay": "0.8"}
    )
    assert "move_limit=0.15 (末期 x0.8 衰减至 move_limit_min=0.005)" in (
        paper_driver._algorithm_note(decayed)
    )


def test_run_banner_reports_output_directory_and_file_list(tmp_path):
    """``[output]`` 回执行列出产物目录与文件清单, 且清单与落盘用的是同一份常量."""
    paper_config.bootstrap_source_path()
    import driver as paper_driver

    output = tmp_path / "outputs" / "case-x" / "analyzer-lfem__order-1"
    note = paper_driver._output_note(output, volume_minimizing=True)
    assert note.startswith(f"{output.resolve()}/ -> ")
    for name in ("summary.json", "history.json", "density_final.vtu",
                 "vtu/density_iter_NNN.vtu", "final_optimizer_state.npz",
                 "outer_history.json"):
        assert name in note
    assert "analyzer-lfem__order-1__crashed/" in note
    # 柔顺度算例不写终态诊断文件, 回执也不能列出.
    compliance_note = paper_driver._output_note(output, volume_minimizing=False)
    assert "final_optimizer_state.npz" not in compliance_note
    assert "density_final.vtu" in compliance_note
    # 清单常量必须与落盘代码引用的键一致.
    assert paper_driver.OUTPUT_FILES["summary"] == "summary.json"
    assert paper_driver.STRESS_OUTPUT_FILES["optimizer_state"] == "final_optimizer_state.npz"


def test_acceptance_solid_threshold_tags_run_directory_and_lambda_max_does_not(cases):
    """C2 验收子集改变停止准则本身, 恒进目录名; lambda_max 与 mu_max 同类只在覆盖时进."""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    label = paper_run._run_label("huzhang", 4, config, {})
    assert "solid_thr-0.5" in label
    assert "acceptance_solid_threshold" not in label
    assert "lambda_max" not in label
    # 全域口径 (None) 复原旧目录名, 与 09-17 之前的产物同名可比
    global_label = paper_run._run_label(
        "huzhang", 4, replace(config, acceptance_solid_threshold=None),
        {"acceptance_solid_threshold": None})
    assert "solid_thr" not in global_label
    # 覆盖 lambda_max 时进标签
    capped_label = paper_run._run_label(
        "huzhang", 4, replace(config, lambda_max=1000.0), {"lambda_max": 1000.0})
    assert "lambda_max-1000.0" in capped_label


def test_optional_float_fields_flow_through_override_and_al_options(cases):
    """现值为 None 的 Optional 字段经 --override 得到数值或 None, 不会退化成字符串."""
    from dataclasses import replace

    paper_config.bootstrap_source_path()
    import driver as paper_run
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = paper_config.flatten_parameters(case)
    config = paper_pipeline.build_stress_config(
        {**parameters, "lambda_max": None, "acceptance_solid_threshold": "none"})
    assert config.lambda_max is None
    assert config.acceptance_solid_threshold is None

    changes = paper_run._override_changes(
        config, {"lambda_max": "1e3", "acceptance_solid_threshold": "0.7"}, {})
    assert changes == {"lambda_max": pytest.approx(1000.0),
                       "acceptance_solid_threshold": pytest.approx(0.7)}
    options = paper_pipeline._build_al_options(replace(config, **changes))
    assert options.lambda_max == pytest.approx(1000.0)
    assert options.acceptance_solid_threshold == pytest.approx(0.7)

    registered = paper_pipeline.build_stress_config(parameters)
    changes = paper_run._override_changes(
        registered, {"lambda_max": "none", "acceptance_solid_threshold": "null"}, {})
    assert changes == {"lambda_max": None, "acceptance_solid_threshold": None}


def test_stress_config_rejects_invalid_lambda_max_and_solid_threshold(cases):
    """check-only 使用的 config 构造阶段拒绝非法的乘子安全阈与验收子集阈值."""
    paper_config.bootstrap_source_path()
    import pipeline as paper_pipeline

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    parameters = paper_config.flatten_parameters(case)
    with pytest.raises(ValueError, match="lambda_max"):
        paper_pipeline.build_stress_config({**parameters, "lambda_max": -1.0})
    with pytest.raises(ValueError, match="acceptance_solid_threshold"):
        paper_pipeline.build_stress_config(
            {**parameters, "acceptance_solid_threshold": 1.5})


def test_algorithm_note_reports_multiplier_cap_and_c2_subset(cases):
    from dataclasses import replace

    import pipeline as paper_pipeline
    import driver as paper_driver

    case = next(c for c in cases if c["id"] == "cantilever-middle-2d-stress")
    config = paper_pipeline.build_stress_config(paper_config.flatten_parameters(case))
    note = paper_driver._algorithm_note(config)
    assert "lambda_max=3000" in note
    assert "acceptance_solid_threshold=0.5" in note

    legacy = paper_driver._algorithm_note(
        replace(config, lambda_max=None, acceptance_solid_threshold=None))
    assert "乘子更新无上限" in legacy
    assert "C2 验收子集: 全域未豁免单元" in legacy
