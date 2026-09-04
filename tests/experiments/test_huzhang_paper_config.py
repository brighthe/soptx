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
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "huzhang_topopt_paper"

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
