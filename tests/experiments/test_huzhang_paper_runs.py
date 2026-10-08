# -*- coding: utf-8 -*-
"""Hu--Zhang 投稿实验各入口之间约定的回归测试.

三个 run 脚本 (``run_fixed_fixed.py`` / ``run_bearing.py`` / ``run_cantilever_stress.py``)
各自持有参数常量与运行目录命名; 再分析、探针、导出与成图模块按这些约定读取产物.
这里只核对约定是否一致, 不做有限元求解 (组装与数值一致性由各模块的冻结再分析自检保证).
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "paper_topopt_huzhang"

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import config  # noqa: E402
import plot  # noqa: E402
import run_bearing  # noqa: E402
import run_cantilever_stress  # noqa: E402
import run_fixed_fixed  # noqa: E402
from analysis import (  # noqa: E402
    bearing_reanalysis,
    compliance_reanalysis,
    discretization_probe,
    stress_metrics,
)


def _module_literal(path: Path, name: str):
    """静态读取模块顶层字面量常量, 不导入模块 (成图模块会拉起 matplotlib)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{path.name} 中没有常量 {name}")


def test_stress_run_label_matches_existing_result_directories():
    """应力算例目录名沿用旧标签规则, 已入库的运行目录因此无需改名."""
    assert run_cantilever_stress.run_label("huzhang", 2) == (
        "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5"
        "__order-2__solid_thr-0.5"
    )


def test_stress_consumers_use_run_script_labels():
    """探针默认构型、图 5.9 的运行目录与 run 脚本的命名一致."""
    expected = tuple(
        run_cantilever_stress.run_label(method, order)
        for method in run_cantilever_stress.METHODS
        for order in run_cantilever_stress.ORDERS
    )
    assert discretization_probe.DEFAULT_DESIGNS == expected
    assert stress_metrics.OUT == config.OUTPUT_DIR / run_cantilever_stress.CASE_ID
    required = _module_literal(EXPERIMENT_ROOT / "plots" / "stress_cubic_convergence.py", "REQUIRED_RUNS")
    assert required[:2] == (
        run_cantilever_stress.run_label("lfem", 3),
        run_cantilever_stress.run_label("huzhang", 3),
    )


def test_stress_al_options_follow_script_constants():
    """AL 目标与优化器共用的选项逐项取自脚本常量."""
    options = dataclasses.asdict(run_cantilever_stress.al_options())
    module = run_cantilever_stress
    expected = {
        "change_tolerance": module.CHANGE_TOLERANCE,
        "stress_tolerance": module.STRESS_TOLERANCE,
        "hold_steps": module.HOLD_STEPS,
        "max_al_iterations": module.MAX_AL_ITERATIONS,
        "mma_iters_per_al": module.MMA_ITERS_PER_AL,
        "mu_0": module.MU_0,
        "mu_max": module.MU_MAX,
        "lambda_max": module.LAMBDA_MAX,
        "move_limit": module.MOVE_LIMIT,
        "change_measure": module.CHANGE_MEASURE,
        "acceptance_solid_threshold": module.ACCEPTANCE_SOLID_THRESHOLD,
    }
    assert {key: options[key] for key in expected} == expected
    assert discretization_probe.DELTA_G == module.STRESS_TOLERANCE


def test_reanalysis_modules_cover_run_script_combinations():
    """表 5.3 / 5.4 的再分析覆盖 run 脚本跑出的全部组合."""
    assert compliance_reanalysis.CASE == run_fixed_fixed.CASE_ID
    assert compliance_reanalysis.DISCRETIZATIONS == tuple(
        (method, order) for method in run_fixed_fixed.METHODS for order in run_fixed_fixed.ORDERS
    )
    assert bearing_reanalysis.CASE == run_bearing.CASE_ID
    assert bearing_reanalysis.GROUPS == tuple(run_bearing.GROUPS)
    assert bearing_reanalysis.DESIGNS == run_bearing.RUNS


def test_plot_cases_point_to_existing_run_scripts():
    """每件成图产物的来源算例都有对应的 run 脚本可提示补跑."""
    for case in plot.discover_cases().values():
        for source in case.source_cases:
            script = plot.RUN_SCRIPTS[source]
            assert (EXPERIMENT_ROOT / script).is_file(), script
