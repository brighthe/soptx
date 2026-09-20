"""子结构示例独立 CLI 的轻量契约测试, 不执行有限元求解."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


PUBLIC_MODULES = (
    "verify_full_trace_convergence",
    "verify_linear_corner_consistency",
)
#: 各入口内联的数值实现名. 入口只负责解析与转发, 测试一律把实现替换为假对象,
#: 因此断言覆盖的是 CLI 契约本身, 与有限元求解无关.
RUNNERS = {
    "verify_full_trace_convergence": "run_convergence_benchmark",
    "verify_linear_corner_consistency": "run_linear_corner_consistency",
}
#: 已迁出本目录的对比类入口. FA 对比与三路对比现由
#: ``experiments/analysis_capability_substructure`` 承担, 不应在示例目录复活.
RELOCATED_ENTRIES = (
    "compare_full_trace_with_fa",
    "compare_linear_corner_with_fa",
    "compare_fa_full_trace_linear_corner",
)


def load_entry(name: str):
    return importlib.import_module(f"examples.substructure_elasticity.{name}")


def stub_runner(module, name, monkeypatch):
    """把入口内联的数值实现换成只记录调用的假对象.

    返回:
        calls: 形如 ``(args, kwargs)`` 的调用记录列表.
    """
    calls: list[tuple] = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(module, RUNNERS[name], fake)
    return calls


def forbid_runner(module, name, monkeypatch):
    """令数值实现一旦被调用即失败, 用于校验提前退出的路径."""
    def fake(*args, **kwargs):
        raise AssertionError("该路径不应进入数值实现")

    monkeypatch.setattr(module, RUNNERS[name], fake)


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_entry_has_no_case_concept(name):
    """独立入口不带算例调度参数, 一个脚本只做一件事."""
    module = load_entry(name)
    options = module.build_parser()._option_string_actions
    assert "--case" not in options
    assert "--list" not in options
    assert "--help" in options


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_entry_help_does_not_run_numerical_implementation(
    name, monkeypatch, capsys,
):
    """``--help`` 以 0 退出并列出公共选项, 不触发任何计算."""
    module = load_entry(name)
    forbid_runner(module, name, monkeypatch)
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "options:" in output and "--problem" in output and "--output-dir" in output


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_default_output_is_independent_of_working_directory(name, monkeypatch, tmp_path):
    """默认输出目录锚定脚本自身位置, 不随调用者的工作目录漂移."""
    module = load_entry(name)
    monkeypatch.chdir(tmp_path)
    args = module.build_parser().parse_args([])
    assert Path(args.output_dir) == Path(module.__file__).resolve().parent / "outputs"


def test_convergence_dispatch_uses_problem_to_select_dimension(monkeypatch, tmp_path):
    """收敛入口由 ``--problem`` 反推维度, 并原样转发其余参数."""
    name = "verify_full_trace_convergence"
    module = load_entry(name)
    calls = stub_runner(module, name, monkeypatch)
    output_dir = str(tmp_path / "results")

    assert module.main([
        "--problem", "HarmonicPoly3D", "--degree", "2", "--levels", "4",
        "--solve-method", "mumps", "--output-dir", output_dir,
    ]) == 0

    assert calls == [((), {
        "dim": 3, "model": "harmonic-poly", "degree": 2, "levels": 4,
        "output_dir": output_dir, "solve_method": "mumps",
    })]
    assert not Path(output_dir).exists()


def test_linear_corner_dispatch_forwards_grid_and_density(monkeypatch, tmp_path):
    """一致性入口转发子结构划分与密度模式, 维度同样由 ``--problem`` 决定."""
    name = "verify_linear_corner_consistency"
    module = load_entry(name)
    calls = stub_runner(module, name, monkeypatch)
    output_dir = str(tmp_path / "results")

    assert module.main([
        "--problem", "FullMBBBeam3d",
        "--n-sub", "6", "2", "2", "--n-fine", "4", "4", "4",
        "--density", "cell", "--output-dir", output_dir,
    ]) == 0

    assert calls == [((3, output_dir), {
        "n_sub": [6, 2, 2], "n_fine": [4, 4, 4], "density_mode": "cell",
    })]
    assert not Path(output_dir).exists()


def test_omitted_problem_preserves_2d_convergence_default(monkeypatch):
    """收敛入口省略 ``--problem`` 时维持二维四层默认."""
    name = "verify_full_trace_convergence"
    module = load_entry(name)
    calls = stub_runner(module, name, monkeypatch)
    assert module.main([]) == 0
    assert calls[0][1]["dim"] == 2
    assert calls[0][1]["levels"] == 4


def test_omitted_problem_preserves_2d_linear_corner_default(monkeypatch):
    """一致性入口省略 ``--problem`` 时维持二维默认, 网格留给实现补齐."""
    name = "verify_linear_corner_consistency"
    module = load_entry(name)
    calls = stub_runner(module, name, monkeypatch)
    assert module.main([]) == 0
    assert calls[0][0][0] == 2
    assert calls[0][1]["n_sub"] is None and calls[0][1]["n_fine"] is None


@pytest.mark.parametrize(("name", "arguments"), (
    ("verify_full_trace_convergence", ["--case", "full_trace_convergence"]),
    ("verify_full_trace_convergence", ["--dim", "3"]),
    ("verify_full_trace_convergence", ["--problem", "FullMBBBeam3d"]),
    ("verify_full_trace_convergence", ["--levels", "1"]),
    ("verify_full_trace_convergence", ["--solve_method", "scipy"]),
    ("verify_linear_corner_consistency", ["--problem", "HarmonicPoly2D"]),
    ("verify_linear_corner_consistency", ["--warmup", "0"]),
    ("verify_linear_corner_consistency", ["--repeat", "1"]),
    ("verify_linear_corner_consistency", ["--case", "linear_corner"]),
    ("verify_linear_corner_consistency", ["--list"]),
))
def test_invalid_or_obsolete_cli_never_runs_implementation(
    name, arguments, monkeypatch,
):
    """非法或已废弃的参数一律以 2 退出, 且不进入数值实现."""
    module = load_entry(name)
    forbid_runner(module, name, monkeypatch)
    with pytest.raises(SystemExit) as error:
        module.main(arguments)
    assert error.value.code == 2


def test_unified_case_entry_was_removed():
    """统一算例入口已拆分为独立脚本, 不得回流."""
    directory = Path(load_entry(PUBLIC_MODULES[0]).__file__).resolve().parent
    assert not (directory / "verify_exact_substructure_static_condensation.py").exists()


@pytest.mark.parametrize("name", RELOCATED_ENTRIES)
def test_comparison_entries_stay_out_of_examples(name):
    """对比类入口已迁往实验目录, 示例目录只保留自检脚本."""
    directory = Path(load_entry(PUBLIC_MODULES[0]).__file__).resolve().parent
    assert not (directory / f"{name}.py").exists()
