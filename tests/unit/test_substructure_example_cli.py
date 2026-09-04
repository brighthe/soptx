"""子结构示例独立 CLI 的轻量契约测试, 不执行有限元求解."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


PUBLIC_MODULES = (
    "verify_full_trace_convergence",
    "compare_full_trace_with_fa",
    "verify_linear_corner_consistency",
    "compare_linear_corner_with_fa",
    "compare_fa_full_trace_linear_corner",
)
COMPARISON_MODULES = PUBLIC_MODULES[1:]


def load_entry(name: str):
    return importlib.import_module(f"examples.substructure_elasticity.{name}")


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_entry_has_no_case_concept(name):
    module = load_entry(name)
    options = module.build_parser()._option_string_actions
    assert "--case" not in options
    assert "--list" not in options
    assert "--help" in options


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_entry_help_does_not_load_numerical_runner(name, monkeypatch, capsys):
    module = load_entry(name)
    monkeypatch.setattr(
        module, "load_runner",
        lambda: (_ for _ in ()).throw(AssertionError("--help 不应加载数值实现")),
    )
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "options:" in output and "--problem" in output and "--output-dir" in output


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_default_output_is_independent_of_working_directory(name, monkeypatch, tmp_path):
    module = load_entry(name)
    monkeypatch.chdir(tmp_path)
    args = module.build_parser().parse_args([])
    assert Path(args.output_dir) == Path(module.__file__).resolve().parent / "outputs"


def test_convergence_dispatch_uses_problem_to_select_dimension(monkeypatch, tmp_path):
    module = load_entry("verify_full_trace_convergence")
    calls = []
    monkeypatch.setattr(module, "load_runner", lambda: lambda *args, **kwargs: calls.append((args, kwargs)))
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


@pytest.mark.parametrize(("name", "runner_name"), (
    ("compare_full_trace_with_fa", "run_full_trace_comparison"),
    ("verify_linear_corner_consistency", "run_linear_corner_consistency"),
    ("compare_linear_corner_with_fa", "run_linear_corner_comparison"),
    ("compare_fa_full_trace_linear_corner", "run_three_path_comparison"),
))
def test_comparison_entry_loads_expected_runner(name, runner_name, monkeypatch):
    module = load_entry(name)
    sentinel = object()
    fake_module = type("FakeModule", (), {runner_name: sentinel})()
    monkeypatch.setattr(module.importlib, "import_module", lambda target: fake_module)
    assert module.load_runner() is sentinel


@pytest.mark.parametrize("name", (
    "compare_full_trace_with_fa",
    "compare_linear_corner_with_fa",
    "compare_fa_full_trace_linear_corner",
))
def test_performance_comparison_dispatch(name, monkeypatch, tmp_path):
    module = load_entry(name)
    calls = []
    monkeypatch.setattr(module, "load_runner", lambda: lambda *args, **kwargs: calls.append((args, kwargs)))
    output_dir = str(tmp_path / "results")

    assert module.main([
        "--problem", "FullMBBBeam3d",
        "--n-sub", "12", "4", "4", "--n-fine", "4", "4", "4",
        "--warmup", "0", "--repeat", "2", "--density", "uniform",
        "--output-dir", output_dir,
    ]) == 0

    assert calls == [((3, output_dir), {
        "n_sub": [12, 4, 4], "n_fine": [4, 4, 4],
        "warmup": 0, "repeat": 2, "density_mode": "uniform",
    })]
    assert not Path(output_dir).exists()


def test_linear_corner_consistency_dispatch_has_no_performance_parameters(monkeypatch, tmp_path):
    module = load_entry("verify_linear_corner_consistency")
    calls = []
    monkeypatch.setattr(module, "load_runner", lambda: lambda *args, **kwargs: calls.append((args, kwargs)))
    output_dir = str(tmp_path / "results")

    assert module.main([
        "--problem", "FullMBBBeam3d",
        "--n-sub", "6", "2", "2", "--n-fine", "4", "4", "4",
        "--density", "cell", "--output-dir", output_dir,
    ]) == 0

    assert calls == [((3, output_dir), {
        "n_sub": [6, 2, 2], "n_fine": [4, 4, 4], "density_mode": "cell",
    })]


@pytest.mark.parametrize("name", COMPARISON_MODULES)
def test_omitted_problem_preserves_2d_comparison_default(name, monkeypatch):
    module = load_entry(name)
    calls = []
    monkeypatch.setattr(module, "load_runner", lambda: lambda *args, **kwargs: calls.append((args, kwargs)))
    assert module.main([]) == 0
    assert calls[0][0][0] == 2


def test_omitted_problem_preserves_2d_convergence_default(monkeypatch):
    module = load_entry("verify_full_trace_convergence")
    calls = []
    monkeypatch.setattr(module, "load_runner", lambda: lambda *args, **kwargs: calls.append((args, kwargs)))
    assert module.main([]) == 0
    assert calls[0][1]["dim"] == 2
    assert calls[0][1]["levels"] == 4


@pytest.mark.parametrize(("name", "arguments"), (
    ("verify_full_trace_convergence", ["--case", "full_trace_convergence"]),
    ("verify_full_trace_convergence", ["--dim", "3"]),
    ("verify_full_trace_convergence", ["--problem", "FullMBBBeam3d"]),
    ("verify_full_trace_convergence", ["--levels", "1"]),
    ("verify_full_trace_convergence", ["--solve_method", "scipy"]),
    ("compare_full_trace_with_fa", ["--problem", "HarmonicPoly2D"]),
    ("compare_full_trace_with_fa", ["--deg", "1"]),
    ("verify_linear_corner_consistency", ["--warmup", "0"]),
    ("verify_linear_corner_consistency", ["--repeat", "1"]),
    ("compare_linear_corner_with_fa", ["--case", "linear_corner"]),
    ("compare_fa_full_trace_linear_corner", ["--list"]),
))
def test_invalid_or_obsolete_cli_never_loads_runner(name, arguments, monkeypatch):
    module = load_entry(name)
    monkeypatch.setattr(
        module, "load_runner",
        lambda: (_ for _ in ()).throw(AssertionError("非法参数不应加载数值实现")),
    )
    with pytest.raises(SystemExit) as error:
        module.main(arguments)
    assert error.value.code == 2


def test_unified_case_entry_was_removed():
    directory = Path(load_entry(PUBLIC_MODULES[0]).__file__).resolve().parent
    assert not (directory / "verify_exact_substructure_static_condensation.py").exists()
