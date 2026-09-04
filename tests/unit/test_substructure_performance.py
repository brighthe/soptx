"""子结构性能测量的进程隔离与统计契约测试, 不启动求解或真实子进程."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest


@pytest.fixture
def process_module():
    """加载只负责进程调度和结果传输的轻量模块."""
    return importlib.import_module(
        "examples.substructure_elasticity._performance_process"
    )


@pytest.fixture
def comparison():
    """加载数值模块, 各测试用假对象替代有限元装配和求解."""
    return importlib.import_module("examples.substructure_elasticity._comparison")


def worker_config(route="fa"):
    """给出用于验证进程协议的最小二维配置."""
    return {
        "route": route, "dim": 2, "n_sub": [2, 2],
        "n_fine": [2, 2], "density_mode": "cell",
    }


def test_peak_rss_reads_linux_vmhwm_in_bytes(process_module, monkeypatch):
    """使用 exec 后本进程的 VmHWM, 将 Linux kB 换算为字节."""
    monkeypatch.setattr(process_module.sys, "platform", "linux")

    def read_status(path, *args, **kwargs):
        assert str(path) == "/proc/self/status"
        return "Name:\tpython\nVmPeak:\t9999 kB\nVmHWM:\t1234 kB\nVmRSS:\t800 kB\n"

    monkeypatch.setattr(Path, "read_text", read_status)
    assert process_module.peak_rss_bytes() == 1234 * 1024


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_peak_rss_rejects_unsupported_platform(process_module, monkeypatch, platform):
    """不能把其他平台的不同单位静默当作 Linux 峰值."""
    monkeypatch.setattr(process_module.sys, "platform", platform)
    with pytest.raises(RuntimeError):
        process_module.peak_rss_bytes()


def test_peak_rss_rejects_missing_vmhwm(process_module, monkeypatch):
    """缺失内核指标时不能写入零内存或假测量值."""
    monkeypatch.setattr(process_module.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "VmRSS:\t800 kB\n")
    with pytest.raises(RuntimeError):
        process_module.peak_rss_bytes()


def test_run_worker_uses_fresh_interpreter_and_does_not_load_displacement(
    process_module, monkeypatch, tmp_path,
):
    """独立解释器按 argv 启动, 返回小记录但不在父进程提前载入位移."""
    calls = []
    expected = {"route": "fa", "pid": 1234, "memory_peak_rss_bytes": 1024}

    class FakeProcess:
        returncode = 0

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            assert json.loads(Path(argv[-3]).read_text(encoding="utf-8")) == worker_config()
            Path(argv[-2]).write_text(json.dumps(expected), encoding="utf-8")
            np.save(argv[-1], np.array([1.0, 0.0]))

        def communicate(self, *args, **kwargs):
            return "", ""

    monkeypatch.setattr(process_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("worker 返回前不应加载位移")
    )
    record = process_module.run_worker(worker_config(), tmp_path / "fa")
    assert record == {**expected, "worker_stdout": "", "worker_stderr": ""}
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv[:3] == [
        process_module.sys.executable, "-m",
        "examples.substructure_elasticity._performance_process",
    ]
    assert not options.get("shell", False)
    assert Path(options["cwd"]) == Path(__file__).resolve().parents[2]
    assert options["env"] == dict(process_module.os.environ)
    assert options["env"] is not process_module.os.environ


@pytest.mark.parametrize("exit_code", [-9, 2])
def test_run_worker_reports_route_and_exit_code(
    process_module, monkeypatch, tmp_path, exit_code,
):
    """失败直接报出路径及退出码, 不读取缺失产物或伪造 PASS."""
    class FailedProcess:
        returncode = exit_code

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, *args, **kwargs):
            return "", "worker failed"

    monkeypatch.setattr(process_module.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError) as error:
        process_module.run_worker(worker_config("full_trace"), tmp_path / "full_trace")
    message = str(error.value)
    assert "full_trace" in message
    assert str(exit_code) in message


def test_run_worker_interrupt_terminates_and_reaps_child(
    process_module, monkeypatch, tmp_path,
):
    """用户中断时终止并回收正在运行的子进程."""
    events = []

    class InterruptedProcess:
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, *args, **kwargs):
            if not events:
                events.append("interrupted")
                raise KeyboardInterrupt
            events.append("reaped")
            self.returncode = -15
            return "", ""

        def terminate(self):
            events.append("terminated")

        def wait(self, *args, **kwargs):
            events.append("reaped")
            self.returncode = -15
            return -15

        def kill(self):
            events.append("killed")

        def poll(self):
            return self.returncode

    monkeypatch.setattr(process_module.subprocess, "Popen", InterruptedProcess)
    with pytest.raises(KeyboardInterrupt):
        process_module.run_worker(worker_config(), tmp_path / "fa")
    assert "terminated" in events and "reaped" in events
    assert events.index("terminated") < events.index("reaped")


def test_timed_analysis_samples_peak_before_diagnostics(comparison, monkeypatch):
    """完整位移已恢复后读取峰值, 残差矩阵转换等诊断仍在采样之后."""
    events = []

    class FakeStiffness:
        def tocsr(self):
            return self

        def to_scipy(self):
            events.append("diagnostics")
            return np.eye(2)

    analyzer = SimpleNamespace(assemble_stiff_matrix=lambda **kwargs: FakeStiffness())
    reference = SimpleNamespace(full_mesh=object(), material=object(), total_full_dofs=2)
    monkeypatch.setattr(comparison, "make_fa_analyzer", lambda *args: analyzer)
    monkeypatch.setattr(comparison, "bm", SimpleNamespace(
        arange=np.arange, int64=np.int64, reshape=np.reshape, to_numpy=np.asarray,
    ))

    def solve(*args, **kwargs):
        events.append("solved")
        return np.array([1.0, 0.0])

    def peak_reader():
        events.append("peak")
        return 123456

    monkeypatch.setattr(comparison, "solve_interface_system", solve)
    displacement, record = comparison.timed_full_analysis(
        "fa", object(), reference, np.array([0.7]),
        np.array([1.0, 0.0]), np.array([1], dtype=np.int64),
        peak_memory_reader=peak_reader,
    )
    assert np.array_equal(displacement, [1.0, 0.0])
    assert record["memory_peak_rss_bytes"] == 123456
    assert events == ["solved", "peak", "diagnostics"]
    assert record["seconds"]["total"] > 0


def test_worker_main_serializes_only_after_peak_sampling(
    comparison, process_module, monkeypatch, tmp_path,
):
    """worker 自行准备问题, 将峰值读取器传入分析, 最后才写位移和 JSON."""
    request = tmp_path / "request.json"
    record_path = tmp_path / "record.json"
    displacement_path = tmp_path / "displacement.npy"
    request.write_text(json.dumps(worker_config()), encoding="utf-8")
    events = []
    objects = tuple(object() for _ in range(5))
    metadata = {"problem": "mock-problem"}

    def peak():
        assert not record_path.exists() and not displacement_path.exists()
        events.append("peak")
        return 123456

    def prepare(**kwargs):
        assert kwargs == {key: value for key, value in worker_config().items() if key != "route"}
        events.append("prepare")
        return (*objects, metadata, 1.25)

    def analysis(route, *args, peak_memory_reader):
        assert route == "fa" and args == objects
        events.append("analysis")
        measured = peak_memory_reader()
        return np.array([1.0, 0.0]), {"memory_peak_rss_bytes": measured}

    def environment():
        events.append("environment")
        return {"thread_environment": {"OMP_NUM_THREADS": "1"}}

    monkeypatch.setattr(process_module, "peak_rss_bytes", peak)
    monkeypatch.setattr(comparison, "prepare_performance_problem", prepare)
    monkeypatch.setattr(comparison, "timed_full_analysis", analysis)
    monkeypatch.setattr(comparison, "performance_environment", environment)
    assert process_module.worker_main([
        str(request), str(record_path), str(displacement_path),
    ]) == 0
    assert events == ["peak", "prepare", "analysis", "peak", "environment"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["route"] == "fa" and record["pid"] > 0
    assert record["memory_peak_rss_bytes"] == 123456
    assert record["preparation_seconds"] == 1.25
    assert record["problem_data"] == metadata
    assert record["environment"]["thread_environment"]["OMP_NUM_THREADS"] == "1"
    assert np.array_equal(np.load(displacement_path, allow_pickle=False), [1.0, 0.0])


def fake_problem_data():
    """生成两条路径共用的标量元数据, 不构造任何网格."""
    return {
        "dimension": "2D", "problem": "HalfMBBBeamRight2d",
        "domain": [0.0, 60.0, 0.0, 20.0],
        "n_sub": [2, 2], "n_fine": [2, 2], "total_fine": [4, 4],
        "full_dofs": 50, "cells": 16, "nodes": 25, "fixed_dofs": 1,
        "degree": 1, "backend": "numpy", "solver": "scipy",
        "density_mode": "cell", "density_sha256": "density-fingerprint",
        "load_sha256": "load-fingerprint", "fixed_dofs_sha256": "fixed-fingerprint",
        "nodes_sha256": "nodes-fingerprint",
        "density_min": 0.4, "density_max": 1.0, "density_mean": 0.7,
        "displacement_shape": [50], "displacement_dtype": "float64",
        "material": {"E": 1.0, "nu": 0.3, "hypothesis": "plane_stress", "penalty": 3.0},
        "load": {"P": -1.0, "resultant": [0.0, -1.0]},
    }


def fake_worker_record(route, pair):
    """试运行给出异常大的时间及内存, 便于发现错误纳入统计."""
    total = {"fa": [1000.0, 6.0, 10.0], "full_trace": [2000.0, 3.0, 5.0]}[route][pair]
    peak = {"fa": [1000000, 200, 400], "full_trace": [2000000, 100, 200]}[route][pair]
    preparation = {"fa": [100.0, 1.0, 3.0], "full_trace": [200.0, 2.0, 4.0]}[route][pair]
    seconds = {"setup": 0.1, "assembly": 0.2, "conditions_solve": 0.3, "total": total}
    if route == "full_trace":
        seconds.update(condensation=0.1, interface_assembly=0.1, recovery=0.1)
    return {
        "route": route, "pid": 1000 + pair * 2 + int(route == "full_trace"),
        "seconds": seconds, "memory_peak_rss_bytes": peak,
        "preparation_seconds": preparation, "compliance": 2.0,
        "system_dofs": 50 if route == "fa" else 42,
        "free_dofs": 49 if route == "fa" else 41,
        "equilibrium_relative_residual": 0.0, "support_relative_error": 0.0,
        "problem_data": fake_problem_data(),
        "environment": {"thread_environment": {}, "threadpools": None},
    }


def test_repeated_comparison_isolates_pairs_and_excludes_trial_samples(
    comparison, process_module, monkeypatch, tmp_path, capsys,
):
    """两条 worker 都结束才读位移, 试运行不纳入时间/峰值/准备统计."""
    calls, directories, loads, displacement_refs = [], [], [], []
    original_load = np.load

    def worker(config, directory):
        if len(calls) >= 2 and len(calls) % 2 == 0:
            assert all(reference() is None for reference in displacement_refs)
        pair = len(calls) // 2
        route = config["route"]
        calls.append(route)
        directory = Path(directory)
        directories.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "displacement.npy", np.ones(50))
        return fake_worker_record(route, pair)

    def load(path, *args, **kwargs):
        assert len(calls) % 2 == 0
        assert kwargs.get("allow_pickle") is False
        loads.append(len(calls))
        displacement = original_load(path, *args, **kwargs)
        displacement_refs.append(weakref.ref(displacement))
        return displacement

    monkeypatch.setattr(process_module, "run_worker", worker)
    monkeypatch.setattr(process_module, "peak_rss_bytes", lambda: 123456)
    monkeypatch.setattr(np, "load", load)
    result = comparison.run_full_trace_comparison(
        2, str(tmp_path / "evidence"), n_sub=[2, 2], n_fine=[2, 2],
        warmup=1, repeat=2, density_mode="cell",
    )
    output = capsys.readouterr().out
    assert "case         full_trace_performance" not in output
    assert output.count("comparison   full_trace 与 FA 比较") == 1
    assert "试运行 1/1  PASS" in output
    assert "测量 1/2  PASS" in output and "测量 2/2  PASS" in output
    assert "[run]" not in output and "[check]" not in output
    assert "[Q25, Q75]" not in output
    assert output.count("详细结果：") == 1
    assert calls == ["fa", "full_trace", "full_trace", "fa", "fa", "full_trace"]
    assert loads == [2, 2, 4, 4, 6, 6]
    assert all(not directory.exists() for directory in directories)
    assert len(result["warmup_samples"]) == 1 and len(result["samples"]) == 2
    assert result["schema_version"] == "full-trace-performance-v2"
    assert result["statistics_seconds"]["fa"]["total"]["median"] == 8.0
    assert result["statistics_seconds"]["full_trace"]["total"]["median"] == 4.0
    assert result["statistics_peak_rss_bytes"]["fa"]["median"] == 300
    assert result["statistics_peak_rss_bytes"]["full_trace"]["median"] == 150
    assert result["statistics_preparation_seconds"]["fa"]["median"] == 2.0
    assert result["statistics_preparation_seconds"]["full_trace"]["median"] == 3.0
    assert result["statistics_preparation_and_analysis_seconds"]["fa"]["median"] == 10.0
    assert result["statistics_preparation_and_analysis_seconds"]["full_trace"]["median"] == 7.0
    assert result["preparation_and_analysis_speedup_ratio_of_medians"] == pytest.approx(10.0 / 7.0)
    assert result["peak_memory_ratio_of_medians"] == 2.0
    assert result["peak_memory_saving_fraction"] == 0.5
    assert result["speedup_ratio_of_medians"] == 2.0
    artifacts = list((tmp_path / "evidence").glob("*.json"))
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_text(encoding="utf-8"))["schema_version"] == result["schema_version"]


@pytest.mark.parametrize("fault", ["metadata", "environment", "shape", "dtype", "nonfinite", "precision"])
def test_inconsistent_worker_result_is_rejected_and_cleans_temporary_files(
    comparison, process_module, monkeypatch, tmp_path, capsys, fault,
):
    """问题或结果不一致时中止, 清理临时位移且不产生 PASS 证据."""
    directories = []

    def worker(config, directory):
        route = config["route"]
        record = fake_worker_record(route, 1)
        displacement = np.ones(50)
        if route == "full_trace":
            if fault == "metadata":
                record["problem_data"]["fixed_dofs_sha256"] = "different-support"
            elif fault == "environment":
                record["environment"]["thread_environment"] = {"OMP_NUM_THREADS": "9"}
            elif fault == "shape":
                displacement = np.ones(49)
            elif fault == "dtype":
                displacement = displacement.astype(np.float32)
            elif fault == "nonfinite":
                displacement[0] = np.nan
            else:
                displacement[0] = 2.0
        directory = Path(directory)
        directories.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "displacement.npy", displacement)
        return record

    monkeypatch.setattr(process_module, "run_worker", worker)
    monkeypatch.setattr(process_module, "peak_rss_bytes", lambda: 123456)
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        comparison.run_full_trace_comparison(
            2, str(output_dir), n_sub=[2, 2], n_fine=[2, 2],
            warmup=0, repeat=1, density_mode="cell",
        )
    assert len(directories) == 2
    assert all(not directory.exists() for directory in directories)
    assert not output_dir.exists()
    assert "PASS" not in capsys.readouterr().out


@pytest.mark.parametrize(("speedup", "saving", "time_text", "memory_text"), [
    (2.14, 0.449, "加速 2.14 倍", "峰值内存节省 44.9%"),
    (0.5, -0.25, "耗时为 FA 的 2.00 倍", "峰值内存增加 25.0%"),
    (1.0, 0.0, "耗时比约为 1.00", "峰值内存变化约为 0.0%"),
])
def test_summary_keeps_correctness_and_phases_without_quartile_columns(
    comparison, capsys, speedup, saving, time_text, memory_text,
):
    """摘要包含试运行最大误差, 保留分项中位数, 不改动 JSON 统计."""
    from copy import deepcopy

    paths = {route: fake_worker_record(route, 1) for route in ("fa", "full_trace")}
    result = {
        "routes": ["fa", "full_trace"], "warmup": 1, "repeat": 2,
        "warmup_samples": [{
            "paths": paths,
            "relative_errors_to_fa": {
                "full_trace": {"displacement": 8e-12, "compliance": 7e-12},
            },
        }],
        "samples": [{
            "paths": paths,
            "relative_errors_to_fa": {
                "full_trace": {"displacement": 1e-12, "compliance": 2e-12},
            },
        }],
        "relative_error_tolerance": 1e-11, "consistency_tolerance": 1e-9,
        "comparisons_to_fa": {"full_trace": {
            "preparation_and_analysis_speedup_ratio_of_medians": speedup,
            "peak_memory_saving_fraction": saving,
        }},
        "statistics_seconds": {
            route: {phase: comparison.timing_statistics([value])
                    for phase, value in record["seconds"].items()}
            for route, record in paths.items()
        },
        "statistics_preparation_seconds": {
            route: comparison.timing_statistics([record["preparation_seconds"]])
            for route, record in paths.items()
        },
        "statistics_preparation_and_analysis_seconds": {
            route: comparison.timing_statistics([record["preparation_seconds"] + record["seconds"]["total"]])
            for route, record in paths.items()
        },
        "statistics_peak_rss_bytes": {
            route: comparison.timing_statistics([record["memory_peak_rss_bytes"]])
            for route, record in paths.items()
        },
    }
    original = deepcopy(result)
    comparison.print_performance_summary(result)
    output = capsys.readouterr().out
    assert result == original
    assert "8.00e-12" in output and "7.00e-12" in output
    assert "柔顺度 (最后一次)" in output
    assert "验收：PASS" in output
    assert time_text in output and memory_text in output
    assert "Q25" not in output and "Q75" not in output
    assert "系统自由度" not in output
    for label in (
        "问题准备", "初始化", "全局/局部刚度装配", "子结构内部消元",
        "接口矩阵映射与装配", "边界条件处理与直接求解", "完整位移恢复",
        "总分析时间", "准备+分析", "峰值 RSS (MiB)", "独立求解自由度",
    ):
        assert label in output
    assert output.index("正确性") < output.index("验收：PASS") < output.index("耗时 (s,")
    assert output.index("耗时 (s,") < output.index("内存与规模") < output.index("结论：")
    assert "仅测量 1 对" not in output

    result["warmup"] = 0
    result["warmup_samples"] = []
    result["repeat"] = 1
    comparison.print_performance_summary(result)
    output = capsys.readouterr().out
    assert "验收：PASS" in output
    assert "仅正式测量 1 组" in output


def test_worker_failure_cleans_pair_and_does_not_write_evidence(
    comparison, process_module, monkeypatch, tmp_path, capsys,
):
    """第二条路径失败时首条路径的临时位移也应清理, 不继续下一对."""
    directories = []

    def worker(config, directory):
        directory = Path(directory)
        directories.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if config["route"] == "full_trace":
            raise RuntimeError("full_trace 子进程失败, exit code=-9")
        np.save(directory / "displacement.npy", np.ones(50))
        return fake_worker_record("fa", 1)

    monkeypatch.setattr(process_module, "run_worker", worker)
    monkeypatch.setattr(process_module, "peak_rss_bytes", lambda: 123456)
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="exit code=-9"):
        comparison.run_full_trace_comparison(
            2, str(output_dir), n_sub=[2, 2], n_fine=[2, 2],
            warmup=0, repeat=2, density_mode="cell",
        )
    assert len(directories) == 2
    assert all(not directory.exists() for directory in directories)
    assert not output_dir.exists()
    assert "PASS" not in capsys.readouterr().out
