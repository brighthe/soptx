"""子结构性能测量的进程隔离与统计契约测试, 不启动求解或真实子进程."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path

import pytest


#: 仓库根目录; Worker 必须以此为工作目录启动, 否则 ``-m`` 解析不到实验包.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
#: Worker 的模块路径, 与 ``-m`` 参数逐字对应.
WORKER_MODULE = "experiments.analysis_capability_substructure._cost_measurement"


@pytest.fixture
def process_module():
    """加载只负责峰值内存、统计与环境采集的轻量模块."""
    return importlib.import_module(
        "experiments.analysis_capability_substructure._performance_process"
    )


@pytest.fixture
def cost_module():
    """加载承载 Worker 进程调度的测量模块, 各测试用假进程替代真实启动."""
    return importlib.import_module(WORKER_MODULE)


def worker_request(route="fa"):
    """给出用于验证进程协议的最小二维请求."""
    return {
        "route": route, "dim": 2, "n_sub": [2, 2],
        "n_fine": [2, 2], "density_mode": "cell",
    }


def worker_record(status="PASS", phase="analysis"):
    """给出 Worker 落盘记录的最小骨架."""
    return {"status": status, "phase": phase, "route": "fa", "pid": 1234}


def run_worker(cost_module, tmp_path, *, monitor=False, label="cost[fa]", route="fa"):
    """以固定参数调用被测函数, 收敛各测试的样板."""
    return cost_module._worker_run(
        worker_request(route), tmp_path / "sample",
        monitor=monitor, monitor_interval=0.5, label=label,
    )


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


def test_timing_statistics_uses_linear_quantiles(process_module):
    """重复测量的四分位点采用 NumPy 默认线性插值, 不做任何裁剪."""
    statistics = process_module.timing_statistics([1.0, 2.0, 3.0, 4.0])
    assert statistics == {
        "median": 2.5, "q25": 1.75, "q75": 3.25, "min": 1.0, "max": 4.0,
    }


def test_worker_run_uses_fresh_interpreter_and_isolated_streams(
    cost_module, monkeypatch, tmp_path,
):
    """独立解释器按 argv 启动, 环境为拷贝, 输出各自落盘."""
    calls = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **options):
            calls.append((command, options))
            Path(command[-1]).write_text(
                json.dumps(worker_record()), encoding="utf-8"
            )

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(cost_module.subprocess, "Popen", FakeProcess)
    record = run_worker(cost_module, tmp_path)

    assert len(calls) == 1
    command, options = calls[0]
    directory = tmp_path / "sample"
    assert command[:4] == [
        cost_module.sys.executable, "-m", WORKER_MODULE, "--worker",
    ]
    assert Path(command[4]) == (directory / "request.json").resolve()
    assert Path(command[5]) == (directory / "record.json").resolve()
    assert json.loads((directory / "request.json").read_text(encoding="utf-8")) \
        == worker_request()
    assert not options.get("shell", False)
    assert Path(options["cwd"]) == REPOSITORY_ROOT
    assert options["env"] == dict(os.environ)
    assert options["env"] is not os.environ
    assert Path(options["stdout"].name) == directory / "stdout.log"
    assert Path(options["stderr"].name) == directory / "stderr.log"

    assert record["status"] == "PASS"
    assert record["worker_returncode"] == 0
    assert record["worker_elapsed_seconds"] > 0
    assert "monitor_observed" not in record
    # 退出码与耗时须回写证据文件, 而不是只留在返回值里.
    assert json.loads((directory / "record.json").read_text(encoding="utf-8")) == record


def test_worker_run_refuses_to_reuse_sample_directory(cost_module, tmp_path):
    """样本目录必须是新建的, 不能覆盖既有证据."""
    (tmp_path / "sample").mkdir()
    with pytest.raises(FileExistsError):
        run_worker(cost_module, tmp_path)


@pytest.mark.parametrize("exit_code", [-9, 2])
def test_worker_run_reports_label_exit_code_and_stderr(
    cost_module, monkeypatch, tmp_path, exit_code,
):
    """失败直接报出标签、退出码与子进程错误输出, 不伪造 PASS."""
    class FailedProcess:
        returncode = exit_code

        def __init__(self, command, **options):
            options["stderr"].write("worker failed")

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(cost_module.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError) as error:
        run_worker(cost_module, tmp_path, label="cost[full_trace]", route="full_trace")

    message = str(error.value)
    assert "cost[full_trace]" in message
    assert str(exit_code) in message
    assert "worker failed" in message
    # 未落盘记录时按缺失记录归档, 而不是当作成功.
    record = json.loads(
        (tmp_path / "sample" / "record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "FAILED"
    assert record["phase"] == "worker_no_record"
    assert record["error"]["type"] == "MissingRecord"


def test_worker_run_marks_interrupted_worker_as_failed(
    cost_module, monkeypatch, tmp_path,
):
    """Worker 中途退出时把停在 RUNNING 的记录判为失败并保留所处阶段."""
    class FailedProcess:
        returncode = 1

        def __init__(self, command, **options):
            Path(command[-1]).write_text(
                json.dumps(worker_record(status="RUNNING", phase="assembly")),
                encoding="utf-8",
            )

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(cost_module.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="assembly"):
        run_worker(cost_module, tmp_path)

    record = json.loads(
        (tmp_path / "sample" / "record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "FAILED"
    assert record["phase"] == "assembly"
    assert record["error"]["type"] == "WorkerExit"


def test_worker_run_rejects_non_pass_record_on_clean_exit(
    cost_module, monkeypatch, tmp_path,
):
    """退出码为零但记录未通过验收时同样报错, 验收以记录为准."""
    class CleanButFailedProcess:
        returncode = 0

        def __init__(self, command, **options):
            options["stderr"].write("residual 超限")
            Path(command[-1]).write_text(
                json.dumps(worker_record(status="FAILED", phase="validation")),
                encoding="utf-8",
            )

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(cost_module.subprocess, "Popen", CleanButFailedProcess)
    with pytest.raises(RuntimeError, match="residual 超限"):
        run_worker(cost_module, tmp_path)


def test_worker_run_interrupt_terminates_and_reaps_child(
    cost_module, monkeypatch, tmp_path,
):
    """用户中断时终止并回收正在运行的子进程, 中断本身继续上抛."""
    events = []

    class InterruptedProcess:
        returncode = None

        def __init__(self, command, **options):
            pass

        def wait(self, timeout=None):
            if not events:
                events.append("interrupted")
                raise KeyboardInterrupt
            events.append("reaped")
            self.returncode = -15
            return -15

        def terminate(self):
            events.append("terminated")

        def kill(self):
            events.append("killed")

    monkeypatch.setattr(cost_module.subprocess, "Popen", InterruptedProcess)
    with pytest.raises(KeyboardInterrupt):
        run_worker(cost_module, tmp_path)

    assert "terminated" in events and "reaped" in events
    assert events.index("terminated") < events.index("reaped")
    assert "killed" not in events


def test_worker_run_kills_child_that_ignores_terminate(
    cost_module, monkeypatch, tmp_path,
):
    """终止超时后必须强杀并回收, 不能留下孤儿进程."""
    events = []

    class StubbornProcess:
        returncode = None

        def __init__(self, command, **options):
            pass

        def wait(self, timeout=None):
            if not events:
                events.append("interrupted")
                raise KeyboardInterrupt
            if timeout is not None and "killed" not in events:
                events.append("timeout")
                raise cost_module.subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
            events.append("reaped")
            return -9

        def terminate(self):
            events.append("terminated")

        def kill(self):
            events.append("killed")

    monkeypatch.setattr(cost_module.subprocess, "Popen", StubbornProcess)
    with pytest.raises(KeyboardInterrupt):
        run_worker(cost_module, tmp_path)

    assert events == ["interrupted", "terminated", "timeout", "killed", "reaped"]


def test_worker_run_records_monitor_observation(cost_module, monkeypatch, tmp_path):
    """开启监控时由调度器给出退出码, 观测量原样并入记录."""
    @dataclass
    class FakeStats:
        peak_rss_bytes: int = 4096
        samples: int = 3

    observed = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **options):
            Path(command[-1]).write_text(
                json.dumps(worker_record()), encoding="utf-8"
            )

        def wait(self, timeout=None):
            raise AssertionError("开启监控时不应直接 wait")

    def fake_wait(process, label, started, interval):
        observed.append((label, interval))
        return 0, FakeStats()

    monkeypatch.setattr(cost_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(cost_module, "_wait_with_monitor", fake_wait)
    record = run_worker(cost_module, tmp_path, monitor=True)

    assert observed == [("cost[fa]", 0.5)]
    assert record["monitor_observed"] == {"peak_rss_bytes": 4096, "samples": 3}
