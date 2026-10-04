"""性能测量的资源监控与环境采集, 不作为用户命令行入口.

调用方各自启动独立 Worker 进程. 父进程侧的资源采样与看板渲染 (``_wait_with_monitor``)
取自已删除的 ``experiments/_common/scheduler.py`` (51f8d2c 之前), 原样移入本模块使本实验
目录自含; 其余为本实验证据口径特有的部分: 峰值内存
采用 Linux 当前进程的 VmHWM 且字段缺失时直接报错, 不返回哨兵值; 另提供重复测量的
统计汇总与依赖版本、线程环境的采集.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence
import unicodedata

import numpy as np

__all__ = [
    "_wait_with_monitor",
    "peak_rss_bytes",
    "performance_environment",
    "timing_statistics",
]


# 以下自 experiments/_common/scheduler.py @ 51f8d2c^ 原样移入, 仅 wait_with_monitor 改名为 _wait_with_monitor.
def display_width(text: str) -> int:
    """计算包含中文字符的真实终端显示字宽 (基于 Unicode East Asian Width 规范)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text)


def pad(text: str, width: int) -> str:
    """按显示字宽向右填充空格."""
    return text + " " * max(0, width - display_width(text))


def _read_proc_status_kib(pid: int) -> tuple[int | None, int | None]:
    """读取 Linux 进程的当前 RSS 与绝对峰值 RSS."""
    values: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as file:
            for line in file:
                name = line.partition(":")[0]
                if name in ("VmRSS", "VmHWM"):
                    values[name] = int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None, None
    return values.get("VmRSS"), values.get("VmHWM")


def _read_meminfo_kib() -> tuple[int | None, int | None]:
    """读取 Linux 整机总内存与当前可用内存."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                name = line.partition(":")[0]
                if name in ("MemTotal", "MemAvailable"):
                    values[name] = int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _read_process_cpu_ticks(pid: int) -> int | None:
    """读取 Linux 进程累计消耗的用户态与内核态 CPU tick."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        return int(fields[11]) + int(fields[12])
    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        ValueError,
        IndexError,
    ):
        return None


def _format_kib(value: int | None) -> str:
    """把 KiB 格式化为适合实时看板展示的容量字符串."""
    if value is None:
        return "--"
    if value >= 2**20:
        return f"{value / 2**20:.2f} GiB"
    return f"{value / 2**10:.1f} MiB"


@dataclass
class WorkerStats:
    """父进程对独立 Worker 资源占用的跨样本观测统计.

    ``last_*`` 为进程存活时最后一次观测值 (进程退出后保留, 不清空);
    ``max_*`` / ``min_*`` 为整个生命周期内的极值. 采样间隔有限, 极值只是真实峰值的下界.
    """

    samples: int = 0
    last_rss_kib: int | None = None
    last_peak_kib: int | None = None
    max_rss_kib: int | None = None
    max_peak_kib: int | None = None
    last_available_kib: int | None = None
    min_available_kib: int | None = None
    last_cpu_percent: float | None = None

    def update(
        self,
        rss_kib: int | None,
        peak_kib: int | None,
        available_kib: int | None,
        cpu_percent: float | None,
    ) -> bool:
        """吸收一次采样; 返回本次采样时 Worker 是否仍存活 (僵尸/已退出进程读不到 VmRSS)."""
        alive = rss_kib is not None
        if alive:
            self.samples += 1
            self.last_rss_kib = rss_kib
            self.max_rss_kib = rss_kib if self.max_rss_kib is None else max(self.max_rss_kib, rss_kib)
            if peak_kib is not None:
                self.last_peak_kib = peak_kib
                self.max_peak_kib = peak_kib if self.max_peak_kib is None else max(self.max_peak_kib, peak_kib)
            if cpu_percent is not None:
                self.last_cpu_percent = cpu_percent
            if available_kib is not None:
                self.last_available_kib = available_kib
                self.min_available_kib = (
                    available_kib if self.min_available_kib is None else min(self.min_available_kib, available_kib)
                )
        return alive

    def to_mib_dict(self) -> dict[str, Any]:
        """以 MiB 为单位导出, 供失败侧车 JSON 落盘."""

        def mib(v: int | None) -> float | None:
            return None if v is None else round(v / 1024, 1)

        return {
            "samples": self.samples,
            "last_rss_MiB": mib(self.last_rss_kib),
            "last_peak_rss_MiB": mib(self.last_peak_kib),
            "max_rss_MiB": mib(self.max_rss_kib),
            "max_peak_rss_MiB": mib(self.max_peak_kib),
            "last_system_available_MiB": mib(self.last_available_kib),
            "min_system_available_MiB": mib(self.min_available_kib),
            "last_cpu_percent": None if self.last_cpu_percent is None else round(self.last_cpu_percent, 1),
        }


def _runtime_monitor_lines(
    label: str,
    pid: int,
    elapsed: float,
    stats: WorkerStats,
    alive: bool,
    available_kib: int | None,
    total_kib: int | None,
) -> list[str]:
    """构造独立 Worker 的实时资源表格; 进程退出后展示死前最后一次观测值."""
    used_percent = None
    if total_kib and available_kib is not None:
        used_percent = 100.0 * (total_kib - available_kib) / total_kib
    suffix = "" if alive else " (last seen)"
    cpu = stats.last_cpu_percent

    rows = [
        ("Case", label),
        ("Worker PID", str(pid)),
        ("Elapsed", f"{elapsed:.1f} s"),
        ("CPU", ("--" if cpu is None else f"{cpu:.1f}%") + suffix),
        ("Current RSS", _format_kib(stats.last_rss_kib) + suffix),
        ("Peak RSS (VmHWM)", _format_kib(stats.last_peak_kib) + suffix),
        ("Peak RSS (observed)", _format_kib(stats.max_rss_kib)),
        ("System Available", _format_kib(available_kib)),
        ("Min System Available", _format_kib(stats.min_available_kib)),
        (
            "System Memory Used",
            "--" if used_percent is None else f"{used_percent:.1f}%",
        ),
    ]
    key_width = max(display_width(key) for key, _ in rows)
    value_width = max(display_width(value) for _, value in rows)
    inner_width = key_width + value_width + 5
    title = " Runtime Monitor " if alive else " Runtime Monitor (worker exited) "
    lines = [f"┌─{title}{'─' * (inner_width - display_width(title) - 1)}┐"]
    for key, value in rows:
        lines.append(f"│ {pad(key, key_width)} : {pad(value, value_width)} │")
    lines.append(f"└{'─' * inner_width}┘")
    return lines


def _wait_with_monitor(
    process: subprocess.Popen[str],
    label: str,
    started: float,
    interval: float,
) -> tuple[int, WorkerStats]:
    """在父进程中采样并原位刷新 Worker 的资源占用表格, 返回退出码与跨样本观测统计."""
    interactive = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    clock_ticks = os.sysconf("SC_CLK_TCK")
    previous_ticks: int | None = None
    previous_time: float | None = None
    rendered_lines = 0
    next_plain_update = 0.0
    stats = WorkerStats()

    def render(alive: bool, now: float) -> None:
        nonlocal rendered_lines, next_plain_update
        total_kib, available_kib = _read_meminfo_kib()
        lines = _runtime_monitor_lines(
            label, process.pid, now - started, stats, alive, available_kib, total_kib
        )
        if interactive:
            if rendered_lines:
                sys.stdout.write(f"\x1b[{rendered_lines}F")
            for line in lines:
                sys.stdout.write(f"\x1b[2K{line}\n")
            sys.stdout.flush()
            rendered_lines = len(lines)
        elif not alive or now >= next_plain_update:
            print(
                f"[monitor] pid={process.pid} elapsed={now - started:.1f}s "
                f"rss={_format_kib(stats.last_rss_kib)} peak={_format_kib(stats.last_peak_kib)} "
                f"observed_max={_format_kib(stats.max_rss_kib)} "
                f"min_avail={_format_kib(stats.min_available_kib)}"
                + ("" if alive else " (worker exited)"),
                flush=True,
            )
            next_plain_update = now + max(5.0, interval)

    while process.poll() is None:
        now = time.perf_counter()
        ticks = _read_process_cpu_ticks(process.pid)
        cpu_percent = None
        if (
            ticks is not None
            and previous_ticks is not None
            and previous_time is not None
            and now > previous_time
        ):
            cpu_percent = 100.0 * (ticks - previous_ticks) / clock_ticks / (
                now - previous_time
            )

        rss_kib, peak_kib = _read_proc_status_kib(process.pid)
        _, available_kib = _read_meminfo_kib()
        alive = stats.update(rss_kib, peak_kib, available_kib, cpu_percent)
        if not alive:
            # 已成僵尸 (status 里没有 VmRSS) 但尚未被 poll 回收: 交给循环外的最终一帧.
            break
        render(True, now)

        previous_ticks = ticks
        previous_time = now
        try:
            process.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            pass

    process.wait()
    # 进程已退出: 再刷一帧, 保留死前最后一次观测值而不是打 "--".
    render(False, time.perf_counter())
    return int(process.returncode or 0), stats



def peak_rss_bytes() -> int:
    """读取 Linux 当前进程地址空间的峰值 RSS, 返回字节数."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("峰值 RSS 测量仅支持 Linux/WSL, 请在 Ubuntu 中运行性能比较脚本.")
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "VmHWM:":
            if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) <= 0:
                raise RuntimeError(f"无法解析峰值 RSS: {line}")
            return int(fields[1]) * 1024
    raise RuntimeError("/proc/self/status 未提供 VmHWM, 不能报告峰值内存.")


def timing_statistics(values: Sequence[float]) -> dict[str, float]:
    """汇总重复测量, 四分位点采用 NumPy 默认线性插值."""
    data = np.asarray(values, dtype=float)
    return {
        'median': float(np.median(data)), 'q25': float(np.quantile(data, 0.25)),
        'q75': float(np.quantile(data, 0.75)), 'min': float(np.min(data)),
        'max': float(np.max(data)),
    }


def performance_environment() -> dict[str, Any]:
    """记录当前工作进程的依赖版本、线程环境及已加载的 BLAS 线程池."""
    environment = {
        'platform': platform.platform(), 'python': platform.python_version(),
        'numpy': np.__version__, 'processor': platform.processor(),
        'logical_cpu_count': os.cpu_count(),
        'thread_environment': {key: os.environ.get(key) for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
        )},
    }
    for package in ('scipy', 'soptx'):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        environment['threadpools'] = None
    else:
        environment['threadpools'] = sorted(
            threadpool_info(), key=lambda item: item.get('filepath', '')
        )
    return environment
