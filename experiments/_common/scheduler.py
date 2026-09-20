# -*- coding: utf-8 -*-
"""独立子进程调度: 每个数据点独占一个 Worker 进程, 父进程负责监控、失败归因与落盘.

各实验目录的 ``run.py`` 先把选中的 ``Case`` 展开为 ``Run`` 列表 (展开规则因目录而异, 如
fa 按 method x route、ea 按 method、alc 按 scheme), 再交给本模块的 ``command_run`` 逐个执行.

失败时写 ``<artifact stem>.failed.json`` 侧车, 含退出信号、父进程采样到的峰值 RSS 下界与
运行期间新增的 dmesg OOM 行 (按启动前快照 -> 退出后新增 归因, 不依赖 pid 与时间戳).
"""

from __future__ import annotations

import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .casefile import Case

# 一次子进程执行计划: (标签, argv, 产物路径, 一句话说明, 环境变量或 None)
Run = tuple[str, list[str], Path, str, dict[str, str] | None]


# ----------------------------------------------------------------------------- 文本
def display_width(text: str) -> int:
    """计算包含中文字符的真实终端显示字宽 (基于 Unicode East Asian Width 规范)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text)


def pad(text: str, width: int) -> str:
    """按显示字宽向右填充空格."""
    return text + " " * max(0, width - display_width(text))


def print_case_table(cases: tuple[Case, ...], columns: Sequence[tuple[str, str, str]]) -> int:
    """打印已注册算例表.

    Parameters
    ----------
    cases : tuple of Case
        已加载的算例.
    columns : sequence of (header, extra_key, default)
        除首列 ``case-id`` 外的各列: 表头、从 ``Case.extra`` 取值的键、缺省显示值.
        ``extra_key`` 为 ``"panel"`` / ``"role"`` / ``"artifact"`` 时改取 Case 同名属性.
    """
    header = ("case-id", *(c[0] for c in columns))
    rows = []
    for c in cases:
        row = [c.id]
        for _, key, default in columns:
            if key in ("panel", "role", "artifact", "summary"):
                row.append(str(getattr(c, key)))
            else:
                row.append(str(c.extra.get(key, default)))
        rows.append(tuple(row))

    widths = [
        max(display_width(row[i]) for row in (header, *rows)) for i in range(len(header))
    ]
    print()
    print("  ".join(pad(value, widths[i]) for i, value in enumerate(header)).rstrip())
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for row in rows:
        print("  ".join(pad(value, widths[i]) for i, value in enumerate(row)).rstrip())
    print()
    return 0


def artifact_name(kind: str, parts: Sequence[str], n: int, device: str, bform: bool = False) -> str:
    """产物文件名: <kind>_<parts>_n<N>[_cuda][_bform].json; 非 CPU 设备加 _cuda 后缀以免与 CPU 结论混淆."""
    name = f"{kind}_{'_'.join(parts)}_n{n}"
    if device != "cpu":
        name += "_cuda"
    if bform:
        name += "_bform"
    return name + ".json"


def expand(requested: str | None, is_all: bool, all_values: Sequence[str], default: str) -> list[str]:
    """把 --method/--route 类覆盖值展开为具体列表: 'all' 或 --all 未指定 → 全部, 否则单值."""
    if requested == "all" or (is_all and not requested):
        return list(all_values)
    if requested:
        return [requested]
    return [default]


# ----------------------------------------------------------------------------- 资源采样
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


def wait_with_monitor(
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


# ----------------------------------------------------------------------------- 失败归因
def describe_exit(returncode: int) -> tuple[str, str | None, bool]:
    """把退出码翻译成可读说明; 返回 (说明, 信号名, 是否疑似 OOM)."""
    if returncode >= 0:
        return f"退出码 {returncode}", None, False
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"
    suspected_oom = name == "SIGKILL"
    text = f"退出码 {returncode} (被信号 {name} 终止"
    if suspected_oom:
        text += ", 疑似被内核 OOM killer 杀死"
    return text + ")", name, suspected_oom


_OOM_NEEDLES = ("invoked oom-killer", "oom-kill:", "Killed process")


def dmesg_snapshot() -> list[str] | None:
    """读取当前 dmesg 全文 (按行); 不可用或无权限时返回 None."""
    try:
        completed = subprocess.run(
            ["dmesg"], capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.splitlines()


def _dmesg_new_oom_lines(before: list[str] | None) -> list[str] | None:
    """返回自 ``before`` 快照之后新增的 OOM 相关 dmesg 行.

    不按 PID 匹配: WSL2 各发行版运行在独立 PID 命名空间, 内核日志里的 pid 是全局编号,
    与发行版内 ``ps`` / ``Popen.pid`` 看到的不同; dmesg 时间戳在 WSL 里也不可靠.
    故以 "启动前快照 -> 退出后新增" 归因. dmesg 不可用时返回 None.
    """
    after = dmesg_snapshot()
    if after is None or before is None:
        return None
    if after[: len(before)] == before:
        new_lines = after[len(before):]
    else:
        # 环形缓冲已翻转: 退回到快照最后一行在新输出中的位置.
        new_lines = after
        if before:
            last = before[-1]
            for idx in range(len(after) - 1, -1, -1):
                if after[idx] == last:
                    new_lines = after[idx + 1:]
                    break
    return [
        line.strip()
        for line in new_lines
        if any(needle in line for needle in _OOM_NEEDLES)
    ]


def _killed_anon_rss_kib(lines: list[str]) -> int | None:
    """从 "Out of memory: Killed process ... anon-rss:NNNkB" 行解析被杀进程的 anon-rss (KiB)."""
    for line in reversed(lines):
        match = re.search(r"Killed process .*?anon-rss:(\d+)kB", line)
        if match:
            return int(match.group(1))
    return None


def failed_sidecar_path(artifact_path: Path) -> Path:
    """失败侧车文件路径: 与产物同名, 后缀改为 .failed.json."""
    return artifact_path.with_name(artifact_path.stem + ".failed.json")


def report_failure(
    *,
    label: str,
    argv: list[str],
    artifact_path: Path,
    pid: int,
    returncode: int,
    elapsed: float,
    started_at: str,
    stats: WorkerStats | None,
    dmesg_before: list[str] | None,
) -> None:
    """打印失败诊断并把它落盘到 .failed.json 侧车."""
    description, signal_name, suspected_oom = describe_exit(returncode)
    print(f"  失败: {description}, 用时 {elapsed:.1f} s")

    if stats is not None and stats.samples:
        print(
            f"  死前观测 (采样 {stats.samples} 次, 为真实峰值下界): "
            f"最后 RSS {_format_kib(stats.last_rss_kib)} | "
            f"观测最大 RSS {_format_kib(stats.max_rss_kib)} | "
            f"最小系统可用 {_format_kib(stats.min_available_kib)}"
        )

    dmesg_lines = _dmesg_new_oom_lines(dmesg_before)
    kernel_anon_rss_kib: int | None = None
    if dmesg_lines is None:
        print("  dmesg: 不可读 (无权限或不可用), 可手动执行 `sudo dmesg | grep -i \"killed process\"`")
    elif dmesg_lines:
        for line in dmesg_lines:
            if "invoked oom-killer" in line:
                continue  # 触发者行信息量低, 只留 oom-kill 与 Killed process 两行.
            print(f"  dmesg: {line}")
        kernel_anon_rss_kib = _killed_anon_rss_kib(dmesg_lines)
        if kernel_anon_rss_kib is not None:
            note = f"  内核记录被杀进程 anon-rss {_format_kib(kernel_anon_rss_kib)}"
            if stats is not None and stats.max_rss_kib:
                ratio = kernel_anon_rss_kib / stats.max_rss_kib
                verdict = "一致" if 0.95 <= ratio <= 1.05 else "不一致, 请核对是否为本 Worker"
                note += f", 与监控观测峰值 {_format_kib(stats.max_rss_kib)} {verdict}"
            note += " (dmesg 中的 pid 为内核全局编号, 与发行版内 PID 不同, 属正常现象)"
            print(note)
    elif suspected_oom:
        print("  dmesg: 运行期间无新增 OOM 记录 (SIGKILL 可能来自其他来源, 如手动 kill -9)")

    total_kib, _ = _read_meminfo_kib()
    payload: dict[str, Any] = {
        "case": label,
        "argv": argv,
        "artifact": artifact_path.name,
        "worker_pid": pid,
        "returncode": returncode,
        "signal": signal_name,
        "suspected_oom": suspected_oom,
        "started_at": started_at,
        "elapsed_s": round(elapsed, 1),
        "system_total_MiB": None if total_kib is None else round(total_kib / 1024, 1),
        "observed": None if stats is None else stats.to_mib_dict(),
        "kernel_killed_anon_rss_MiB": (
            None if kernel_anon_rss_kib is None else round(kernel_anon_rss_kib / 1024, 1)
        ),
        "dmesg": dmesg_lines,
    }
    sidecar = failed_sidecar_path(artifact_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  失败记录已落盘 -> {sidecar}")


# ----------------------------------------------------------------------------- 执行
def command_run(
    runs: Sequence[Run],
    *,
    cwd: Path,
    check_only: bool,
    skip_existing: bool,
    monitor: bool,
    monitor_interval: float,
) -> int:
    """逐个以独立子进程运行执行计划; 返回失败个数.

    Parameters
    ----------
    runs : sequence of Run
        由各目录 ``run.py`` 的 ``resolve_runs`` 展开好的 (label, argv, artifact_path, summary, env).
    cwd : Path
        子进程工作目录, 通常为仓库根.
    check_only : bool
        只打印命令不执行.
    skip_existing : bool
        产物已存在时跳过.
    monitor : bool
        父进程实时采样 Worker 的 CPU 与内存并原位刷新.
    monitor_interval : float
        采样间隔 (s).
    """
    failed = 0
    total = len(runs)
    for index, (label, argv, artifact_path, summary, env) in enumerate(runs, start=1):
        prefix = f"[{index}/{total}] {label}"
        if skip_existing and artifact_path.is_file():
            print(f"{prefix}: 产物已存在, 跳过")
            continue

        printable = " ".join(argv)
        if check_only:
            print(f"{prefix}: {printable}")
            continue

        if total > 1:
            print(f"\n{prefix}: {summary}", flush=True)
        started = time.perf_counter()
        started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        stats: WorkerStats | None = None
        dmesg_before = dmesg_snapshot()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE if monitor else None,
            text=True,
        )
        try:
            if monitor:
                returncode, stats = wait_with_monitor(
                    process,
                    label,
                    started,
                    monitor_interval,
                )
                worker_stdout, _ = process.communicate()
                if worker_stdout:
                    print(worker_stdout, end="")
            else:
                returncode = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        elapsed = time.perf_counter() - started

        if returncode != 0:
            failed += 1
            report_failure(
                label=label,
                argv=argv,
                artifact_path=artifact_path,
                pid=process.pid,
                returncode=returncode,
                elapsed=elapsed,
                started_at=started_at,
                stats=stats,
                dmesg_before=dmesg_before,
            )
        elif not artifact_path.is_file():
            failed += 1
            print(f"  失败: 进程正常退出但产物未生成 -> {artifact_path}")
        else:
            # 本次成功: 清掉同名的历史失败侧车, 避免与新产物并存造成误读.
            stale = failed_sidecar_path(artifact_path)
            if stale.is_file():
                stale.unlink()

    if failed:
        print(f"\n{failed} 个任务执行失败。")
    return failed
