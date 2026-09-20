# -*- coding: utf-8 -*-
"""分阶段内存与耗时测量.

CPU 口径: 每个阶段先记 before = 当前 VmRSS, 再向 /proc/self/clear_refs 写 5 重置 VmHWM,
阶段结束读 VmHWM 作为该阶段的绝对峰值 peak, net = peak - before. 全程峰值
(process_max_rss) = 各阶段峰值的最大值 (clear_refs 会连带重置 ru_maxrss, 故不用它).
决定是否 OOM 的是绝对峰值而不是净增.

CUDA 口径: 以 ``torch.cuda.memory_allocated`` 为 before, ``reset_peak_memory_stats`` 后以
``max_memory_allocated`` 为 peak, 字段名与 CPU 完全一致, 由 ``memory_kind`` 区分.
"""

from __future__ import annotations

import contextlib
import gc
import resource
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence

DEFAULT_MEMORY_TOTAL = 47.04 * 2**30  # WSL 来宾 MemTotal (字节), 仅作事实记录
MEMORY_BUDGET = 45 * 2**30  # 峰值内存预算 (字节): 容量结论以进程绝对峰值 RSS 不超过此值为界


def rss_kib() -> int:
    """进程 RSS 高水位 (ru_maxrss, KiB). 注意: 向 clear_refs 写 5 会连带重置它, 测量阶段内不要依赖."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _read_self_status(*names: str) -> Dict[str, int]:
    """读取 /proc/self/status 中指定字段 (KiB)."""
    values: Dict[str, int] = {}
    with open("/proc/self/status", encoding="utf-8") as file:
        for line in file:
            name = line.partition(":")[0]
            if name in names:
                values[name] = int(line.split()[1])
    return values


def cur_rss_kib() -> int:
    """进程当前常驻内存 VmRSS (KiB)."""
    return _read_self_status("VmRSS").get("VmRSS", -1)


def peak_rss_kib() -> int:
    """进程 VmHWM (KiB): 自上次重置以来的 RSS 高水位."""
    return _read_self_status("VmHWM").get("VmHWM", -1)


def reset_peak_rss() -> bool:
    """向 /proc/self/clear_refs 写 5, 把 VmHWM 重置为当前 RSS; 内核不支持时返回 False."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as file:
            file.write("5")
    except OSError:
        return False
    return True


@dataclass
class StageRecord:
    """单个阶段的内存与耗时记录 (KiB / s)."""

    before_kib: int
    peak_kib: int
    t_s: float

    @property
    def net_kib(self) -> int:
        return max(0, self.peak_kib - self.before_kib)


class StageMeter:
    """按阶段记录 before / peak / net 的 CPU 内存测量器 (基于 VmHWM 重置).

    用法::

        meter = StageMeter()
        with meter.stage("stage1"):
            K_e = integrator.assembly(vs)
        fields = meter.fields()   # stage1_before_MiB / stage1_peak_MiB / stage1_net_MiB / t_stage1_s

    每个阶段的 peak 是该阶段内的绝对 RSS 高水位, 与 OOM 直接可比; net 是相对阶段开始时的净增.
    """

    memory_kind = "rss"

    def __init__(self) -> None:
        self.records: Dict[str, StageRecord] = {}
        self.reset_supported = True

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        gc.collect()
        before = cur_rss_kib()
        if not reset_peak_rss():
            self.reset_supported = False
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        peak = max(peak_rss_kib(), before)
        self.records[name] = StageRecord(before, peak, t1 - t0)

    def combine(self, name: str, parts: Sequence[str]) -> None:
        """把连续的若干子阶段合成一个阶段: before 取首个, peak 取最大, 耗时求和."""
        recs = [self.records[p] for p in parts]
        self.records[name] = StageRecord(
            recs[0].before_kib, max(r.peak_kib for r in recs), sum(r.t_s for r in recs)
        )

    def net_bytes(self, name: str) -> int:
        return self.records[name].net_kib * 1024

    def peak_bytes(self, name: str) -> int:
        return self.records[name].peak_kib * 1024

    def before_bytes(self, name: str) -> int:
        return self.records[name].before_kib * 1024

    def seconds(self, name: str) -> float:
        return self.records[name].t_s

    def fields(self) -> Dict[str, Any]:
        """导出全部阶段字段 (MiB / s) 与进程级绝对高水位."""
        out: Dict[str, Any] = {}
        for name, rec in self.records.items():
            out[f"{name}_before_MiB"] = round(rec.before_kib / 1024, 1)
            out[f"{name}_peak_MiB"] = round(rec.peak_kib / 1024, 1)
            out[f"{name}_net_MiB"] = round(rec.net_kib / 1024, 1)
            out[f"t_{name}_s"] = round(rec.t_s, 3)
        out["process_max_rss_MiB"] = round(self.max_peak_kib() / 1024, 1)
        out["peak_reset_supported"] = self.reset_supported
        return out

    def max_peak_kib(self) -> int:
        """全部已测阶段峰值的最大值 = 测量区间内的进程绝对高水位.

        注意 clear_refs 重置 VmHWM 时会连带重置 ru_maxrss, 所以全程峰值不能再读 ru_maxrss.
        """
        return max((rec.peak_kib for rec in self.records.values()), default=0)


class CudaStageMeter(StageMeter):
    """与 ``StageMeter`` 同字段名的 CUDA 显存分阶段测量器 (PyTorch 分配器口径).

    before = ``memory_allocated``, peak = ``max_memory_allocated`` (阶段开始时 ``reset_peak_memory_stats``),
    阶段前后均 ``synchronize``. ``process_max_rss_MiB`` 字段沿用同名以便 compare 脚本统一读取,
    含义为测量区间内的显存绝对高水位; 由 ``memory_kind = "vram"`` 区分.
    """

    memory_kind = "vram"

    def __init__(self, device: Any) -> None:
        super().__init__()
        import torch

        self._torch = torch
        self.device = device

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        torch = self._torch
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        before = int(torch.cuda.memory_allocated(self.device)) // 1024
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        peak = max(int(torch.cuda.max_memory_allocated(self.device)) // 1024, before)
        self.records[name] = StageRecord(before, peak, t1 - t0)
