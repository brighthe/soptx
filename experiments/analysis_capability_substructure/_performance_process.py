"""性能测量的资源监控与环境采集, 不作为用户命令行入口.

调用方各自启动独立 Worker 进程. 父进程侧的资源采样与看板渲染统一复用
``experiments._common.scheduler``, 本模块只保留本实验证据口径特有的部分: 峰值内存
采用 Linux 当前进程的 VmHWM 且字段缺失时直接报错, 不返回哨兵值; 另提供重复测量的
统计汇总与依赖版本、线程环境的采集.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import numpy as np

from experiments._common.scheduler import wait_with_monitor as _wait_with_monitor

__all__ = [
    "_wait_with_monitor",
    "peak_rss_bytes",
    "performance_environment",
    "timing_statistics",
]


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
    for package in ('scipy', 'fealpy', 'soptx'):
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
