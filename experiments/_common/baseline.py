# -*- coding: utf-8 -*-
"""单核硬件基线: memcpy 带宽与 dgemm 算力.

各实验目录的算子乘指标 (有效带宽下界 GB/s、GFLOP/s) 只有与同一台机器、同一线程数下的硬件上限
相比才能判断内核是内存受限还是算力受限. 本模块给出两条单线程基线:

- memcpy: 两块 float64 大数组 ``np.copyto(dst, src)``, 字节数按读 + 写计 (2 x 数组字节数),
  报告 GB/s. 这是顺序访问的搬运上限, gather / scatter 的随机访问达不到它.
- dgemm: ``a @ b`` (OpenBLAS / MKL dgemm), 浮点数按 2 n^3 计, 报告 GFLOP/s. 这是单核
  向量化乘加的实际上限 (非理论峰值).

线程数由调用方通过环境变量 (``OMP_NUM_THREADS`` 等) 限制为 1, 本模块只记录不设置;
若安装了 ``threadpoolctl`` 则同时记录 BLAS 线程池的实际线程数以便核对.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from typing import Any

import numpy as np

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
GIB = 2**30


def thread_facts() -> dict[str, Any]:
    """记录线程数相关事实: 线程环境变量与 (若可用) BLAS 线程池的实际线程数."""
    facts: dict[str, Any] = {f"env_{k}": os.environ.get(k) for k in THREAD_ENV_VARS}
    try:
        from threadpoolctl import threadpool_info

        facts["threadpools"] = [
            {k: info.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
            for info in threadpool_info()
        ]
    except Exception:  # noqa: BLE001 - 未安装 threadpoolctl 时不阻断测量
        facts["threadpools"] = None
    return facts


def blas_facts() -> dict[str, Any]:
    """记录 numpy 构建时链接的 BLAS / LAPACK 名称与版本 (numpy >= 1.25 的 dicts 模式)."""
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        return {
            "blas": {k: deps.get("blas", {}).get(k) for k in ("name", "version")},
            "lapack": {k: deps.get("lapack", {}).get(k) for k in ("name", "version")},
        }
    except Exception:  # noqa: BLE001
        return {"blas": None, "lapack": None}


def _timed(fn: Any, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def measure_memcpy(size_bytes: int = 2 * GIB, repeats: int = 10) -> dict[str, Any]:
    """单线程 memcpy 带宽.

    Parameters
    ----------
    size_bytes : int
        每块数组的字节数 (float64), 默认 2 GiB; 两块共占 2 x size_bytes 内存.
    repeats : int
        计时重复次数, 取中位数.

    Returns
    -------
    dict
        ``memcpy_gbps`` = 2 x size_bytes / 中位耗时 / 1e9 (读 + 写), 以及各次耗时.
    """
    n = size_bytes // 8
    src = np.ones(n, dtype=np.float64)
    dst = np.zeros(n, dtype=np.float64)
    np.copyto(dst, src)  # 预热: 触发 dst 的页分配, 不计入
    times = _timed(lambda: np.copyto(dst, src), repeats)
    t_med = statistics.median(times)
    moved = 2 * n * 8
    return {
        "memcpy_array_bytes": int(n * 8),
        "memcpy_bytes_moved": int(moved),
        "memcpy_repeats": repeats,
        "memcpy_seconds_median": round(t_med, 6),
        "memcpy_seconds_min": round(min(times), 6),
        "memcpy_seconds_all": [round(t, 6) for t in times],
        "memcpy_gbps": round(moved / t_med / 1e9, 2),
    }


def measure_dgemm(size: int = 4096, repeats: int = 5, seed: int = 0) -> dict[str, Any]:
    """单线程 dgemm 算力.

    Parameters
    ----------
    size : int
        方阵阶数 n, 默认 4096 (三块矩阵共 3 x 128 MiB).
    repeats : int
        计时重复次数, 取中位数.
    seed : int
        随机矩阵种子.

    Returns
    -------
    dict
        ``dgemm_gflops`` = 2 n^3 / 中位耗时 / 1e9, 以及各次耗时.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size))
    b = rng.standard_normal((size, size))
    _ = a @ b  # 预热
    times = _timed(lambda: a @ b, repeats)
    t_med = statistics.median(times)
    flops = 2.0 * size**3
    return {
        "dgemm_size": size,
        "dgemm_flops": int(flops),
        "dgemm_repeats": repeats,
        "dgemm_seconds_median": round(t_med, 6),
        "dgemm_seconds_min": round(min(times), 6),
        "dgemm_seconds_all": [round(t, 6) for t in times],
        "dgemm_gflops": round(flops / t_med / 1e9, 2),
    }


def measure_baseline(
    memcpy_bytes: int = 2 * GIB,
    memcpy_repeats: int = 10,
    dgemm_size: int = 4096,
    dgemm_repeats: int = 5,
) -> dict[str, Any]:
    """依次测 memcpy 与 dgemm 并附上环境事实, 返回可直接落盘的扁平字典."""
    out: dict[str, Any] = {
        "panel": "baseline",
        "device": "CPU",
        "device_type": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        **blas_facts(),
        **thread_facts(),
    }
    out.update(measure_memcpy(memcpy_bytes, memcpy_repeats))
    out.update(measure_dgemm(dgemm_size, dgemm_repeats))
    return out


def print_baseline(out: dict[str, Any]) -> None:
    """打印基线摘要 (与各目录 run.py 的树状看板同风格)."""
    pools = out.get("threadpools")
    pool_text = (
        ", ".join(f"{p.get('internal_api')}={p.get('num_threads')}" for p in pools) if pools else "threadpoolctl 不可用"
    )
    env_text = ", ".join(f"{k}={out.get(f'env_{k}')}" for k in THREAD_ENV_VARS)
    blas = out.get("blas") or {}
    print("\n● [cpu-baseline] 单核硬件基线")
    print(f"  ├── Env           : {env_text} | pools: {pool_text}")
    print(f"  ├── numpy / BLAS  : numpy {out.get('numpy_version')} | {blas.get('name')} {blas.get('version')}")
    print(
        f"  ├── memcpy        : {out.get('memcpy_array_bytes', 0) / GIB:.0f} GiB x2, "
        f"x{out.get('memcpy_repeats')} | median {out.get('memcpy_seconds_median', 0) * 1000:.1f} ms | "
        f"{out.get('memcpy_gbps', 0):.2f} GB/s (read + write)"
    )
    print(
        f"  └── dgemm         : n = {out.get('dgemm_size')}, x{out.get('dgemm_repeats')} | "
        f"median {out.get('dgemm_seconds_median', 0):.3f} s | {out.get('dgemm_gflops', 0):.2f} GFLOP/s\n"
    )
