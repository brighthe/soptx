"""按阶段记录耗时与 CPU 内存的工具.

内存口径: 阶段开始记 ``before = VmRSS``, 向 ``/proc/self/clear_refs`` 写 ``5`` 把 ``VmHWM``
重置为当前 ``VmRSS``, 阶段结束读 ``VmHWM`` 为该阶段的峰值 ``peak``, 净增 ``net = peak - before``.
依赖 Linux 的 ``/proc`` 接口, 其他平台直接报错, 不返回哨兵值. 默认只计 CPU 内存; ``measure`` 传入
CUDA 设备时另记该设备上 PyTorch 分配器的显存, 并在计时前后同步设备, 使异步 kernel 的耗时计入所在阶段.

另记阶段内的主缺页次数 (需从磁盘读回内存页, 含从交换空间换入) 与阶段结束时已换出的内存量
``VmSwap``, 用于事后判断耗时异常是否来自换页; 二者只写入记录, 不打印.

``blas_environment`` 记录进程实际加载的 BLAS 库与逻辑 CPU 数, 标明耗时数据的运行环境.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Iterator, Optional

_GIB_PER_KIB = 1.0 / 2 ** 20
_BYTES_PER_GIB = float(2 ** 30)


def _status_kib(field: str) -> int:
    """读 ``/proc/self/status`` 中的内存字段.

    Parameters
    ----------
    field : 字段名, 如 ``'VmRSS'`` 或 ``'VmHWM'``.

    Returns
    -------
    value : 字段值, 单位 KiB.

    Raises
    ------
    RuntimeError
        字段缺失.
    """
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith(field + ':'):
                return int(line.split()[1])
    raise RuntimeError(f'/proc/self/status 未提供 {field}, 不能报告内存')


def _cuda_device(device: Any) -> Any:
    """把 ``measure`` 的 ``device`` 参数解析为 CUDA 设备.

    Parameters
    ----------
    device : 设备名、``torch.device`` 或 None.

    Returns
    -------
    cuda : CUDA 设备时为对应的 ``torch.device``, 否则为 None. 非 CUDA 设备不导入 torch.
    """
    if device is None or not str(device).startswith('cuda'):
        return None
    import torch
    return torch.device(device)


@contextmanager
def measure(records: dict, name: str, label: Optional[str] = None,
            device: Any = None) -> Iterator[None]:
    """测量 ``with`` 块的耗时与 CPU 内存, 结果写入 ``records[name]``.

    Parameters
    ----------
    records : 收集各阶段结果的字典.
    name : 阶段名.
    label : 非 None 时阶段结束即打印一行, 以 ``[label/name]`` 为前缀; 进程中途被杀时,
        最后打印的阶段之后的那个阶段即出事阶段.
    device : 计算设备, 如 ``'cpu'``、``'cuda'`` 或 ``torch.device``. 为 CUDA 设备时另记显存,
        并在计时前后调用 ``torch.cuda.synchronize``; 其余取值 (含默认 None) 与不传相同.

    Notes
    -----
    ``records[name]`` 为 ``dict(seconds, peak_gib, net_gib, after_gib, major_faults, swap_gib)``:
    耗时 (s), 阶段峰值, 峰值相对阶段开始的净增, 阶段结束时的常驻内存, 阶段内主缺页次数, 阶段
    结束时已换出的内存量 (内存量均为 GiB). ``with`` 块抛出异常时不写入记录. 打印行只含耗时与
    内存峰值, 主缺页与换出量只在记录中.

    CUDA 设备下记录另含 ``gpu_peak_gib``, ``gpu_net_gib``, ``gpu_after_gib`` 与
    ``gpu_reserved_gib``: 阶段内 ``torch.cuda.max_memory_allocated`` 的峰值 (阶段开始时以
    ``reset_peak_memory_stats`` 重置), 峰值相对阶段开始的净增, 阶段结束时已分配的显存, 阶段结束时
    分配器保留的显存. 只计 PyTorch 分配器经手的显存, 不含 CUDA 上下文与 cuBLAS / cuSPARSE 等库
    的工作区; 打印行另附显存峰值.
    """
    cuda = _cuda_device(device)
    if cuda is not None:
        import torch
        torch.cuda.synchronize(cuda)
        gpu_before = torch.cuda.memory_allocated(cuda)
        torch.cuda.reset_peak_memory_stats(cuda)
    before = _status_kib('VmRSS')
    faults_before = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    # 写 5 把 VmHWM 重置为当前 VmRSS, 使其只反映本阶段的峰值
    with open('/proc/self/clear_refs', 'w') as clear_refs:
        clear_refs.write('5')
    start = perf_counter()
    yield
    if cuda is not None:
        torch.cuda.synchronize(cuda)
    seconds = perf_counter() - start
    after = _status_kib('VmRSS')
    peak = max(before, after, _status_kib('VmHWM'))
    major_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt - faults_before
    swap = _status_kib('VmSwap') * _GIB_PER_KIB
    records[name] = dict(seconds=seconds, peak_gib=peak * _GIB_PER_KIB,
                         net_gib=(peak - before) * _GIB_PER_KIB, after_gib=after * _GIB_PER_KIB,
                         major_faults=major_faults, swap_gib=swap)
    gpu_text = ''
    if cuda is not None:
        gpu_peak = max(gpu_before, torch.cuda.max_memory_allocated(cuda))
        records[name].update(gpu_peak_gib=gpu_peak / _BYTES_PER_GIB,
                             gpu_net_gib=(gpu_peak - gpu_before) / _BYTES_PER_GIB,
                             gpu_after_gib=torch.cuda.memory_allocated(cuda) / _BYTES_PER_GIB,
                             gpu_reserved_gib=torch.cuda.memory_reserved(cuda) / _BYTES_PER_GIB)
        gpu_text = (f', 显存峰值 {gpu_peak / _BYTES_PER_GIB:.2f} GB '
                    f'(+{(gpu_peak - gpu_before) / _BYTES_PER_GIB:.2f} GB)')
    if label is not None:
        print(f'[{label}/{name}] {seconds:.2f} s, 峰值 {peak * _GIB_PER_KIB:.2f} GB '
              f'(+{(peak - before) * _GIB_PER_KIB:.2f} GB){gpu_text}', flush=True)


def blas_environment() -> dict[str, Any]:
    """记录进程实际加载的 BLAS 库与逻辑 CPU 数, 供写入运行配置.

    BLAS 库取自 ``/proc/self/maps``: conda 的 ``libblas`` 只是转发层, 构建信息看不出实际
    实现. 与 ``measure`` 不同, 这里只是辅助记录, 非 Linux 下 ``blas_libraries`` 记为 None,
    不报错.

    Returns
    -------
    environment : ``blas_libraries`` 为已加载的 BLAS 动态库文件名, ``logical_cpu_count`` 为逻辑 CPU 数.
    """
    try:
        with open('/proc/self/maps') as maps:
            libraries = sorted({Path(line.split()[-1]).name for line in maps
                                if any(name in line for name in ('blis', 'openblas', 'mkl_'))})
    except OSError:
        libraries = None
    return dict(blas_libraries=libraries, logical_cpu_count=os.cpu_count())
