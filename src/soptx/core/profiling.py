"""按阶段记录耗时与 CPU 内存的工具.

内存口径: 阶段开始记 ``before = VmRSS``, 向 ``/proc/self/clear_refs`` 写 ``5`` 把 ``VmHWM``
重置为当前 ``VmRSS``, 阶段结束读 ``VmHWM`` 为该阶段的峰值 ``peak``, 净增 ``net = peak - before``.
依赖 Linux 的 ``/proc`` 接口, 其他平台直接报错, 不返回哨兵值. 只计 CPU 内存, 不含 GPU 显存.

另记阶段内的主缺页次数 (需从磁盘读回内存页, 含从交换空间换入) 与阶段结束时已换出的内存量
``VmSwap``, 用于事后判断耗时异常是否来自换页; 二者只写入记录, 不打印.
"""

from __future__ import annotations

from contextlib import contextmanager
import resource
from time import perf_counter
from typing import Iterator, Optional

_GIB_PER_KIB = 1.0 / 2 ** 20


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


@contextmanager
def measure(records: dict, name: str, label: Optional[str] = None) -> Iterator[None]:
    """测量 ``with`` 块的耗时与 CPU 内存, 结果写入 ``records[name]``.

    Parameters
    ----------
    records : 收集各阶段结果的字典.
    name : 阶段名.
    label : 非 None 时阶段结束即打印一行, 以 ``[label/name]`` 为前缀; 进程中途被杀时,
        最后打印的阶段之后的那个阶段即出事阶段.

    Notes
    -----
    ``records[name]`` 为 ``dict(seconds, peak_gib, net_gib, after_gib, major_faults, swap_gib)``:
    耗时 (s), 阶段峰值, 峰值相对阶段开始的净增, 阶段结束时的常驻内存, 阶段内主缺页次数, 阶段
    结束时已换出的内存量 (内存量均为 GiB). ``with`` 块抛出异常时不写入记录. 打印行只含耗时与
    内存峰值, 主缺页与换出量只在记录中.
    """
    before = _status_kib('VmRSS')
    faults_before = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    # 写 5 把 VmHWM 重置为当前 VmRSS, 使其只反映本阶段的峰值
    with open('/proc/self/clear_refs', 'w') as clear_refs:
        clear_refs.write('5')
    start = perf_counter()
    yield
    seconds = perf_counter() - start
    after = _status_kib('VmRSS')
    peak = max(before, after, _status_kib('VmHWM'))
    major_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt - faults_before
    swap = _status_kib('VmSwap') * _GIB_PER_KIB
    records[name] = dict(seconds=seconds, peak_gib=peak * _GIB_PER_KIB,
                         net_gib=(peak - before) * _GIB_PER_KIB, after_gib=after * _GIB_PER_KIB,
                         major_faults=major_faults, swap_gib=swap)
    if label is not None:
        print(f'[{label}/{name}] {seconds:.2f} s, 峰值 {peak * _GIB_PER_KIB:.2f} GB '
              f'(+{(peak - before) * _GIB_PER_KIB:.2f} GB)', flush=True)
