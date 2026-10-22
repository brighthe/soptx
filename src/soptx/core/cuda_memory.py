"""限制进程在 CUDA 设备与主机上可用的内存, 使超出上限时直接报错, 并记录失败点.

Windows (WDDM) 驱动下显存不足时, 驱动会把分配静默落到系统内存 (sysmem fallback); 主机内存不足时, 系统会
换出到 swap. 两者都使程序不报错但显著变慢, 计时与内存数据随之失真. ``limit_cuda_memory`` 给 PyTorch 缓存
分配器设固定上限, ``limit_cuda_memory_to_free`` 按启动时的空闲显存减去余量设上限, 超出即抛
``torch.OutOfMemoryError``; ``limit_cpu_memory`` 以 ``RLIMIT_AS`` 限制启动后新增的虚拟内存, 超出即由 NumPy
抛 ``MemoryError`` 或由 PyTorch 抛含 ``DefaultCPUAllocator`` 的 ``RuntimeError``. ``is_out_of_memory`` 识别
这几类异常, ``record_out_of_memory`` 把失败点写入 ``failure.json``. 只在需要时导入 torch, 不用 PyTorch 的
进程不受影响. 主机内存部分依赖 Linux 的 ``/proc``.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
from typing import Any, Sequence

from .profiling import _GIB_PER_KIB, _status_kib

_BYTES_PER_GIB = float(2 ** 30)

#: 默认为 cuSPARSE、cuBLAS 等不经 PyTorch 分配器的显存预留的余量, 单位 GiB.
#: 390x65x65 网格上 FA 报显存不足时整卡已无空闲, 据此估计分配器之外约占 0.8 GiB, 0.25 GiB 时上限压到物理显存边缘.
DEFAULT_MARGIN_GIB = 1.0


def limit_cuda_memory(device: Any, limit_gib: float) -> dict[str, float]:
    """把 PyTorch 缓存分配器在 ``device`` 上的上限固定为 ``limit_gib``.

    固定上限使各次运行的可用显存相同, 不随启动时其他进程 (如桌面) 的占用而变.

    Parameters
    ----------
    device : str or torch.device
        CUDA 设备; ``'cuda'`` 即当前设备.
    limit_gib : float
        分配器上限, 单位 GiB.

    Returns
    -------
    record : dict
        ``free_gib`` (调用时的空闲显存)、``total_gib`` (设备总显存)、``limit_gib`` 与 ``fraction``
        (上限占总显存的比例), 供写入运行配置.

    Raises
    ------
    ValueError
        ``device`` 不是 CUDA 设备, ``limit_gib`` 不为正, 或调用时的空闲显存小于 ``limit_gib``.

    Notes
    -----
    空闲显存小于上限时拒绝运行: 否则分配器可在物理显存用尽前继续申请, 超出部分可能被驱动落到系统内存.
    上限只约束经 PyTorch 分配器的显存, 不含 CUDA 上下文与 cuSPARSE、cuBLAS 等库自行申请的工作区; 这部分
    使总占用越过物理显存时, 是否报错取决于驱动的系统内存回退设置. 上限作用于保留量而非已分配量, 宜在首次
    分配显存之前调用, 其余说明同 ``limit_cuda_memory_to_free``.
    """
    import torch

    cuda = torch.device(device)
    if cuda.type != 'cuda':
        raise ValueError(f"device 须为 CUDA 设备, 得到 {device!r}")
    # set_per_process_memory_fraction 只接受带编号的设备, 'cuda' 补为当前设备的编号
    if cuda.index is None:
        cuda = torch.device('cuda', torch.cuda.current_device())
    if limit_gib <= 0:
        raise ValueError(f"limit_gib 须为正数, 得到 {limit_gib}")
    free, total = torch.cuda.mem_get_info(cuda)
    limit = limit_gib * _BYTES_PER_GIB
    if free < limit:
        raise ValueError(f"空闲显存 {free / _BYTES_PER_GIB:.2f} GiB 小于上限 {limit_gib:.2f} GiB; "
                         "先关闭占用显存的程序")
    fraction = limit / total
    torch.cuda.set_per_process_memory_fraction(fraction, cuda)
    return dict(free_gib=free / _BYTES_PER_GIB, total_gib=total / _BYTES_PER_GIB, limit_gib=float(limit_gib),
                fraction=fraction)


def limit_cuda_memory_to_free(device: Any = 'cuda', margin_gib: float = DEFAULT_MARGIN_GIB) -> dict[str, float]:
    """把 PyTorch 缓存分配器在 ``device`` 上的上限设为当前空闲显存减去余量.

    Parameters
    ----------
    device : str or torch.device, optional
        CUDA 设备, 默认 ``'cuda'``, 即当前设备.
    margin_gib : float, optional
        为不经 PyTorch 分配器的显存预留的余量, 单位 GiB, 默认 ``DEFAULT_MARGIN_GIB``.

    Returns
    -------
    record : dict
        ``free_gib`` (调用时的空闲显存)、``total_gib`` (设备总显存)、``margin_gib``、``limit_gib``
        (设定的上限) 与 ``fraction`` (上限占总显存的比例), 供写入运行配置.

    Raises
    ------
    ValueError
        ``device`` 不是 CUDA 设备, ``margin_gib`` 为负, 或空闲显存不大于余量.

    Notes
    -----
    上限作用于缓存分配器的保留量 (``torch.cuda.memory_reserved``) 而非已分配量; 碎片使保留量高于已分配量,
    显存紧张时可能在已分配量尚未达到上限时报错. 本进程此前已保留的显存计入上限, 宜在首次分配显存之前调用.
    空闲显存只在调用时读取一次, 之后其他进程占用的变化不会更新上限.
    """
    import torch

    cuda = torch.device(device)
    if cuda.type != 'cuda':
        raise ValueError(f"device 须为 CUDA 设备, 得到 {device!r}")
    # set_per_process_memory_fraction 只接受带编号的设备, 'cuda' 补为当前设备的编号
    if cuda.index is None:
        cuda = torch.device('cuda', torch.cuda.current_device())
    if margin_gib < 0:
        raise ValueError(f"margin_gib 须为非负数, 得到 {margin_gib}")
    free, total = torch.cuda.mem_get_info(cuda)
    margin = margin_gib * _BYTES_PER_GIB
    if free <= margin:
        raise ValueError(f"空闲显存 {free / _BYTES_PER_GIB:.2f} GiB 不大于余量 {margin_gib:.2f} GiB")
    fraction = (free - margin) / total
    torch.cuda.set_per_process_memory_fraction(fraction, cuda)
    return dict(free_gib=free / _BYTES_PER_GIB, total_gib=total / _BYTES_PER_GIB, margin_gib=float(margin_gib),
                limit_gib=(free - margin) / _BYTES_PER_GIB, fraction=fraction)


def limit_cpu_memory(limit_gib: float) -> dict[str, float]:
    """以 ``RLIMIT_AS`` 把本进程启动后新增的虚拟内存限制在 ``limit_gib`` 以内.

    上限取调用时的虚拟内存 ``VmSize`` 加 ``limit_gib``, 扣除导入库时映射的固定地址空间 (导入 torch 时可达数
    GiB), NumPy 与 PyTorch 后端均适用.

    Parameters
    ----------
    limit_gib : float
        启动后可新增的虚拟内存, 单位 GiB.

    Returns
    -------
    record : dict
        ``base_virtual_gib`` (调用时的 ``VmSize``)、``available_gib`` (调用时系统的 ``MemAvailable``)、
        ``limit_gib`` 与 ``rlimit_as_gib`` (设定的 ``RLIMIT_AS`` 软限制), 供写入运行配置.

    Raises
    ------
    ValueError
        ``limit_gib`` 不为正, 或调用时系统可用内存小于 ``limit_gib``.

    Notes
    -----
    可用内存小于上限时拒绝运行: 否则上限以内的申请也可能换出到 swap. 虚拟内存在申请时即整块计入, 尚未写入
    的页与线程栈、malloc arena 的预留也算在内, 故该上限比按常驻内存 (RSS) 计的上限更严, 可能稍早报错. 调用之
    后加载的动态库同样计入. 只设软限制, 硬限制不变. 宜在导入全部依赖之后、开始大块分配之前调用.
    """
    if limit_gib <= 0:
        raise ValueError(f"limit_gib 须为正数, 得到 {limit_gib}")
    available_kib = None
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            if line.startswith('MemAvailable:'):
                available_kib = int(line.split()[1])
                break
    if available_kib is None:
        raise RuntimeError('/proc/meminfo 中没有 MemAvailable')
    available_gib = available_kib * _GIB_PER_KIB
    if available_gib < limit_gib:
        raise ValueError(f"系统可用内存 {available_gib:.2f} GiB 小于上限 {limit_gib:.2f} GiB; "
                         "先关闭占用内存的程序")
    base_bytes = _status_kib('VmSize') * 1024
    soft = base_bytes + int(limit_gib * _BYTES_PER_GIB)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and soft > hard:
        raise ValueError(f"RLIMIT_AS 硬限制 {hard / _BYTES_PER_GIB:.2f} GiB 小于所需的 {soft / _BYTES_PER_GIB:.2f} GiB")
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    return dict(base_virtual_gib=base_bytes / _BYTES_PER_GIB, available_gib=available_gib,
                limit_gib=float(limit_gib), rlimit_as_gib=soft / _BYTES_PER_GIB)


def is_out_of_memory(error: BaseException) -> bool:
    """判断异常是否为内存或显存不足.

    Parameters
    ----------
    error : BaseException
        捕获的异常.

    Returns
    -------
    bool
        ``MemoryError`` (含 NumPy 的数组内存错误)、``torch.OutOfMemoryError``, 或消息含 ``DefaultCPUAllocator``
        的 ``RuntimeError`` (PyTorch 在 CPU 上分配失败) 时为 True.
    """
    if isinstance(error, MemoryError):
        return True
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and 'DefaultCPUAllocator' in str(error)


def record_out_of_memory(output: Path, device: Any, progress: dict, error: BaseException,
                         setup_order: Sequence[str], iteration_order: Sequence[str]) -> dict[str, Any]:
    """内存或显存不足时把失败点写入 ``output / 'failure.json'`` 并打印一行.

    Parameters
    ----------
    output : pathlib.Path
        运行输出目录, 须已存在.
    device : str or torch.device
        发生不足的设备; 以 ``'cuda'`` 开头时记显存, 否则记主机内存.
    progress : dict
        运行中就地维护的进度: ``phase`` ('准备' 或 '迭代')、``iteration``、准备阶段记录 ``setup_stages``、
        当前所在阶段组的记录 ``stages``, 以及 ``limit_cuda_memory`` 系列的返回值 ``gpu_memory`` 或
        ``limit_cpu_memory`` 的返回值 ``cpu_memory``.
    error : BaseException
        捕获的异常, 其消息通常含本次请求的内存量.
    setup_order, iteration_order : sequence of str
        准备阶段与每轮迭代各计时块的执行顺序, 用于判定失败阶段.

    Returns
    -------
    failure : dict
        写入 ``failure.json`` 的内容.

    Notes
    -----
    ``measure`` 只在阶段正常结束时写入记录, 失败阶段取执行顺序中第一个尚无记录的阶段; 不足若发生在两个计时块
    之间, 记为其后的阶段. ``failed_stage_peak_gib`` 为失败阶段内至失败前的峰值, 不含失败的那次请求: CUDA 下为
    已分配显存 (``measure`` 在阶段开始时重置了峰值统计), CPU 下为 ``VmHWM`` (``measure`` 在阶段开始时重置).
    失败点单独写入 ``failure.json``, 不写 ``summary.json``.
    """
    cuda = str(device).startswith('cuda')
    phase = progress.get('phase')
    stages = progress.get('stages', {})
    order = setup_order if phase == '准备' else iteration_order
    failed_stage = next((name for name in order if name not in stages), None) if phase else None
    if cuda:
        import torch

        cuda_device = torch.device(device)
        memory = dict(failed_stage_peak_gib=torch.cuda.max_memory_allocated(cuda_device) / _BYTES_PER_GIB,
                      allocated_gib=torch.cuda.memory_allocated(cuda_device) / _BYTES_PER_GIB,
                      reserved_gib=torch.cuda.memory_reserved(cuda_device) / _BYTES_PER_GIB)
        limit = progress.get('gpu_memory') or {}
    else:
        memory = dict(failed_stage_peak_gib=_status_kib('VmHWM') * _GIB_PER_KIB,
                      rss_gib=_status_kib('VmRSS') * _GIB_PER_KIB,
                      virtual_gib=_status_kib('VmSize') * _GIB_PER_KIB)
        limit = progress.get('cpu_memory') or {}
    failure = dict(error='cuda_out_of_memory' if cuda else 'cpu_out_of_memory',
                   exception=type(error).__name__, message=str(error), phase=phase,
                   iteration=progress.get('iteration'), failed_stage=failed_stage, **memory,
                   gpu_memory=progress.get('gpu_memory'), cpu_memory=progress.get('cpu_memory'),
                   setup_stages=progress.get('setup_stages'),
                   iteration_stages=stages if phase == '迭代' else None)
    path = Path(output) / 'failure.json'
    path.write_text(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    where = phase if phase != '迭代' else f'迭代 {progress["iteration"]}'
    kind = '显存' if cuda else '内存'
    print(f'[失败] {kind}不足于 {where}/{failed_stage}, 阶段内峰值 {failure["failed_stage_peak_gib"]:.2f} GiB, '
          f'上限 {limit.get("limit_gib", float("nan")):.2f} GiB; 详见 {path}', flush=True)
    return failure


def record_cuda_out_of_memory(output: Path, device: Any, progress: dict, error: Exception,
                              setup_order: Sequence[str], iteration_order: Sequence[str]) -> dict[str, Any]:
    """显存不足时把失败点写入 ``failure.json``, 即 CUDA 设备上的 ``record_out_of_memory``.

    Parameters
    ----------
    output, device, progress, error, setup_order, iteration_order
        同 ``record_out_of_memory``; ``device`` 须为 CUDA 设备.

    Returns
    -------
    failure : dict
        写入 ``failure.json`` 的内容.
    """
    return record_out_of_memory(output, device, progress, error, setup_order, iteration_order)
