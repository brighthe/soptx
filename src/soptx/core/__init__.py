"""SOPTX 各子系统共用的底层基础设施.

这里的内容都与具体领域无关: 日志、计时、按阶段的耗时与内存记录、CUDA 显存与主机内存上限、简单的结果记录、MUMPS 侧的
MPI 激活钩子, 以及共用的数值下界常数.

分析器要求 ``pde`` 与 ``interpolation_scheme`` 满足的结构协议 *不* 属于基础设施 (它们
描述的是弹性力学概念), 因此放在 :mod:`soptx.protocols`. 把它们排除在外, 是为了不让
本包仅因 "人人都可以导入" 而变成领域类型的堆放处.
"""

from .cuda_memory import (
    is_out_of_memory,
    limit_cpu_memory,
    limit_cuda_memory,
    limit_cuda_memory_to_free,
    record_cuda_out_of_memory,
    record_out_of_memory,
)
from .logging import BaseLogged
from .mpi_runtime import ensure_mpi_initialized
from .profiling import blas_environment, measure
from .results import SolverResult
from .timing import timer

__all__ = [
    "BaseLogged",
    "SolverResult",
    "blas_environment",
    "ensure_mpi_initialized",
    "is_out_of_memory",
    "limit_cpu_memory",
    "limit_cuda_memory",
    "limit_cuda_memory_to_free",
    "measure",
    "record_cuda_out_of_memory",
    "record_out_of_memory",
    "timer",
]
