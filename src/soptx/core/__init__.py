"""SOPTX 各子系统共用的底层基础设施.

这里的内容都与具体领域无关: 日志、计时、按阶段的耗时与内存记录、简单的结果记录、MUMPS 侧的 MPI 激活钩子,
以及证据工具在没有 MPI runtime 时也要复现的数值缺省值.

分析器要求 ``pde`` 与 ``interpolation_scheme`` 满足的结构协议 *不* 属于基础设施 (它们
描述的是弹性力学概念), 因此放在 :mod:`soptx.protocols`. 把它们排除在外, 是为了不让
本包仅因 "人人都可以导入" 而变成领域类型的堆放处.
"""

from .logging import BaseLogged
from .mpi_runtime import ensure_mpi_initialized
from .profiling import measure
from .results import SolverResult
from .timing import timer

__all__ = [
    "BaseLogged",
    "SolverResult",
    "ensure_mpi_initialized",
    "measure",
    "timer",
]
