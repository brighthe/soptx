"""MUMPS 调用前的 MPI 上下文激活.

系统 libmumps 链接在 OpenMPI 上, 而 PyMUMPS 会对一个 Fortran 通信子句柄调用
``MPI_Comm_f2c``; 若此时进程内的 MPI 尚未初始化, 该转换会直接 abort 进程 ——
它是进程级 abort 而不是 Python 异常, 因此 ``try/except ImportError`` 拦不住。

导入 ``mpi4py.MPI`` 会执行 ``MPI_Init``, 这正是所需的激活动作。但它是一个
进程级副作用, 所以不放在 :mod:`soptx` 包根: 那会让每一次 ``import soptx.*``
(包括 :mod:`soptx.core.numerics` 与不依赖 MPI 的证据工具链) 都被迫初始化 MPI。
本模块把它收成一个显式调用, 只在真正分派到 MUMPS 的求解路径上触发。

ABI 由构建方式决定, 本模块不干预: 当前环境的 mpi4py 是直链构建
(``MPI.cpython-*.so`` 经 ``DT_NEEDED`` 绑定 ``libmpi.so.40``), ABI 在链接期
已固定, 运行时无从选择。若将来换成 ABI-generic 构建(运行时 ``dlopen`` 选库),
匹配应由部署层(conda ``activate.d``、作业启动脚本)负责, 而不是由库在导入时
改写进程环境变量。
"""

from __future__ import annotations


def ensure_mpi_initialized() -> bool:
    """确保 MPI 上下文已激活, 供 MUMPS 直接求解器安全调用.

    幂等: ``mpi4py.MPI`` 只在首次导入时执行 ``MPI_Init``, 之后重复调用等价于
    一次 ``sys.modules`` 命中。在 ``mpirun`` 下 MPI 通常已由 runtime 初始化,
    此时本函数同样是空操作。

    返回:
        bool: mpi4py 可用且 MPI 上下文已激活为 True; 环境未安装 mpi4py 为
        False —— 此时调用方若继续走 MUMPS, 失败应由 MUMPS 自己报出, 本函数
        不代它决定。
    """
    try:
        from mpi4py import MPI  # noqa: F401  导入即触发 MPI_Init
    except ImportError:
        return False
    return True


__all__ = ["ensure_mpi_initialized"]
