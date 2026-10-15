"""SOPTX 包根.

稳定对象一律由各自所属的子包导出, 包根只暴露版本号, 并且必须不带任何进程级
副作用的导入. 导入任何 ``soptx`` 子模块都会先执行本文件, 因此这里若 eager 写
``from mpi4py import MPI``, 就会触发 ``MPI_Init``, 使每个使用方都被迫初始化 MPI.

MUMPS 求解前的 MPI 上下文激活由
:func:`soptx.core.mpi_runtime.ensure_mpi_initialized` 负责, 在真正分派到
MUMPS 的求解点上显式调用.
"""

__version__ = "1.2.0.dev0"

__all__ = ["__version__"]
