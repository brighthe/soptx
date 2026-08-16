"""SOPTX public package root.

Stable objects are exported by their owning subpackages.  The root package
intentionally exposes only the project version.
"""

import os

# 自动匹配系统 libmumps 所链接的 OpenMPI ABI 并激活 MPI 上下文，防止调用 MUMPS 求解器时触发 MPI_Comm_f2c abort
os.environ.setdefault("MPI4PY_MPIABI", "openmpi")

try:
    from mpi4py import MPI
except ImportError:
    pass

__version__ = "1.1.0.dev0"

__all__ = ["__version__"]

