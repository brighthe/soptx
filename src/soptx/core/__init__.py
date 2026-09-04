"""Low-level infrastructure shared by SOPTX subsystems.

Everything here is domain-free: logging, timing, a small result record, the
MUMPS-side MPI activation hook and the numeric defaults that the evidence
tooling has to reproduce without FEALPy or an MPI runtime.

The structural protocols an analyzer requires of its ``pde`` and
``interpolation_scheme`` are *not* infrastructure -- they name elasticity
concepts -- so they live in :mod:`soptx.protocols` instead.  Keeping them out
stops this package from becoming the place where domain types accumulate
merely because it is the layer everyone is allowed to import.
"""

from .logging import BaseLogged
from .mpi_runtime import ensure_mpi_initialized
from .results import SolverResult
from .timing import timer

__all__ = [
    "BaseLogged",
    "SolverResult",
    "ensure_mpi_initialized",
    "timer",
]
