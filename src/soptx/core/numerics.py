"""Shared numeric defaults for iterative solvers and evidence tooling.

These constants are needed on both sides of a validation pipeline: by the
solvers themselves, and by the evidence tooling that has to reproduce a run's
convergence criteria without ever constructing one.  That tooling must be able
to run on machines without FEALPy or an MPI runtime, which is why this module
sits in :mod:`soptx.core` -- layer 0, deliberately free of runtime FEALPy and
mpi4py imports -- rather than under ``soptx.fem``, where importing anything
pulls in FEALPy through the package ``__init__``.

It is intentionally not re-exported from ``soptx.core.__init__``: consumers
import it by its full path, so the dependency shows up at every call site.

Numbers that encode an *acceptance gate* rather than a solver default do not
belong here; they belong to whichever example or study defines that gate.
"""

from __future__ import annotations


#: Iteration cap for the Krylov solvers in :mod:`soptx.solvers.overlap`.
DEFAULT_MAX_ITERATIONS = 1000

#: Relative residual tolerance.
DEFAULT_RTOL = 1.0e-10

#: Absolute residual tolerance.
DEFAULT_ATOL = 1.0e-12

#: Iterations between true-residual recomputations inside CG.
RESIDUAL_REFRESH = 20

#: Lower bound used whenever a norm appears in a denominator.
NORM_FLOOR = 1.0e-30


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RTOL",
    "NORM_FLOOR",
    "RESIDUAL_REFRESH",
]
