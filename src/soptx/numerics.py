"""Shared numeric defaults for iterative solvers and evidence tooling.

These constants are needed on both sides of a validation pipeline: by the
solvers themselves, and by the evidence tooling that has to reproduce a run's
convergence criteria without ever constructing one.  That tooling must be able
to run on machines without FEALPy or an MPI runtime, so this module is kept
free of every heavy import — it lives directly under ``soptx`` rather than
under ``soptx.fem`` for exactly that reason, since importing anything from
``soptx.fem`` pulls in FEALPy through its package ``__init__``.

Numbers that encode an *acceptance gate* rather than a solver default do not
belong here; they belong to whichever example or study defines that gate.
"""

from __future__ import annotations


#: Iteration cap for the Krylov solvers in :mod:`soptx.fem.solvers`.
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
