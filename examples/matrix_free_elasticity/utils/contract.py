"""Stage-1 numeric contract for the Matrix-Free elasticity baseline.

Every tolerance, default value and supported range used by ``run.py``,
``validate.py`` and ``sync_results.py`` is defined here exactly once, so a
tightened gate can never be applied on one side of the pipeline only.

The *shape* of the summary those numbers end up in belongs to :mod:`schema`,
which also owns ``SCHEMA_VERSION``.

This module must stay free of FEALPy, SOPTX and mpi4py imports: the evidence
tooling has to run on machines without an MPI runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE = "soptx/matrix-free-elasticity/stage-1"

SUPPORTED_DIMENSIONS = (2, 3)
SUPPORTED_RANKS = (1, 2)
SUPPORTED_DEGREES = (1,)
OPERATOR_LEVELS = ("ea", "fa")

OPERATOR_STORAGE = {
    "ea": "cached-element-matrices",
    "fa": "global-csr",
}
DISTRIBUTED_REPRESENTATION = "equal-status-overlapping-copies"

DEFAULT_DIMENSION = 3
DEFAULT_DEGREE = 1
DEFAULT_RESOLUTION = 4
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_RTOL = 1.0e-10
DEFAULT_ATOL = 1.0e-12
RESIDUAL_REFRESH = 20
REFERENCE_RANDOM_SEED = 20260727

# Single-run gates, checked by both run.py and validate.py.
BOUNDARY_ABSOLUTE_TOL = 1.0e-12
MATVEC_RELATIVE_TOL = 1.0e-12
SYMMETRY_RELATIVE_TOL = 1.0e-12
EXPLICIT_SOLUTION_RELATIVE_TOL = 1.0e-8

# Cross-run gates, checked by validate.py.
PARALLEL_SOLUTION_RELATIVE_TOL = 1.0e-9
EA_FA_SOLUTION_RELATIVE_TOL = 1.0e-9
PARALLEL_L2_DIFFERENCE_TOL = 1.0e-10
MINIMUM_FINAL_L2_ORDER = 1.5

# Coarse/medium/fine refinements driven by validate.py.
REFINEMENTS = {
    2: (8, 16, 32),
    3: (4, 8, 16),
}

# Lower bound used whenever a norm appears in a denominator.
NORM_FLOOR = 1.0e-30


def residual_limit(
    rhs_norm: float,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Absolute residual a converged run must reach for the given RHS."""

    return max(atol, rtol * rhs_norm)


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved single-run specification."""

    dimension: int
    degree: int
    resolution: tuple[int, ...]
    operator_level: str
    benchmark: bool
    max_iterations: int
    rtol: float
    atol: float
    output_path: Path
    summary_path: Path
    solution_path: Path

    @property
    def operator_storage(self) -> str:
        return OPERATOR_STORAGE[self.operator_level]

    def residual_limit(self, rhs_norm: float) -> float:
        return residual_limit(rhs_norm, rtol=self.rtol, atol=self.atol)
