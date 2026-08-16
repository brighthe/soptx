"""Stage-1 numeric contract for the Matrix-Free elasticity baseline.

Every tolerance, default value and supported range used by this pipeline and by
``examples/matrix_free_elasticity``'s demo scripts is defined here exactly once,
so a tightened gate can never be applied on one side only.

Three modules could plausibly own these numbers; two of them must not:

- :mod:`soptx.numerics` owns *solver defaults*.  Its docstring is explicit that
  a number encoding an **acceptance gate** belongs to the study defining that
  gate, not to the solver, so the tolerances below stay out of it.  The solver
  defaults are re-exported here instead, so every consumer reads them from one
  place and the gate helpers can state a convergence criterion in the same
  numbers the solver actually used.
- ``schema`` owns the *shape* of the summary these numbers end up in, plus
  ``SCHEMA_VERSION``.  Values live here, layout lives there.

That leaves this package, which is where the gate is enforced.  The numbers sit
next to ``validate.py`` rather than in the example, because the example is two
demo scripts and this pipeline is also the fealpy fork's pre-merge gate.

This module must stay free of FEALPy and mpi4py imports: the evidence tooling
has to run on machines without an MPI runtime.  :mod:`soptx.numerics` is safe
precisely because it carries no such imports either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soptx.numerics import (
    DEFAULT_ATOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RTOL,
    NORM_FLOOR,
    RESIDUAL_REFRESH,
)


STAGE = "soptx/matrix-free-elasticity/stage-1"

SUPPORTED_DIMENSIONS = (2, 3)
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
REFERENCE_RANDOM_SEED = 20260727

# Single-run gates, checked by both run.py and validate.py.
BOUNDARY_ABSOLUTE_TOL = 1.0e-12
MATVEC_RELATIVE_TOL = 1.0e-12
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


def residual_limit(
    rhs_norm: float,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Absolute residual a converged run must reach for the given RHS."""

    return max(atol, rtol * rhs_norm)


def matvec_reference_gates(matvec: dict) -> dict[str, bool]:
    """把 ``soptx.fem.verification.serial_references`` 的产出逐项对上阈值.

    判据本身也只写一次, 不只是阈值: ``compare_lagrange.py`` 与 ``report.py`` 调
    同一个函数, 就不会出现"两边阈值相同但一边漏了正定性探针"这种漂移. 入参是
    纯 dict, 所以本模块仍然不碰 FEALPy.

    前两条是这个脚本的正题 —— EA 与 FA 在裸算子和施加边界条件后是否给出同一个
    结果, 两条走的是不同代码路径, 不能并成一条. 第三条是唯一一条不以 "FA 是对的"
    为前提的检查: 它问这个离散系统本身是否退化. 曾经并列的对称性判据已删除, 因为
    ``dirichlet_matvec`` 通过就蕴含了它 (FA 严格对称), 且它从来只在 FA 存在时才评估.
    """

    return {
        "raw_matvec": matvec["raw_relative_error"] <= MATVEC_RELATIVE_TOL,
        "dirichlet_matvec": (
            matvec["dirichlet_relative_error"] <= MATVEC_RELATIVE_TOL
        ),
        "positive_definite": matvec["random_vector_energy"] > 0.0,
    }


def explicit_solution_gate(relative_error: float) -> bool:
    """CG 解与 FA 直接解的相对差是否落在门禁内."""

    return relative_error <= EXPLICIT_SOLUTION_RELATIVE_TOL


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
