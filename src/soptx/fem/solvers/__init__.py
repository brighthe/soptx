"""Finite-element solver workflows.

``matrix_free_analyzer`` is not re-exported here: it reaches
``soptx.fem.distributed`` and therefore ``mpi4py``, an optional extra.  Import
it as ``from soptx.fem.solvers.matrix_free_analyzer import ...``.  The
matrix-free *solver* below has no MPI import and is safe to export eagerly.
"""

from .huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from .lagrange_fem_analyzer import LagrangeFEMAnalyzer
from .matrix_free_solver import (
    PreparedLinearSystem,
    solve_matrix_free_system,
    solver_diagnostics,
    weighted_cg,
    weighted_norm,
)

__all__ = [
    "HuZhangMFEMAnalyzer",
    "LagrangeFEMAnalyzer",
    "PreparedLinearSystem",
    "solve_matrix_free_system",
    "solver_diagnostics",
    "weighted_cg",
    "weighted_norm",
]
