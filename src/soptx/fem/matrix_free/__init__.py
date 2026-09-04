"""Matrix-Free 算子链: EA 懒装配算子、线性系统封装与一步正向求解.

依赖方向为 solve → operator → (soptx.fem.analyzers), 以及
solve → soptx.solvers. 重叠加权内积与它的 CG 装配点与具体物理无关,
不在本包, 见 :mod:`soptx.solvers.overlap`.

本包两个模块都不导入 mpi4py, 因此全部 eager 导出.
"""

from .operator import ElasticityEAOperator
from .solve import (
    PreparedLinearSystem,
    solve_ea_system,
    solve_matrix_free_system,
    solver_diagnostics,
)

__all__ = [
    "ElasticityEAOperator",
    "PreparedLinearSystem",
    "solve_ea_system",
    "solve_matrix_free_system",
    "solver_diagnostics",
]
