"""四范式求解路径适配器注册表."""

from __future__ import annotations

from .base import ExperimentContext, ParadigmSolver, SolveResult
from .lagrange import LagrangeSolver
from .pinn import PINNSolver
from .substructure import SubstructureSolver
from .piml import PIMLSolver

# 登记名到适配器类的映射. 顺序即 2x2 定位表的行列顺序: 先经典后代理, 先全局后局部.
SOLVER_REGISTRY: dict[str, type[ParadigmSolver]] = {
    LagrangeSolver.NAME: LagrangeSolver,
    PINNSolver.NAME: PINNSolver,
    SubstructureSolver.NAME: SubstructureSolver,
    PIMLSolver.NAME: PIMLSolver,
}

__all__ = [
    "ExperimentContext",
    "ParadigmSolver",
    "SolveResult",
    "LagrangeSolver",
    "PINNSolver",
    "SubstructureSolver",
    "PIMLSolver",
    "SOLVER_REGISTRY",
]
