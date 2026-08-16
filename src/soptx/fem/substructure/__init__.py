"""子结构静力缩聚: 网格管理, 精确缩聚, PIML 代理与全局接口装配."""

from .mesh import SubstructureMesh, SubstructurePrototype
from .condensation import (
    StaticCondensationBase,
    FEAStaticCondensation,
)
from .piml_surrogate import (
    PIMLSurrogateNet,
    PIMLStaticCondensation,
    SurrogateContractError,
)
from .assembler import GlobalAssembler, InterfaceSystem
from .solve import solve_interface_system

__all__ = [
    "SubstructureMesh",
    "SubstructurePrototype",
    "StaticCondensationBase",
    "FEAStaticCondensation",
    "PIMLSurrogateNet",
    "PIMLStaticCondensation",
    "SurrogateContractError",
    "GlobalAssembler",
    "InterfaceSystem",
    "solve_interface_system",
]
