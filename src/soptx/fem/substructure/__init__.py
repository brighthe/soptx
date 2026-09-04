"""子结构静力缩聚: 网格管理, 精确缩聚, PIML 代理与全局接口装配."""

from .mesh import SubstructureMesh, SubstructurePrototype, build_substructures
from .condensation import (
    StaticCondensationBase,
    FEAStaticCondensation,
    StreamingShapeFunctionCondensation,
)
from .piml_surrogate import (
    PIMLSurrogateNet,
    PIMLStaticCondensation,
    ShapeFunctionSurrogateNet,
    ShapeFunctionCondensation,
    SurrogateContractError,
)
from .assembler import GlobalAssembler, InterfaceSystem
from .reductions import (
    CondensationReductionAdapter,
    ExactSchurReduction,
    LocalReduction,
    LocalReductionBatchResult,
    LocalReductionResult,
    PIMLShapeReduction,
    PIMLStiffnessReduction,
    ReductionDiagnostics,
)
from .streaming import (
    ElementStrainEnergyBatch,
    TraceStiffnessBatch,
    iter_exact_element_energy_batches,
    iter_exact_trace_stiffness_batches,
)
from .traces import FullTraceBasis, LinearCornerTraceBasis, TraceBasis
from .operator import InterfaceOperator
from .problem_adapter import (
    InterfaceConditions,
    project_problem_conditions_to_full_system,
    project_problem_conditions_to_interface_system,
    project_problem_conditions_to_macro_system,
    project_problem_conditions_to_nodes,
)
from .solve import solve_interface_system
from .case_setup import (
    set_random_seed,
    make_density_fields,
    sample_random_density,
    train_surrogate,
)

__all__ = [
    "SubstructureMesh",
    "SubstructurePrototype",
    "StaticCondensationBase",
    "FEAStaticCondensation",
    "StreamingShapeFunctionCondensation",
    "PIMLSurrogateNet",
    "PIMLStaticCondensation",
    "ShapeFunctionSurrogateNet",
    "ShapeFunctionCondensation",
    "SurrogateContractError",
    "LocalReduction",
    "LocalReductionBatchResult",
    "LocalReductionResult",
    "ReductionDiagnostics",
    "CondensationReductionAdapter",
    "ExactSchurReduction",
    "PIMLShapeReduction",
    "PIMLStiffnessReduction",
    "ElementStrainEnergyBatch",
    "TraceStiffnessBatch",
    "iter_exact_element_energy_batches",
    "iter_exact_trace_stiffness_batches",
    "GlobalAssembler",
    "TraceBasis",
    "FullTraceBasis",
    "LinearCornerTraceBasis",
    "InterfaceSystem",
    "InterfaceOperator",
    "InterfaceConditions",
    "project_problem_conditions_to_full_system",
    "project_problem_conditions_to_interface_system",
    "project_problem_conditions_to_macro_system",
    "project_problem_conditions_to_nodes",
    "solve_interface_system",
    "set_random_seed",
    "build_substructures",
    "make_density_fields",
    "sample_random_density",
    "train_surrogate",
]
