"""子结构静力缩聚: 网格管理, 精确缩聚, PIML 代理与全局接口装配."""

from .mesh import (
    SubstructureMesh,
    SubstructurePrototype,
    build_modulus_substructures,
    build_substructures,
)
from .condensation import (
    StaticCondensationBase,
    ExactSchurCondensation,
    StreamingShapeFunctionCondensation,
)
from .piml_surrogate import (
    ReducedStiffnessCondensation,
    ShapeFunctionCondensation,
    SurrogateContractError,
)
from .assembler import GlobalAssembler, InterfaceSystem
from .layout import HasGlobalDofs, InterfaceDofsView, StructuredSubstructureLayout
from .recovery import recover_full_displacement
from .reduction_adapter import normalize_local_reduction
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
from .independent_targets import IndependentPredictionDecoder
from .operator import InterfaceOperator
from .problem_adapter import (
    InterfaceConditions,
    project_problem_conditions_to_full_system,
    project_problem_conditions_to_interface_system,
    project_problem_conditions_to_macro_system,
    project_problem_conditions_to_nodes,
)
from .solve import (
    ConstrainedSolveResult,
    solve_constrained_system,
    solve_interface_system,
)
from .case_setup import (
    set_random_seed,
    make_density_fields,
    sample_random_density,
    train_reduced_stiffness_surrogate,
)

__all__ = [
    "SubstructureMesh",
    "SubstructurePrototype",
    "StaticCondensationBase",
    "ExactSchurCondensation",
    "StreamingShapeFunctionCondensation",
    "ReducedStiffnessCondensation",
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
    "StructuredSubstructureLayout",
    "GlobalAssembler",
    "IndependentPredictionDecoder",
    "TraceBasis",
    "FullTraceBasis",
    "LinearCornerTraceBasis",
    "InterfaceSystem",
    "InterfaceDofsView",
    "HasGlobalDofs",
    "normalize_local_reduction",
    "recover_full_displacement",
    "InterfaceOperator",
    "InterfaceConditions",
    "project_problem_conditions_to_full_system",
    "project_problem_conditions_to_interface_system",
    "project_problem_conditions_to_macro_system",
    "project_problem_conditions_to_nodes",
    "ConstrainedSolveResult",
    "solve_constrained_system",
    "solve_interface_system",
    "set_random_seed",
    "build_substructures",
    "build_modulus_substructures",
    "make_density_fields",
    "sample_random_density",
    "train_reduced_stiffness_surrogate",
]
