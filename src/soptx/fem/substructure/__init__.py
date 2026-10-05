"""子结构静力缩聚: 网格管理, 精确缩聚, PIML 代理与全局接口装配.

依赖 PyTorch 的 PIML 代理符号 (见 ``_TORCH_EXPORTS``) 由模块级 ``__getattr__`` 惰性
加载, 使精确缩聚路线在未安装 torch (``pyproject.toml`` 中的可选依赖 ``pinn``) 的
环境中仍可导入.
"""

from importlib import import_module
from typing import TYPE_CHECKING

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
    schur_complement,
)
from .assembler import GlobalAssembler, InterfaceSystem
from .interface_space import (
    INTERFACE_SPACE_KINDS,
    InterfaceSpace,
    assemble_interface_stiffness,
    build_interface_pattern,
    build_interface_space,
)
from .layout import HasGlobalDofs, InterfaceDofsView, StructuredSubstructureLayout
from .recovery import (
    recover_full_displacement,
    recover_full_displacement_batches,
)
from .reduction_adapter import normalize_local_reduction
from .reductions import (
    CondensationReductionAdapter,
    ExactSchurReduction,
    LocalReduction,
    LocalReductionBatchResult,
    LocalReductionResult,
    ReductionDiagnostics,
)
from .streaming import (
    ElementStrainEnergyBatch,
    InternalDisplacementBatch,
    LocalCondensationBatch,
    TraceStiffnessBatch,
    assemble_exact_interface_system,
    iter_exact_condensation_batches,
    iter_exact_element_energy_batches,
    iter_exact_internal_displacement_batches,
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
from .solve import (
    ConstrainedSolveResult,
    solve_constrained_system,
    solve_interface_system,
)

if TYPE_CHECKING:
    # 运行期由下面的 ``__getattr__`` 惰性加载; 但静态分析器 (Pyright/Pylance) 不会
    # 解析模块级 ``__getattr__``, 于是这些类在 IDE 里退化成 ``Any``, 表现为无语义
    # 高亮、无补全、无类型检查. 这段仅在类型检查期生效的导入把符号还给分析器,
    # 运行期不执行, 因此惰性加载行为不受影响.
    from .case_setup import (
        make_density_fields as make_density_fields,
        sample_random_density as sample_random_density,
        set_random_seed as set_random_seed,
        train_reduced_stiffness_surrogate as train_reduced_stiffness_surrogate,
    )
    from .independent_targets import (
        IndependentPredictionDecoder as IndependentPredictionDecoder,
    )
    from .piml_surrogate import (
        ReducedStiffnessCondensation as ReducedStiffnessCondensation,
        ShapeFunctionCondensation as ShapeFunctionCondensation,
        SurrogateContractError as SurrogateContractError,
    )
    from .reductions import (
        PIMLShapeReduction as PIMLShapeReduction,
        PIMLStiffnessReduction as PIMLStiffnessReduction,
    )

_TORCH_EXPORTS = {
    "ReducedStiffnessCondensation": (".piml_surrogate", "ReducedStiffnessCondensation"),
    "ShapeFunctionCondensation": (".piml_surrogate", "ShapeFunctionCondensation"),
    "SurrogateContractError": (".piml_surrogate", "SurrogateContractError"),
    "PIMLShapeReduction": (".reductions", "PIMLShapeReduction"),
    "PIMLStiffnessReduction": (".reductions", "PIMLStiffnessReduction"),
    "IndependentPredictionDecoder": (".independent_targets", "IndependentPredictionDecoder"),
    "set_random_seed": (".case_setup", "set_random_seed"),
    "make_density_fields": (".case_setup", "make_density_fields"),
    "sample_random_density": (".case_setup", "sample_random_density"),
    "train_reduced_stiffness_surrogate": (".case_setup", "train_reduced_stiffness_surrogate"),
}

__all__ = [
    "SubstructureMesh",
    "SubstructurePrototype",
    "StaticCondensationBase",
    "ExactSchurCondensation",
    "StreamingShapeFunctionCondensation",
    "schur_complement",
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
    "InternalDisplacementBatch",
    "LocalCondensationBatch",
    "TraceStiffnessBatch",
    "assemble_exact_interface_system",
    "iter_exact_condensation_batches",
    "iter_exact_element_energy_batches",
    "iter_exact_internal_displacement_batches",
    "iter_exact_trace_stiffness_batches",
    "StructuredSubstructureLayout",
    "GlobalAssembler",
    "INTERFACE_SPACE_KINDS",
    "InterfaceSpace",
    "assemble_interface_stiffness",
    "build_interface_pattern",
    "build_interface_space",
    "IndependentPredictionDecoder",
    "TraceBasis",
    "FullTraceBasis",
    "LinearCornerTraceBasis",
    "InterfaceSystem",
    "InterfaceDofsView",
    "HasGlobalDofs",
    "normalize_local_reduction",
    "recover_full_displacement",
    "recover_full_displacement_batches",
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


def __getattr__(name: str):
    try:
        module_name, object_name = _TORCH_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
