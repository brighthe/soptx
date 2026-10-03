"""子结构局部缩聚策略的统一公共入口.

依赖 PyTorch 的 PIML 策略 (``PIMLShapeReduction``, ``PIMLStiffnessReduction``) 由
模块级 ``__getattr__`` 惰性加载, 使 ``ExactSchurReduction`` 在未安装 torch 的环境
中仍可导入.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import (
    CondensationReductionAdapter,
    LocalReduction,
    LocalReductionBatchResult,
    LocalReductionResult,
    ReductionDiagnostics,
)
from .exact_schur import ExactSchurReduction

if TYPE_CHECKING:
    # 仅在类型检查期生效, 把惰性符号还给静态分析器; 运行期不执行.
    from .piml_shape import PIMLShapeReduction as PIMLShapeReduction
    from .piml_stiffness import PIMLStiffnessReduction as PIMLStiffnessReduction

_TORCH_EXPORTS = {
    "PIMLShapeReduction": (".piml_shape", "PIMLShapeReduction"),
    "PIMLStiffnessReduction": (".piml_stiffness", "PIMLStiffnessReduction"),
}

__all__ = [
    "CondensationReductionAdapter", "LocalReduction", "LocalReductionBatchResult",
    "LocalReductionResult", "ReductionDiagnostics", "ExactSchurReduction",
    "PIMLShapeReduction", "PIMLStiffnessReduction",
]


def __getattr__(name: str):
    try:
        module_name, object_name = _TORCH_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
