"""Topology-optimization objective functions."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 运行期由下面的 ``__getattr__`` 惰性加载; 但静态分析器 (Pyright/Pylance) 不会
    # 解析模块级 ``__getattr__``, 于是这些类在 IDE 里退化成 ``Any``, 表现为无语义
    # 高亮、无补全、无类型检查. 这段仅在类型检查期生效的导入把符号还给分析器,
    # 运行期不执行, 因此惰性加载行为不受影响.
    from .augmented_lagrangian import (
        AugmentedLagrangianObjective as AugmentedLagrangianObjective,
    )
    from .compliance import ComplianceObjective as ComplianceObjective
    from .mechanism import CompliantMechanismObjective as CompliantMechanismObjective
    from .volume import VolumeObjective as VolumeObjective

_EXPORTS = {
    "AugmentedLagrangianObjective": (
        ".augmented_lagrangian",
        "AugmentedLagrangianObjective",
    ),
    "ComplianceObjective": (".compliance", "ComplianceObjective"),
    "CompliantMechanismObjective": (
        ".mechanism",
        "CompliantMechanismObjective",
    ),
    "VolumeObjective": (".volume", "VolumeObjective"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
