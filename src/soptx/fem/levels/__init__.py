# -*- coding: utf-8 -*-
"""装配层级 (assembly level) 扩展.

同一个离散算子按常驻形式分成若干层级, 每个层级一个类, 对外接口一致. 层级名到类的
映射由 ``registry`` 维护, 调用方用 ``create_level('fa'|'ea'|'pa'|'ua', ...)`` 构造.
当前已实现 FA (``FullAssembly``), EA (``ElementAssembly``), PA (``PartialAssembly``)
与 UA (``UnassembledAssembly``); LA 需要多 rank, 尚未加入.

``SharedReferenceElementAssembly`` 是 EA 在平移类结构化网格上的变体 (每类共享一份参考
单元矩阵), 不进注册表, 只能显式构造, 见 ``shared_reference``.
"""

from .base import AssemblyLevelExtension
from .registry import available_levels, create_level, register_level
from .element import ElementAssembly
from .full import FullAssembly
from .partial import PartialAssembly, quadrature_geometry
from .shared_reference import SharedReferenceElementAssembly
from .unassembled import UnassembledAssembly

__all__ = [
    "AssemblyLevelExtension",
    "ElementAssembly",
    "FullAssembly",
    "PartialAssembly",
    "SharedReferenceElementAssembly",
    "UnassembledAssembly",
    "available_levels",
    "create_level",
    "quadrature_geometry",
    "register_level",
]
