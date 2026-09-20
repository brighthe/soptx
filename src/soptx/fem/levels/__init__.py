# -*- coding: utf-8 -*-
"""装配层级 (assembly level) 扩展.

同一个离散算子按常驻形式分成若干层级, 每个层级一个类, 对外接口一致. 层级名到类的
映射由 ``registry`` 维护, 调用方用 ``create_level('fa'|'ea'|'pa'|'ua', ...)`` 构造.
当前已实现 FA (``FullAssembly``), EA (``ElementAssembly``), PA (``PartialAssembly``)
与 UA (``UnassembledAssembly``); LA 需要多 rank, 尚未加入.
"""

from .base import AssemblyLevelExtension
from .registry import available_levels, create_level, register_level
from .element import ElementAssembly
from .full import FullAssembly
from .partial import PartialAssembly
from .unassembled import UnassembledAssembly

__all__ = [
    "AssemblyLevelExtension",
    "ElementAssembly",
    "FullAssembly",
    "PartialAssembly",
    "UnassembledAssembly",
    "available_levels",
    "create_level",
    "register_level",
]
