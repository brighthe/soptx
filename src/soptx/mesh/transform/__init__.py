# 移植自 brighthe/fealpy ``fealpy/mesh/transform/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""改变网格状态的变换.

本包收录改变网格状态的算法, 如加密、粗化、移动、光顺与重新剖分 (SOPTX 只移植了
一致加密). 参考单元到物理单元的映射工具在 :mod:`soptx.mesh.mapping` 中.
"""

from .uniform import (
    uniform_refine,
    uniform_refine_hexahedron,
    uniform_refine_prism,
    uniform_refine_quadrilateral,
    uniform_refine_edge,
    uniform_refine_tetrahedron,
    uniform_refine_triangle,
)

__all__ = [
    "uniform_refine",
    "uniform_refine_edge",
    "uniform_refine_triangle",
    "uniform_refine_quadrilateral",
    "uniform_refine_tetrahedron",
    "uniform_refine_prism",
    "uniform_refine_hexahedron",
]
