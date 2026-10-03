# 移植自 brighthe/fealpy ``fealpy/mesh/transform/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Mesh-state transformations.

This package contains algorithms that change a mesh state, such as refinement,
coarsening, movement, smoothing, and remeshing.  Reference-to-physical entity
mapping helpers belong to :mod:`fealpy.mesh.mapping` instead.
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
