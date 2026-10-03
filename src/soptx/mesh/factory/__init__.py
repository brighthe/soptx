# 移植自 brighthe/fealpy ``fealpy/mesh/factory/__init__.py`` @ f474a5775, 仅保留 SOPTX 用到的四类网格.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from .base import ClassicMeshView, register_classic_view, VIEW_REGISTRY
from .hexahedron_mesh import HexahedronMesh
from .quadrangle_mesh import QuadrangleMesh
from .tetrahedron_mesh import TetrahedronMesh
from .triangle_mesh import TriangleMesh

__all__ = [
    'ClassicMeshView',
    'register_classic_view',
    'VIEW_REGISTRY',
    'TriangleMesh',
    'QuadrangleMesh',
    'TetrahedronMesh',
    'HexahedronMesh',
]
