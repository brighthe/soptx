# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典参考实体的 Schema: 节点、边、三角形、四边形、四面体、六面体、三棱柱与四棱锥."""

from .hexahedron import HexahedronSchema, LagrangeHexahedronSchema
from .node import NodeSchema
from .prism import LagrangePrismSchema, PrismSchema
from .pyramid import LagrangePyramidSchema, PyramidSchema
from .quadrilateral import LagrangeQuadrilateralSchema, QuadrilateralSchema
from .edge import LagrangeEdgeSchema, EdgeSchema
from .tetrahedron import LagrangeTetrahedronSchema, TetrahedronSchema
from .triangle import LagrangeTriangleSchema, TriangleSchema
