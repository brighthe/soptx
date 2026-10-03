# 移植自 brighthe/fealpy ``fealpy/quadrature/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from .quadrature import Quadrature

from .gauss_legendre import GaussLegendreQuadrature
from .gauss_lobatto import GaussLobattoQuadrature
from .triangle import TriangleQuadrature
from .quadrangle import QuadrangleQuadrature
from .tetrahedron import TetrahedronQuadrature
from .tensor_product import TensorProductQuadrature
