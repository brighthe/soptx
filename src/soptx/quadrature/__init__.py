# 移植自 brighthe/fealpy ``fealpy/quadrature/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""参考单元上的数值积分公式.

积分点一律以重心坐标给出 (张量积公式为各方向一维重心坐标的元组), 权重已按参考
单元测度归一化, 总和为 1.
"""

from .quadrature import Quadrature

from .gauss_legendre import GaussLegendreQuadrature
from .gauss_lobatto import GaussLobattoQuadrature
from .triangle import TriangleQuadrature
from .quadrangle import QuadrangleQuadrature
from .tetrahedron import TetrahedronQuadrature
from .tensor_product import TensorProductQuadrature
