# 移植自 brighthe/fealpy ``fealpy/quadrature/quadrangle.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from ..backend import backend_manager as bm
from .gauss_legendre import GaussLegendreQuadrature


class QuadrangleQuadrature(GaussLegendreQuadrature):
    def make(self, index: int):
        bcs, ws = super().make(index)
        weights = bm.tensordot(ws, ws, axes=0)
        return (bcs, bcs), weights
