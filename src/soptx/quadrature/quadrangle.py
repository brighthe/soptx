# 移植自 brighthe/fealpy ``fealpy/quadrature/quadrangle.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""四边形上的张量积 Gauss--Legendre 积分公式."""

from ..backend import backend_manager as bm
from .gauss_legendre import GaussLegendreQuadrature


class QuadrangleQuadrature(GaussLegendreQuadrature):
    """两个方向都取 ``n = index`` 点 Gauss--Legendre 公式的张量积.

    对各方向次数都不超过 ``2n - 1`` 的多项式精确.
    """
    def make(self, index: int):
        """生成张量积积分点与权重.

        Returns
        -------
        tuple
            ``((bcs, bcs), weights)``: 两个方向的一维重心坐标 ``(n, 2)``, 以及
            ``(n, n)`` 的权重矩阵 (未展平).
        """
        bcs, ws = super().make(index)
        weights = bm.tensordot(ws, ws, axes=0)
        return (bcs, bcs), weights
