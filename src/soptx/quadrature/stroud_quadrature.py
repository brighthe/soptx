# 移植自 brighthe/fealpy ``fealpy/quadrature/stroud_quadrature.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""单纯形上的 Stroud 锥积公式."""

import numpy as np
from scipy.special import j_roots, factorial
from ..backend import backend_manager as bm

class StroudQuadrature:
    """``dim`` 维单纯形上的 Stroud 锥积 (conical product) 公式.

    在超立方体上取各方向的 ``n`` 点 Gauss--Jacobi 公式做张量积, 再经 Duffy 型
    映射变到单纯形, 共 ``n ** dim`` 个积分点, 对次数不超过 ``2n - 1`` 的多项式
    精确. 积分点以重心坐标 ``(NQ, dim+1)`` 给出, 权重总和为 1.

    Parameters
    ----------
    dim : int
        单纯形维数.
    n : int
        每个方向的积分点数.
    """
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        p, self.weights = self._compute_quadrature()
        self.points = self._to_simplex(p)

        self.points  = bm.tensor(self.points, dtype=bm.float64)
        self.weights = bm.tensor(self.weights, dtype=bm.float64)

    def _to_simplex(self, points):
        d = self.dim
        shape = points.shape[:-1]
        bcs = np.zeros(shape+(d+1, ), dtype=np.float64)
        bcs[:, 0] = points[:, 0]
        for i in range(1, d):
            bcs[:, i] = points[:, i] * (1-bcs[:, :i].sum(axis=-1))
        bcs[:, d] = 1-bcs[:, :d].sum(axis=-1)
        return bcs

    def _compute_quadrature(self):
        d = self.dim
        n = self.n

        points = []
        weights = []
        for i in range(1, d+1):
            p, w, s = j_roots(n, d-i, 0, mu=True)
            points.append((p+1)/2)
            weights.append(w/s)
        points = np.meshgrid(*points)
        weights = np.meshgrid(*weights)

        points = np.array([p.flatten() for p in points]).T
        weights = np.prod([w.flatten() for w in weights], axis=0)#/factorial(d)
        return points, weights

    def get_points(self):
        """返回积分点的重心坐标."""
        return self.points

    def get_weights(self):
        """返回权重."""
        return self.weights

    def get_points_and_weights(self):
        """返回 ``(积分点, 权重)``."""
        return self.points, self.weights

    get_quadrature_points_and_weights = get_points_and_weights














