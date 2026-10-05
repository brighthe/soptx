# 移植自 brighthe/fealpy ``fealpy/quadrature/tensor_product.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""一维积分公式的张量积."""

from ..backend import backend_manager as bm
from .quadrature import Quadrature

class TensorProductQuadrature(Quadrature):
    """一维积分公式的张量积.

    Parameters
    ----------
    qfs : sequence of Quadrature or Quadrature
        ``TD`` 为 None 时是各方向的一维公式序列; 否则是一个一维公式, 各方向共用.
    TD : int, optional
        方向数. 默认 None, 即取 ``len(qfs)``.

    Notes
    -----
    ``quadpts`` 为各方向一维重心坐标的元组, ``weights`` 为各方向权重外积展平后的
    一维张量. 本类不调用基类构造函数, 没有 ``dtype`` 与 ``device`` 属性.
    """
    def __init__(self, qfs, TD=None):

        if TD is None:
            TD = len(qfs) #积分公式的个数
            self.quadpts = () # 空元组
            weights = ()
            for i, qf in enumerate(qfs):
                bcs, ws = qf.get_quadrature_points_and_weights()
                self.quadpts += (bcs, )
                weights += (ws, )
        else: # TD 是一个整数, qfs 是一个积分公式
            bcs, ws = qfs.get_quadrature_points_and_weights()
            self.quadpts = TD*(bcs, ) 
            weights = TD*(ws, )

        # 构造 einsum 运算字符串
        s0 = 'abcdef'
        s = ''
        for i in range(TD):
            s = s + s0[i]
            if i < TD-1:
                s = s + ', '
        s = s + '->' + s0[:TD]
        self.weights = bm.einsum(s, *weights).reshape(-1)

    def number_of_quadrature_points(self):
        """积分点总数, 即各方向点数之积."""
        n = self.weights.shape[0]
        return n 
