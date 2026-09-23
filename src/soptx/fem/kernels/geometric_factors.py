# -*- coding: utf-8 -*-
"""单元上的几何因子 (geometric factors).

矩阵自由算子分解

    A = P^T G^T B^T D B G P

中的 B 一项要把参考单元上的导数换到物理坐标:

    grad_x phi = grad_xi phi . J^{-1}

等号右边两个因子的存储量级完全不同: 参考梯度 (NQ, ldof, TD) 与单元无关, 全网格共享
一份; J^{-1} 是 (NC, NQ, TD, GD), 每单元一份. 本模块只持后者, 前者留在
``ReferenceBasis`` 里 -- 与 MFEM 把 ``DofToQuad`` (参考量) 与 ``GeometricFactors``
(逐单元量) 分成两个结构、libCEED 把 ``CeedBasis`` 与几何 QData 分开喂给算子的切法一致.

分开的实际好处是常驻量能按 "带不带 NC" 分项报出来: PA 相对 EA 的存储优势正来自
每单元量 O(NQ * GD^2) 不带 ldof 维, 合在一个类里统计就看不出这条分界线.

积分权重与 Jacobi 行列式不在本模块内: 它们与相对密度系数融成每积分点一个标量, 由
``LinearElasticQFunction`` 持有, 随设计变量更新时重算的也正是那一份.
"""

from fealpy.typing import TensorLike


class GeometricFactors:
    """逐单元的几何因子 J^{-1}.

    Parameters
    ----------
    jacobi_inverse : (NC, NQ, TD, GD) 的几何因子, 即 d xi_r / d x_b.

    Notes
    -----
    本类只是这份数组的持有者, 不做运算. 用到它的张量收缩在 ``gradients`` 模块, 只吃
    数组: 由 ``PartialAssembly`` 取出本类的 ``jacobi_inverse`` 与 ``ReferenceBasis``
    的 ``grad`` 交给它. MFEM 同样如此 -- 积分子持有 ``GeometricFactors``, 调 kernel
    时只传原始数组.
    """

    def __init__(self, jacobi_inverse: TensorLike) -> None:
        if jacobi_inverse.ndim != 4:
            raise ValueError(
                "jacobi_inverse 必须是 (NC, NQ, TD, GD) 的四维数组, 得到 "
                f"shape={tuple(jacobi_inverse.shape)}"
            )

        self._jacobi_inverse = jacobi_inverse
        self._n_cells = int(jacobi_inverse.shape[0])
        self._n_quad = int(jacobi_inverse.shape[1])
        self._top_dim = int(jacobi_inverse.shape[2])
        self._geo_dim = int(jacobi_inverse.shape[3])

    @property
    def jacobi_inverse(self) -> TensorLike:
        """几何因子 J^{-1}, 形状 (NC, NQ, TD, GD)"""
        return self._jacobi_inverse

    @property
    def n_cells(self) -> int:
        """单元数 NC"""
        return self._n_cells

    @property
    def n_quad(self) -> int:
        """每单元积分点数 NQ"""
        return self._n_quad

    @property
    def top_dimension(self) -> int:
        """拓扑维数 TD"""
        return self._top_dim

    @property
    def geo_dimension(self) -> int:
        """几何维数 GD"""
        return self._geo_dim

    def persistent_bytes(self) -> int:
        """常驻内存字节数, 每单元 O(NQ * TD * GD).

        这是 PA 常驻量里唯一带 NC 的几何项; 与之相对, ``ReferenceBasis`` 那一份与网格
        规模无关.
        """
        return int(self._jacobi_inverse.nbytes)

    def __repr__(self) -> str:
        return (f"GeometricFactors(n_cells={self._n_cells}, "
                f"n_quad={self._n_quad}, top_dim={self._top_dim}, "
                f"geo_dim={self._geo_dim})")
