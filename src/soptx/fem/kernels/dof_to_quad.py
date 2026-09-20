# -*- coding: utf-8 -*-
"""自由度到积分点的插值算子 (dof-to-quad).

本模块实现矩阵自由算子分解

    A = P^T G^T B^T D B G P

中的 B 一项: E 向量 (单元自由度向量) 与 Q 向量 (积分点上的量) 之间的映射. 对位移
元而言, B 交出的是各积分点处的物理梯度 grad u, 其转置把与 grad u 共轭的量送回单元
自由度.

常驻的是参考单元上的基函数梯度加每单元的几何因子, 而不是每单元的物理梯度:

    grad_x phi = grad_xi phi . J^{-1}

参考梯度 (NQ, ldof, R) 与单元无关, 全网格共享一份; 每单元只留 J^{-1}
(NC, NQ, R, GD). 于是每单元常驻量是 O(NQ * GD^2), 不带 ldof 维 -- 这正是 PA 相对
EA (每单元 O((GD*ldof)^2)) 的存储优势所在, 阶数越高差距越大. 若改存每单元物理梯度
(NC, NQ, ldof, GD), 常驻量重新带上 ldof 维, 在任何阶数下都不会比 EA 更省.

与 libCEED 的 ``CeedBasis`` 及 MFEM 的 ``DofToQuad`` 加几何因子对应. 积分权重与
Jacobi 行列式不在本类内: 它们是逐点算子 D 的一部分, 由 ``QFunction`` 持有.
"""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


class DofToQuad:
    """B 算子: 单元自由度向量与积分点物理梯度之间的映射.

    Parameters
    ----------
    grad_ref : (NQ, ldof, R) 的参考单元基函数梯度, 全网格共享一份.
    jacobi_inverse : (NC, NQ, R, GD) 的几何因子, 即 d xi_r / d x_b.
    dof_permutation : (ldof, GD) 的整数数组, ``dof_permutation[i, d]`` 是标量自由度
        i 的第 d 个分量在单元自由度向量里的槽位. 由 ``flatten_indices`` 按张量空间的
        ``dof_priority`` 生成, 与 ``LinearElasticMaterial.strain_matrix`` 用的是同一
        套约定.

    Notes
    -----
    梯度的分量次序统一为 ``grad[c, q, d, b] = d u_d / d x_b``: 前一个是位移分量, 后
    一个是求导方向. ``QFunction`` 按同一约定读写.

    多列右端项的批量维一律在最后 (见 ``ElementRestriction``), 因此本类各方法的形状
    都写成 "单元维在前, 批量维在后", einsum 用尾置省略号带过批量维.
    """

    def __init__(self,
                grad_ref: TensorLike,
                jacobi_inverse: TensorLike,
                dof_permutation: TensorLike,
            ) -> None:
        self._grad_ref = grad_ref
        self._jacobi_inverse = jacobi_inverse

        # 正向按 (i, d) 的字典序取单元自由度向量, 反向把结果送回原槽位; 两者都是纯
        # gather, 不用 set_at, 免得在函数式后端上引入原地写
        perm = bm.reshape(dof_permutation, (-1, ))
        self._perm = perm
        self._inv_perm = bm.argsort(perm)

        self._n_quad = int(grad_ref.shape[0])
        self._local_dofs = int(grad_ref.shape[1])
        self._ref_dim = int(grad_ref.shape[2])
        self._geo_dim = int(jacobi_inverse.shape[-1])

    @property
    def grad_ref(self) -> TensorLike:
        """参考单元上的基函数梯度, 形状 (NQ, ldof, R)"""
        return self._grad_ref

    @property
    def jacobi_inverse(self) -> TensorLike:
        """几何因子 J^{-1}, 形状 (NC, NQ, R, GD)"""
        return self._jacobi_inverse

    @property
    def n_quad(self) -> int:
        """每单元积分点数 NQ"""
        return self._n_quad

    @property
    def local_dofs(self) -> int:
        """标量空间的单元局部自由度数 ldof"""
        return self._local_dofs

    @property
    def ref_dimension(self) -> int:
        """参考单元维数 R"""
        return self._ref_dim

    @property
    def geo_dimension(self) -> int:
        """几何维数 GD"""
        return self._geo_dim

    def to_canonical(self, u_E: TensorLike) -> TensorLike:
        """单元自由度向量 (NC, GD*ldof, ...) 重排成 (NC, ldof, GD, ...)"""
        u = u_E[:, self._perm]

        return bm.reshape(u, (u.shape[0], self._local_dofs, self._geo_dim)
                            + tuple(u.shape[2:]))

    def from_canonical(self, y: TensorLike) -> TensorLike:
        """(NC, ldof, GD, ...) 展平回单元自由度向量 (NC, GD*ldof, ...)"""
        y_flat = bm.reshape(y, (y.shape[0], self._local_dofs * self._geo_dim)
                            + tuple(y.shape[3:]))

        return y_flat[:, self._inv_perm]

    def gradient(self, u_E: TensorLike) -> TensorLike:
        """E 向量到 Q 向量: 求各积分点处的物理梯度.

        Parameters
        ----------
        u_E : (NC, GD*ldof) 或 (NC, GD*ldof, B) 的单元自由度向量, 批量维在后.

        Returns
        -------
        grad_u : (NC, NQ, GD, GD) 或 (NC, NQ, GD, GD, B) 的物理梯度,
            ``[c, q, d, b]`` 为 d u_d / d x_b.
        """
        u = self.to_canonical(u_E)

        # 先在参考单元上求导, 再用几何因子换到物理坐标; 两步都不展开 ldof x GD 的稠密
        # 小矩阵
        grad_ref_u = bm.einsum('qir, cid... -> cqrd...', self._grad_ref, u)

        return bm.einsum('cqrb, cqrd... -> cqdb...',
                        self._jacobi_inverse, grad_ref_u)

    def gradient_transpose(self, s_Q: TensorLike) -> TensorLike:
        """Q 向量到 E 向量: ``gradient`` 的转置.

        Parameters
        ----------
        s_Q : (NC, NQ, GD, GD) 或 (NC, NQ, GD, GD, B) 的与 grad u 共轭的量, 下标约定
            同 ``gradient``.

        Returns
        -------
        y_E : (NC, GD*ldof) 或 (NC, GD*ldof, B) 的单元自由度向量.
        """
        t = bm.einsum('cqrb, cqdb... -> cqrd...', self._jacobi_inverse, s_Q)
        y = bm.einsum('qir, cqrd... -> cid...', self._grad_ref, t)

        return self.from_canonical(y)

    def basis_gradients(self) -> TensorLike:
        """展开每单元每积分点的物理基函数梯度, 形状 (NC, NQ, ldof, GD).

        这正是本类刻意不常驻的那个量, 只在取对角时作为临时量用一次: 取对角每次求解
        只做一遍, 而 ``gradient`` 每次 matvec 都要走.
        """
        return bm.einsum('qir, cqrb -> cqib', self._grad_ref, self._jacobi_inverse)

    def persistent_bytes(self) -> int:
        """常驻内存字节数.

        参考梯度是全网格共享的一份, 与单元数无关; J^{-1} 是每单元的.
        """
        return int(self._grad_ref.nbytes) + int(self._jacobi_inverse.nbytes)

    def __repr__(self) -> str:
        n_cells = int(self._jacobi_inverse.shape[0])

        return (f"DofToQuad(n_cells={n_cells}, n_quad={self._n_quad}, "
                f"local_dofs={self._local_dofs}, ref_dim={self._ref_dim}, "
                f"geo_dim={self._geo_dim})")
