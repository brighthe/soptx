# -*- coding: utf-8 -*-
"""物理梯度: 基函数算子 B 及其转置 B^T 的张量收缩."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


def physical_gradient(u_E: TensorLike,
                    *,
                    reference_grad: TensorLike,
                    jacobi_inverse: TensorLike,
                ) -> TensorLike:
    """由单元自由度求各积分点处的位移物理梯度 grad u.

    Parameters
    ----------
    u_E : (NC, ldof, GD[, NB]) 的单元自由度向量.
    reference_grad : (NQ, ldof, TD) 的参考单元基函数梯度.
    jacobi_inverse : (NC, NQ, TD, GD) 的 Jacobi 矩阵逆.

    Returns
    -------
    grad_u : (NC, NQ, GD, GD[, NB]) 的物理梯度.
    """
    # 参考单元上的位移梯度 d u_d / d xi_r: (NC, NQ, TD, GD[, NB])
    grad_ref_u = bm.einsum('qir, cid... -> cqrd...', reference_grad, u_E)

    # 物理坐标下的位移梯度 d u_d / d x_b: (NC, NQ, GD, GD[, NB])
    grad_u = bm.einsum('cqrb, cqrd... -> cqdb...', jacobi_inverse, grad_ref_u)

    return grad_u


def physical_gradient_transpose(s_Q: TensorLike,
                            *,
                            reference_grad: TensorLike,
                            jacobi_inverse: TensorLike,
                        ) -> TensorLike:
    """把与 grad u 共轭的积分点量送回单元自由度, 即 ``physical_gradient`` 的转置.

    Parameters
    ----------
    s_Q : (NC, NQ, GD, GD) 或 (NC, NQ, GD, GD, NB) 的与 grad u 共轭的量, 与
        ``physical_gradient`` 的输出同形状.
    reference_grad : (NQ, ldof, TD) 的参考单元基函数梯度, 必须与
        ``physical_gradient`` 用的是同一份.
    jacobi_inverse : (NC, NQ, TD, GD) 的 Jacobi 矩阵逆, 必须与 ``physical_gradient``
        用的是同一份.

    Returns
    -------
    y_E : (NC, ldof, GD) 或 (NC, ldof, GD, NB) 的单元自由度向量, 布局与
        ``physical_gradient`` 的输入相同, 可直接交给 ``ElementRestriction.scatter_add``.

    Notes
    -----
    对应分解中的 B^T. 两个收缩是 ``physical_gradient`` 那两个倒序, 各自对调自由下标
    与求和下标; 对积分点 q 的求和 (数值积分) 在第二个收缩里完成, 积分权重已由
    ``LinearElasticQFunction`` 乘进 ``s_Q``.
    """
    # 参考坐标下的共轭量: (NC, NQ, TD, GD[, NB])
    s_ref = bm.einsum('cqrb, cqdb... -> cqrd...', jacobi_inverse, s_Q)

    # 对积分点求和, 送回单元自由度: (NC, ldof, GD[, NB])
    y_E = bm.einsum('qir, cqrd... -> cid...', reference_grad, s_ref)

    return y_E


def physical_basis_gradients(*,
                            reference_grad: TensorLike,
                            jacobi_inverse: TensorLike,
                        ) -> TensorLike:
    """展开每单元每积分点的物理基函数梯度, 形状 (NC, NQ, ldof, GD).

    这正是 PA 刻意不常驻的那个量, 只在取对角时作为临时量用一次: 取对角每次求解只做
    一遍, 而 ``physical_gradient`` 每次 matvec 都要走.

    Parameters
    ----------
    reference_grad : (NQ, ldof, TD) 的参考单元基函数梯度.
    jacobi_inverse : (NC, NQ, TD, GD) 的 Jacobi 矩阵逆.
    """
    return bm.einsum('qir, cqrb -> cqib', reference_grad, jacobi_inverse)
