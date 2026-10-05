# 移植自 brighthe/fealpy ``fealpy/mesh/mapping.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考实体到物理实体的映射所诱导的数值变换.

本模块收录依赖 Jacobi 矩阵的积分权与向量值变换. 这些操作属于参考实体与物理实体
之间的几何映射, 不同于加密、粗化、移动、重新剖分等改变网格状态的变换.
"""

from ..backend import Tensor, bm

__all__ = [
    "integral_transform",
    "piola_transform_covariant",
    "piola_transform_contravariant",
]


def check_jacobi_matrix(J: Tensor) -> None:
    """检查 Jacobi 矩阵是否至少二维, 否则抛 ``ValueError``."""
    if J.ndim < 2:
        raise ValueError("The Jacobian matrix must have at least 2 dimensions.")


def is_square_jacobi_matrix(J: Tensor) -> bool:
    """Jacobi 矩阵最后两维是否为方阵."""
    return J.shape[-2] == J.shape[-1]


def integral_transform(
    value: Tensor,
    J: Tensor,
) -> Tensor:
    """积分变换: 把值除以测度因子 ``W``.

    方阵时 ``W = det(J)``, 否则 ``W = sqrt(det(J^T J))``.

    Parameters
    ----------
    value : Tensor
        待变换的张量值函数的值, 形状 ``(...[, any_dim])``.
    J : Tensor
        Jacobi 矩阵, 形状 ``(..., phy_dim, ref_dim)``.

    Returns
    -------
    Tensor
        变换后的值, 形状 ``(...[, any_dim])``.
    """
    check_jacobi_matrix(J)

    if is_square_jacobi_matrix(J):
        W = bm.linalg.det(J)
    else:
        W = bm.sqrt(bm.linalg.det(bm.einsum("...xi, ...xj -> ...ij", J, J)))

    return value / W


def piola_transform_covariant(
    value: Tensor,
    J: Tensor,
) -> Tensor:
    """协变向量的 Piola 变换.

    方阵时为 ``J^{-T} v``, 否则为 ``J (J^T J)^{-1} v``.

    Parameters
    ----------
    value : Tensor
        待变换的向量值, 形状 ``(..., ref_dim)``.
    J : Tensor
        Jacobi 矩阵, 形状 ``(..., phy_dim, ref_dim)``.

    Returns
    -------
    Tensor
        变换后的值, 形状 ``(..., phy_dim)``.
    """
    check_jacobi_matrix(J)

    if is_square_jacobi_matrix(J):
        J_inv_T = bm.linalg.inv(J).mT
        return bm.einsum("...xj, ...j -> ...x", J_inv_T, value)
    else:
        G = bm.einsum("...xi, ...xj -> ...ij", J, J)
        G_inv = bm.linalg.inv(G)
        return bm.einsum("...xi, ...ij, ...j -> ...x", J, G_inv, value)


def piola_transform_contravariant(
    value: Tensor,
    J: Tensor,
) -> Tensor:
    """逆变向量的 Piola 变换: ``J v / W``, ``W`` 同 ``integral_transform``.

    Parameters
    ----------
    value : Tensor
        待变换的向量值, 形状 ``(..., ref_dim)``.
    J : Tensor
        Jacobi 矩阵, 形状 ``(..., phy_dim, ref_dim)``.

    Returns
    -------
    Tensor
        变换后的值, 形状 ``(..., phy_dim)``.
    """
    check_jacobi_matrix(J)

    if is_square_jacobi_matrix(J):
        W = bm.linalg.det(J)
    else:
        W = bm.sqrt(bm.linalg.det(bm.einsum("...xi, ...xj -> ...ij", J, J)))

    return bm.einsum("...xi, ...i -> ...x", J, value) / W
