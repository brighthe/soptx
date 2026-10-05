# 移植自 brighthe/fealpy ``fealpy/functional.py`` @ f474a5775, 移入 soptx.fem.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""积分子使用的数值积分内核.

形状记号: ``C`` 为实体数, ``Q`` 为积分点数, ``I``、``J`` 为局部基函数数, 基函数
张量 ``(C, Q, I, ...)`` 末尾的 ``...`` 为每个自由度的分量形状. 不依赖单元的基函数
(单纯形上的 Lagrange 基) 单元轴长度为 1.
"""

from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, CoefLike
from .coef import is_scalar, is_tensor, fill_axis


def integral(value: TensorLike, weights: TensorLike, measure: TensorLike, *,
             entity_type=False) -> TensorLike:
    """对积分点上的值做数值积分.

    Parameters
    ----------
    value : TensorLike
        积分点上的被积函数值, 形状 ``(..., C, Q)``.
    weights : TensorLike
        积分权重, 形状 ``(Q, )``.
    measure : TensorLike
        实体测度, 形状 ``(C, )``.
    entity_type : bool, optional
        为 True 时返回每个实体上的积分, 否则返回全部实体之和. 默认 False.

    Returns
    -------
    TensorLike
        ``entity_type`` 为 True 时形状 ``(..., C)``, 否则 ``(...)``.
    """
    subs = '...c' if entity_type else '...'
    return bm.einsum(f'c, q, ...cq -> {subs}', measure, weights, value)


def linear_integral(basis: TensorLike, weights: TensorLike, measure: TensorLike,
                    source: Optional[CoefLike]=None,
                    batched: bool=False) -> TensorLike:
    """线性型的单元积分 ``(f, v)``.

    Parameters
    ----------
    basis : TensorLike
        积分点上的检验基函数值, 形状 ``(C, Q, I, ...)``.
    weights : TensorLike
        积分权重, 形状 ``(Q, )``.
    measure : TensorLike
        实体测度, 形状 ``(C, )``.
    source : Number or TensorLike, optional
        源项, 默认 None (视为 1). 张量形状为 ``(C, )``、``(C, Q)`` 或
        ``(C, Q, ...)``; ``batched`` 为 True 时首轴为批量维. 函数须先经
        ``process_coef_func`` 求值, 直接传入会抛 ``TypeError``.
    batched : bool, optional
        源项是否带批量维. 默认 False.

    Returns
    -------
    TensorLike
        源项为 ``(C, Q, ...)`` 时形状 ``(C, I)``; 为 ``(C, )`` 或 ``(C, Q)`` 时
        形状 ``(C, I, ...)``. ``batched`` 为 True 时首轴为批量维.

    Raises
    ------
    TypeError
        源项不是数或张量.
    """
    if source is None:
        return bm.einsum('c, q, cq... -> c...', measure, weights, basis)

    if is_scalar(source):
        return bm.einsum('c, q, cq... -> c...', measure, weights, basis) * source

    elif is_tensor(source):
        dof_shape = basis.shape[3:]
        basis = basis.reshape(*basis.shape[:3], -1) # (C, Q, I, dof_numel)
        # 不依赖单元的基函数 (单纯形上的 Lagrange 基) 单元轴长度为 1. einsum 只按
        # 浮点运算量选择收缩路径, 会把该轴广播到 C, 物化出与网格同规模的中间量.
        # 先把权重并入基函数, 只剩两个操作数的收缩, 最大的中间量就是结果本身.
        cellwise_basis = basis.shape[0] != 1

        if source.ndim <= 2 + int(batched):
            source = fill_axis(source, 3 if batched else 2)
            if cellwise_basis:
                r = bm.einsum(f'c, q, cqid, ...cq -> ...cid', measure, weights, basis, source)
            else:
                kernel = basis[0] * weights[:, None, None]          # (Q, I, dof_numel)
                r = bm.einsum('...cq, qid -> ...cid', source, kernel) * measure[:, None, None]
            return bm.reshape(r, r.shape[:-1] + dof_shape)
        else:
            source = fill_axis(source, 4 if batched else 3)
            if cellwise_basis:
                return bm.einsum(f'c, q, cqid, ...cqd -> ...ci', measure, weights, basis, source)
            kernel = bm.swapaxes(basis[0], -1, -2) * weights[:, None, None]  # (Q, dof_numel, I)
            return bm.einsum('...cqd, qdi -> ...ci', source, kernel) * measure[:, None]

    else:
        raise TypeError(f"source should be int, float or TensorLike, but got {type(source)}.")


def bilinear_integral(basis1: TensorLike, basis2: TensorLike, weights: TensorLike,
                      measure: TensorLike,
                      coef: Optional[CoefLike]=None,
                      batched: bool=False) -> TensorLike:
    """双线性型的单元积分 ``(c u, v)``.

    Parameters
    ----------
    basis1 : TensorLike
        积分点上的第一组基函数值, 形状 ``(C, Q, I, ...)``.
    basis2 : TensorLike
        积分点上的第二组基函数值, 形状 ``(C, Q, J, ...)``.
    weights : TensorLike
        积分权重, 形状 ``(Q, )``.
    measure : TensorLike
        实体测度, 形状 ``(C, )``.
    coef : Number or TensorLike, optional
        系数, 默认 None (视为 1). 张量形状为 ``(C, )``、``(C, Q)`` 或
        ``(C, Q, ...)``, 去掉批量维后为 4 维时视为作用在两组分量之间的矩阵系数;
        ``batched`` 为 True 时首轴为批量维. 函数须先经 ``process_coef_func``
        求值, 直接传入会抛 ``TypeError``.
    batched : bool, optional
        系数是否带批量维. 默认 False.

    Returns
    -------
    TensorLike
        形状 ``(C, I, J)``; ``batched`` 为 True 时为 ``(B, C, I, J)``.

    Raises
    ------
    TypeError
        系数不是数或张量.
    """
    basis1 = basis1.reshape(*basis1.shape[:3], -1) # (C, Q, I, dof_numel)
    basis2 = basis2.reshape(*basis2.shape[:3], -1) # (C, Q, J, dof_numel)

    if coef is None:
        return bm.einsum(f'q, c, cqid, cqjd -> cij', weights, measure, basis1, basis2)

    if is_scalar(coef):
        return bm.einsum(f'q, c, cqid, cqjd -> cij', weights, measure, basis1, basis2) * coef

    elif is_tensor(coef):
        ndim = coef.ndim - int(batched)
        if ndim == 4:
            return  bm.einsum(f'q, c, cqid, cqjn, ...cqdn -> ...cij', weights, measure, basis1, basis2, coef)
        else:
            coef = fill_axis(coef, 4 if batched else 3)
            return bm.einsum(f'q, c, cqid, cqjd, ...cqd -> ...cij', weights, measure, basis1, basis2, coef)
        
    else:
        raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")

def get_semilinear_coef(value:TensorLike, coef: Optional[CoefLike]=None, batched: bool=False):
    """把系数乘到半线性项的值上, 张量系数先在末尾补轴以便广播.

    Parameters
    ----------
    value : TensorLike
        半线性项在积分点上的值.
    coef : Number or TensorLike, optional
        系数. 注意 None 时会计算 ``None * value`` 并抛 ``TypeError``.
    batched : bool, optional
        系数是否带批量维. 默认 False.

    Returns
    -------
    TensorLike
        乘以系数后的值.

    Raises
    ------
    TypeError
        系数不是数或张量.
    """

    if coef is None:
        return coef * value

    if is_scalar(coef):
        return coef * value

    if is_tensor(coef):
        coef = fill_axis(coef, value.ndim + 1 if batched else value.ndim)
        return coef * value
    else:
        raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")
