# 移植自 brighthe/fealpy ``fealpy/functionspace/functional.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""张量基函数生成与对称张量工具."""

import string
import numpy as np
from typing import Tuple

from itertools import combinations_with_replacement

from ..typing import TensorLike
from ..backend import backend_manager as bm
from .utils import tensor_basis


def generate_tensor_basis(basis: TensorLike, shape: Tuple[int, ...], dof_priority=True) -> TensorLike:
    """由标量空间的基函数生成张量空间的基函数.

    Parameters
    ----------
    basis : TensorLike
        标量空间的基函数, 形状 ``(..., ldof)``.
    shape : tuple of int
        每个自由度的分量形状.
    dof_priority : bool, optional
        为 True 时自由度优先排列, 否则分量优先. 默认 True.

    Returns
    -------
    TensorLike
        形状 ``(..., ldof*numel, *shape)``, ``numel`` 为 ``shape`` 的元素数.
    """
    kwargs = bm.context(basis)
    factor = tensor_basis(shape, **kwargs) # (numel, numel)
    # 计算张量积
    tb = bm.tensordot(basis, factor, axes=0) # (1, ldof ,ldof, numel, numel)
    ldof = basis.shape[-1]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        # 如果 dof_priority 为 True, 交换 ldof 和 numel 这两个维度的位置
        tb = bm.swapaxes(tb, -ndim-1, -ndim-2) # (1, ldof, numel, ldof, numel)

    tb = tb.reshape(basis.shape[:-1] + (numel*ldof,) + shape) # (1, ldof, ldof*numel, numel)

    return tb


def generate_tensor_grad_basis(grad_basis: TensorLike, shape: Tuple[int, ...], dof_priority=True) -> TensorLike:
    """由标量空间的基函数梯度生成张量空间的基函数梯度.

    Parameters
    ----------
    grad_basis : TensorLike
        标量空间的基函数梯度, 形状 ``(..., ldof, GD)``.
    shape : tuple of int
        每个自由度的分量形状.
    dof_priority : bool, optional
        为 True 时自由度优先排列, 否则分量优先. 默认 True.

    Returns
    -------
    TensorLike
        形状 ``(..., ldof*numel, *shape, GD)``, ``numel`` 为 ``shape`` 的元素数.
    """
    factor = tensor_basis(shape, dtype=grad_basis.dtype)
    s0 = "abcde"[:len(shape)]
    tb = bm.einsum(f'...jz, n{s0} -> ...jn{s0}z', grad_basis, factor)
    ldof, GD = grad_basis.shape[-2:]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = bm.swapaxes(tb, -ndim-2, -ndim-3)

    return tb.reshape(grad_basis.shape[:-2] + (numel*ldof,) + shape + (GD,))

def custom_next_permutation(arr, compare_function):
    """按 ``compare_function`` 给出的序把 ``arr`` 原地变为字典序的下一个排列.

    Returns
    -------
    bool
        存在下一个排列时为 True; 已是最大排列时为 False, ``arr`` 不变.
    """
    n = len(arr)
    i = n - 2
    while i >= 0 and compare_function(arr[i], arr[i + 1]) >= 0:
        i -= 1
    if i == -1:
        # 如果没有找到降序的元素, 说明当前排列已经是最大的排列
        return False
    # 从右向左查找第一个大于 arr[i] 的元素
    j = n - 1
    while compare_function(arr[j], arr[i]) <= 0:
        j -= 1
    # 交换 arr[i] 和 arr[j]
    arr[i], arr[j] = arr[j], arr[i]
    # 反转 arr[i+1:], 使其成为升序
    arr[i + 1:] = arr[i + 1:][::-1]
    return True

def span_array(arr, alpha):
    """计算张量积 ``arr^alpha``: 第 ``i`` 个向量自乘 ``alpha[i]`` 次后依次外积.

    Parameters
    ----------
    arr : TensorLike
        形状 ``(NC, l, d)``.
    alpha : TensorLike
        各向量的次数, 形状 ``(l, )``.

    Returns
    -------
    TensorLike
        形状 ``(NC,) + (d,) * sum(alpha)``.
    """
    N = bm.sum(alpha)
    s = string.ascii_lowercase[:N]
    ss = 'i'+',i'.join(s)
    s = ss+'->i'+s

    tup = (s, )
    for i in range(len(alpha)):
        for j in range(alpha[i]):
            tup = tup + (arr[:, i], )
    return bm.einsum(*tup)

def symmetry_span_array(arr, alpha):
    """计算 ``arr^alpha`` 的对称部分: 对各因子位置的全部不同排列取平均.

    Parameters
    ----------
    arr : TensorLike
        形状 ``(NC, l, d)``.
    alpha : TensorLike
        各向量的次数, 形状 ``(l, )``.

    Returns
    -------
    TensorLike
        与 ``span_array(arr, alpha)`` 同形状.
    """
    M = span_array(arr, alpha)

    N = bm.sum(alpha)
    idx = [i for i in range(N)]
    idx1 = []
    for count, value in enumerate(alpha):
        idx1.extend([count] * value)
    ret = bm.zeros_like(M) # TODO 可以优化
    count = 0
    while True:
        #for i in idx:
        #    ret += bm.transpose(M, 0, i+1) 
        ret += bm.transpose(M, (0, ) + tuple([i+1 for i in idx]))
        count += 1
        sss = custom_next_permutation(idx, lambda x, y : idx1[x]-idx1[y])
        if not sss:
            ret /= count
            break
    return ret

def symmetry_index(d, r, dtype=None, device=None):
    """``d`` 维 ``r`` 阶张量展平后, 其对称部分各独立分量的位置与重数.

    Parameters
    ----------
    d : int
        维数.
    r : int
        阶数.
    dtype : dtype, optional
        索引的整数类型, 默认 ``bm.int32``.
    device : device, optional
        设备.

    Returns
    -------
    symidx : TensorLike
        各独立分量 (指标非降的组合) 在展平张量中的位置.
    num : TensorLike
        各独立分量在完整张量中出现的次数.
    """
    dtype = dtype if dtype is not None else bm.int32
    symidx0 = bm.tensor(list(combinations_with_replacement(range(d), r)),
                        dtype=dtype, device=device)
    coe = bm.flip(d**bm.arange(r, dtype=dtype, device=device))

    symidx = bm.einsum('ij,j->i', bm.astype(symidx0, bm.float64), bm.astype(coe, bm.float64))
    symidx = bm.astype(symidx, dtype)

    midx = bm.multi_index_matrix(r, d-1)
    #midx0 = bm.zeros_like(midx) 
    #for i in range(d):
    #    midx0[:, i] = bm.sum(symidx0 == i, axis=1) 
    #print(midx0-midx)
    #midx = midx0

    P = bm.concatenate([bm.tensor([1],device=device), bm.cumprod(bm.arange(r+1, device=device)[1:], axis=0)],
                       axis=0, dtype=dtype)
    num = P[r]/bm.prod(P[midx], axis=1, dtype=bm.float64)
    return symidx, num
