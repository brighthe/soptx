# 移植自 brighthe/fealpy ``fealpy/sparse/_spmm.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""稀疏--稠密矩阵乘法的通用实现, 后端未提供专用内核时使用."""

from typing import Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT

_Size = Tuple[int, ...]


def _shape_check(spshape: _Size, xshape: _Size):
    if len(spshape) != 2:
        raise ValueError("Sparse tensor to matrix multiplication must be"
                        f"2-D in sparse dims, but got {spshape}")

    if len(xshape) == 1:
        if spshape[-1] != xshape[0]:
            raise ValueError("Incompatible shapes detected in "
                             "sparse-dense matrix-vector multiplication, "
                            f"{spshape}(sparse dims) and {xshape}")
    elif len(xshape) >= 2:
        if spshape[-1] != xshape[-2]:
            raise ValueError("Incompatible shapes detected in "
                             "sparse-dense matrix multiplication, "
                            f"{spshape}(sparse dims) and {xshape}")
    else:
        raise ValueError(f"Illegal vector shape {xshape} found in the "
                         "sparse-dense multiplication")


def spmm_coo(indices: _DT, values: _DT, spshape: _Size, x: _DT) -> _DT:
    """COO 矩阵乘以稠密向量或矩阵.

    Parameters
    ----------
    indices : TensorLike
        行列索引, 形状 ``(2, nnz)``.
    values : TensorLike
        非零元的值, 形状 ``(*batch, nnz)``.
    spshape : tuple of int
        稀疏矩阵形状 ``(M, K)``.
    x : TensorLike
        形状 ``(K, )`` 或 ``(*batch, K, N)``.

    Returns
    -------
    TensorLike
        形状 ``(*batch, M)`` 或 ``(*batch, M, N)``.

    Raises
    ------
    ValueError
        形状不相容.
    """
    _shape_check(spshape, x.shape)
    row = indices[0]
    col = indices[1]

    if x.ndim == 1:
        new_vals = values * x[col]
        shape = new_vals.shape[:-1] + (spshape[0], )
        result = bm.zeros(shape, dtype=x.dtype)
        result = bm.index_add(result, row, new_vals, axis=-1)
        return result

    else: # x.ndim >= 2
        new_vals = values[..., None] * x[..., col, :] # (*batch, nnz, x_col)
        shape = new_vals.shape[:-2] + (spshape[0], x.shape[-1])
        result = bm.zeros(shape, dtype=x.dtype)
        result = bm.index_add(result, row, new_vals, axis=-2)
        return result


def spmm_csr(crow: _DT, col: _DT, values: _DT, spshape: _Size, x: _DT) -> _DT:
    """CSR 矩阵乘以稠密向量或矩阵, 逐行计算.

    Parameters
    ----------
    crow, col : TensorLike
        压缩行指针与列索引.
    values : TensorLike
        非零元的值, 形状 ``(*batch, nnz)``.
    spshape : tuple of int
        稀疏矩阵形状 ``(M, K)``.
    x : TensorLike
        形状 ``(K, )`` 或 ``(*batch, K, N)``.

    Returns
    -------
    TensorLike
        形状 ``(M, )`` 或 ``(*batch, M, N)``.

    Raises
    ------
    ValueError
        形状不相容.
    """
    _shape_check(spshape, x.shape)
    nrow = spshape[0]
    unsqueezed = False

    if x.ndim == 1:
        x = x[:, None]
        unsqueezed = True

    shape = x.shape[:-2] + (nrow, x.shape[-1])
    result = bm.zeros(shape, dtype=x.dtype)

    for i in range(nrow):
        start = crow[i]
        end = crow[i + 1]
        r = bm.einsum('...i, ...ij -> ...j', values[..., start:end], x[..., col[start:end], :])
        # result[..., i, :] = r
        result = bm.set_at(result, (Ellipsis, i, slice(None)), r)


    if unsqueezed:
        result = result[..., 0]

    return result
