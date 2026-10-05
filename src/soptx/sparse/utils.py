# 移植自 brighthe/fealpy ``fealpy/sparse/utils.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""稀疏张量的形状检查与索引工具."""

from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike, Size


def check_shape_match(shape1: Size, shape2: Size):
    """两个完整形状不同时抛 ``ValueError``."""
    if shape1 != shape2:
        raise ValueError(f"shape mismatch: {shape1} != {shape2}")


def check_spshape_match(spshape1: Size, spshape2: Size):
    """两个稀疏维形状不同时抛 ``ValueError``."""
    if spshape1 != spshape2:
        raise ValueError(f"sparse shape mismatch: {spshape1} != {spshape2}")


def _dense_shape(values: Optional[TensorLike]):
    if values is None:
        return tuple()
    else:
        return values.shape[:-1]


def _dense_ndim(values: Optional[TensorLike]):
    if values is None:
        return 0
    else:
        return values.ndim - 1


def shape_to_strides(shape: Size, item_size: int):
    """按行优先 (C 序) 计算各轴的步长, 最后一轴步长为 ``item_size``."""
    strides = [item_size, ]

    for i in range(1, len(shape)):
        strides.append(strides[-1] * shape[-i])

    return tuple(reversed(strides))


def flatten_indices(indices: TensorLike, shape: Size) -> TensorLike:
    """把多维索引 ``(D, nnz)`` 按行优先展平为一维索引, 返回形状 ``(1, nnz)``."""
    nnz = indices.shape[-1]
    strides = shape_to_strides(shape, 1)
    kwargs = bm.context(indices)
    flatten = bm.zeros((nnz,), **kwargs)

    for d, s in enumerate(strides):
        flatten += indices[d, :] * s

    return flatten[None, ...]


def tril_coo(indices: TensorLike, values: TensorLike, k: int=0):
    """复制 COO 稀疏张量最后两个稀疏维中第 ``k`` 条对角线及以下的部分, 返回 ``(索引, 值)``."""
    tril_pos = (indices[-2] + k) >= indices[-1]
    new_indices = bm.copy(indices[:, tril_pos])
    new_values = bm.copy(values[..., tril_pos])

    return new_indices, new_values
