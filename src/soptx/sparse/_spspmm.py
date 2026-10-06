# 移植自 brighthe/fealpy ``fealpy/sparse/_spspmm.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""稀疏--稀疏矩阵乘法的通用实现, 后端未提供 ``csr_spspmm`` 时使用.

numpy 与 pytorch 后端都提供 ``csr_spspmm``, 本模块目前只在 COO 格式且后端缺少
专用内核时才会用到.
"""

from typing import Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT

_Size = Tuple[int, ...]


def _shape_check(spshape1: _Size, spshape2: _Size):
    if len(spshape1) != 2 or len(spshape2) != 2:
        raise ValueError("Sparse tensors to matmul must be both 2-D for sparse dims, "
                        f"but got shape {spshape1} and {spshape2}")
    if spshape1[1] != spshape2[0]:
        raise ValueError("Incompatible shapes detected in "
                         "sparse-sparse matrix multiplication, "
                        f"got shape {spshape1} and {spshape2}.")


def spspmm_coo(indices1: _DT, values1: _DT, spshape1: _Size,
               indices2: _DT, values2: _DT, spshape2: _Size) -> Tuple[_DT, _DT, _Size]:
    """两个 COO 矩阵相乘: 对中间维逐个取外积后拼接, 结果未合并重复索引.

    Returns
    -------
    tuple
        ``(索引, 值, 形状)``.

    Raises
    ------
    ValueError
        稀疏维不是二维、形状不相容, 或两者的稠密维不同.
    """
    _shape_check(spshape1, spshape2)

    structure = values1.shape[:-1]
    if values2.shape[:-1] != structure:
        raise ValueError(f"the dense shape of matrix2 ({values2.shape[:-1]}) "
                         f"must match that of matrix1 {structure}")

    size = spshape1[1]
    indices_list = []
    values_list = []

    for i in range(size):
        left_col_flag = (indices1[1, :] == i)

        if not bm.any(left_col_flag, axis=0):
            continue

        right_row_flag = (indices2[0, :] == i)

        if not bm.any(right_row_flag, axis=0):
            continue

        row = indices1[0, left_col_flag]
        col = indices2[1, right_row_flag]
        nnz = col.shape[0] * row.shape[0]
        idx = bm.meshgrid(row, col, indexing='ij')
        idx = bm.reshape(bm.stack(idx, axis=0), (2, nnz))
        val1 = values1[..., left_col_flag]
        val2 = values2[..., right_row_flag]

        val = bm.einsum('...i, ...j -> ...ij', val1, val2).reshape(*structure, nnz)
        indices_list.append(idx)
        values_list.append(val)

    indices = bm.concat(indices_list, axis=1)
    values = bm.concat(values_list, axis=-1)
    return indices, values, (spshape1[0], spshape2[1])
