# 移植自 brighthe/fealpy ``fealpy/sparse/ops.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""稀疏矩阵的构造与拼接: 对角矩阵、单位矩阵、横向/纵向拼接与分块矩阵."""

from typing import overload, Optional, Union, Literal

from ..backend import backend_manager as bm
from ..backend import TensorLike
from .coo_tensor import COOTensor
from .csr_tensor import CSRTensor


@overload
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            *, index_dtype=None) -> CSRTensor: ...
@overload
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            format: Literal['csr'], *, index_dtype=None) -> CSRTensor: ...
@overload
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            format: Literal['coo'], *, index_dtype=None) -> COOTensor: ...
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            format: Optional[str] = 'csr', *, index_dtype=None):
    """由对角线数据构造稀疏矩阵, 约定同 ``scipy.sparse.spdiags``.

    ``data[k, j]`` 放在第 ``diags[k]`` 条对角线的第 ``j`` 列, 即位置
    ``(j - diags[k], j)``; 落在矩阵外的项与零值被丢弃.

    Parameters
    ----------
    data : TensorLike
        对角线数据, 形状 ``(num_diags, len_diags)``; 只有一条对角线时也可为一维.
    diags : TensorLike or int
        对角线编号: 0 为主对角线, ``k > 0`` 为第 ``k`` 条上对角线, ``k < 0`` 为
        第 ``-k`` 条下对角线.
    M, N : int
        矩阵形状.
    format : {'csr', 'coo'}, optional
        结果格式, 默认 'csr'.
    index_dtype : dtype, optional
        索引的整数类型, 默认 ``bm.int64``.

    Returns
    -------
    CSRTensor or COOTensor
        稀疏矩阵.

    Raises
    ------
    ValueError
        ``data`` 超过二维、行数与对角线数不符, 或 ``diags`` 有重复.
    TypeError
        ``diags`` 既不是张量也不是整数.
    """
    is_scalar = False
    index_dtype = bm.int64 if index_dtype is None else index_dtype
    if data.ndim > 2:
        raise ValueError(f'the data must be a 2-D tensor, but got {data.ndim}-D')

    if isinstance(diags, TensorLike):
        diags = diags.flatten()
        diags = bm.astype(diags, index_dtype)
        if len(diags) > 1:
            if data.shape[0] != len(diags):
                raise ValueError(f'number of diagonals data: {data.shape[0]} does not match the number of diags: {len(diags)}')

            if len(bm.unique(diags)) != len(diags):
                raise ValueError('diags array contains duplicate values')

            diags = diags[:, None]
            num_diags, len_diags = data.shape

        else:
            is_scalar = True

    elif isinstance(diags, int):
        is_scalar = True
    else:
        raise TypeError(f"diags must be a tensor or int ,but got {type(diags)}")

    if is_scalar:
        if data.ndim == 1 or data.shape[0] == 1:
            data = data.flatten()
            num_diags = 1
            len_diags = data.shape[0]
        else:
            raise ValueError(f'number of diagonals data: {data.shape[0]} does not match the number of diags: 1')

    diags_inds = bm.arange(len_diags, device=bm.get_device(data), dtype=index_dtype)
    row = diags_inds - diags

    mask = (row >= 0)
    mask &= (row < M)
    mask &= (diags_inds < N)
    mask &= (data != 0)
    row = row[mask]

    if is_scalar:
        col = diags_inds[mask]
    else:
        col = bm.tile(diags_inds, [num_diags])[mask.ravel()]

    data = data[mask]
    indices = bm.stack((row, col), axis=0)
    diag_tensor = COOTensor(indices, data, spshape=(M, N))

    if format == 'coo':
        return diag_tensor

    csr = diag_tensor.tocsr()
    if csr.crow.dtype == index_dtype and csr.col.dtype == index_dtype:
        return csr
    return CSRTensor(
        bm.astype(csr.crow, index_dtype),
        bm.astype(csr.col, index_dtype),
        csr.values,
        csr.sparse_shape,
    )

def vstack(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    """纵向拼接 CSR 矩阵.

    Parameters
    ----------
    blocks : list of CSRTensor or None
        一维的块列表, None 跳过; 各块须为 CSR 且列数相同 (不检查).
    format : {'csr', 'coo'}, optional
        结果格式, 默认 'csr'.
    dtype : dtype, optional
        结果的数值类型.

    Returns
    -------
    CSRTensor or COOTensor
        拼接后的矩阵.

    Raises
    ------
    ValueError
        ``blocks`` 为空或不是一维列表.
    """
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks must be no empty.')

    if any(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 1-D')

    M = len(blocks)
    col = []
    values = []
    nb = 0
    nr = 0
    for i in range(M):
        if blocks[i] == None:
            continue
        if nb == 0:
            blocks_crow = [blocks[i].crow]
            nnz = blocks[i].nnz
            fblocks_idx = i

        nr = nr + blocks[i]._spshape[0]
        col.append(blocks[i].indices)
        values.append(blocks[i].values)
        if nb > 0:
            blocks_crow.append(nnz + blocks[i].crow[1:])
            nnz = nnz + blocks[i].nnz
        nb = nb + 1
    indices = bm.concat(col, axis=0)
    crow = bm.concat(blocks_crow, axis=0)
    values = bm.concat(values, axis=0)
    if dtype != None:
        values = values.astype(dtype)

    A = CSRTensor(crow, indices, values, spshape=(nr, blocks[fblocks_idx]._spshape[1]))
    if format == 'coo':
        return A.tocoo()
    return A

def hstack(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    """横向拼接稀疏矩阵.

    Parameters
    ----------
    blocks : list of SparseTensor or None
        一维的块列表, None 跳过; 各块行数须相同 (不检查).
    format : {'csr', 'coo'}, optional
        结果格式, 默认 'csr'.
    dtype : dtype, optional
        结果的数值类型.

    Returns
    -------
    CSRTensor or COOTensor
        拼接后的矩阵.

    Raises
    ------
    ValueError
        ``blocks`` 为空或不是一维列表.
    """
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks must be no empty.')

    if any(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 1-D')

    M = len(blocks)
    row_list = []
    col_list = []
    values_list = []
    cum_col = 0
    nb = 0
    for i in range(M):
        if blocks[i] == None:
            continue

        if nb == 0:
            fblocks_idx = i


        row_list.append(blocks[i].nonzero_slice[0])
        col_list.append(cum_col + blocks[i].nonzero_slice[1])
        values_list.append(blocks[i].values) 
        cum_col += blocks[i]._spshape[1]
        nb = nb + 1        
    row = bm.concat(row_list, axis=0)
    col = bm.concat(col_list, axis=0)
    indices = bm.stack((row, col), axis=0)
    values = bm.concat(values_list, axis=0)
    if dtype != None:
        values = values.astype(dtype)

    A = COOTensor(indices, values, spshape=(blocks[fblocks_idx]._spshape[0], cum_col))
    if format=='csr':
        return A.tocsr()
    return A


def bmat(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    """由二维块列表组装分块稀疏矩阵, 用法仿 ``scipy.sparse.bmat``.

    Parameters
    ----------
    blocks : list of list of SparseTensor or None
        二维块列表, None 表示零块.
    format : {'csr', 'coo'}, optional
        结果格式, 默认 'csr'.
    dtype : dtype, optional
        结果的数值类型.

    Returns
    -------
    CSRTensor or COOTensor
        分块矩阵.

    Raises
    ------
    ValueError
        ``blocks`` 为空、不是二维列表, 或同一块行/块列的维数不一致.

    Notes
    -----
    含 None 块时按 COO 统一组装, 结果正确. 不含 None 时走另一条路径, 有误:
    单块行 (``1 x N``) 返回的是列表; 单块列 (``M x 1``, ``M > 1``) 只返回第一个块,
    且不报错. 仓库内的调用都含 None 块.
    """
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks cannot be empty.')

    if not all(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 2-D')

    if any(isinstance(item2, list) for item1 in blocks for item2 in item1):
        raise ValueError('Blocks must be 2-D')

    M = len(blocks)
    N = len(blocks[0])

    if all(None not in blocks[b] for b in range(M)):
        if N > 1:
            blocks = [[hstack(blocks[b], format=format, dtype=dtype) for b in range(M)]]
        if M > 1:
            A = vstack(blocks[0], format=format, dtype=dtype)
        else:
            A = blocks[0]
        if dtype is not None:
            A = A.astype(dtype)
        return A

    ii = []
    jj = []
    nnz = 0
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                if nnz == 0:
                    kwargs1 = bm.context(blocks[i][j].crow)
                    kwargs2 = bm.context(blocks[i][j].values)
                    brow_lengths = bm.zeros(M, **kwargs1)
                    bcol_lengths = bm.zeros(N, **kwargs1)
                nnz = nnz + blocks[i][j].nnz

                A = blocks[i][j].tocoo()
                blocks[i][j] = A
                if brow_lengths[i] == 0:
                    brow_lengths[i] = A._spshape[0]
                elif brow_lengths[i] != A._spshape[0]:
                    msg = (f'blocks[{i},:] has incompatible row dimensions. '
                           f'Got blocks[{i},{j}].shape[0] == {A._spshape[0]}, '
                           f'expected {brow_lengths[i]}.')
                    raise ValueError(msg)
                ii.append(i)
                jj.append(j)
                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A._spshape[1]
                elif bcol_lengths[j] != A._spshape[1]:
                    msg = (f'blocks[:,{j}] has incompatible column '
                           f'dimensions. '
                           f'Got blocks[{i},{j}].shape[1] == {A._spshape[1]}, '
                           f'expected {bcol_lengths[j]}.')
                    raise ValueError(msg)

    row_offsets = bm.concat((bm.tensor([0], **kwargs1), bm.cumsum(brow_lengths, axis=0)))
    col_offsets = bm.concat((bm.tensor([0], **kwargs1), bm.cumsum(bcol_lengths, axis=0)))

    shape = (row_offsets[-1], col_offsets[-1])

    data = bm.empty(nnz, **kwargs2)
    row = bm.empty(nnz, **kwargs1)
    col = bm.empty(nnz, **kwargs1)

    nnz = 0
    for i, j in zip(ii, jj):
        B = blocks[i][j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = bm.add(B.row, row_offsets[i])
        col[idx] = bm.add(B.col, col_offsets[j])
        nnz += B.nnz
    indices = bm.stack((row, col), axis=0)
    A = COOTensor(indices, data, spshape=shape)

    if format == 'csr':
        return A.tocsr()
    return A

@overload
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None) -> CSRTensor:...
@overload
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None, *, format: Literal['csr']) -> CSRTensor: ... 
@overload
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None, *, format: Literal['coo']) -> COOTensor: ... 
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None, *, format: Optional[str] = 'csr'):
    """构造指定对角线上全为 1 的稀疏矩阵.

    Parameters
    ----------
    M : int
        行数.
    N : int, optional
        列数, 默认等于 ``M``.
    diags : TensorLike or int, optional
        放 1 的对角线编号, 含义同 ``spdiags``, 默认 0 (主对角线).
    dtype : dtype, optional
        元素类型, 默认为后端的默认浮点类型 (float64).
    device : device, optional
        设备.
    format : {'csr', 'coo'}, optional
        结果格式, 默认 'csr'.

    Returns
    -------
    CSRTensor or COOTensor
        稀疏矩阵.
    """
    if N is None:
        N = M

    if isinstance(diags, TensorLike):
        nd = len(diags)
    else:
        nd = 1

    values = bm.ones((nd, M), dtype=dtype, device=device)
    return spdiags(values, diags=diags, M=M, N=N, format=format)
