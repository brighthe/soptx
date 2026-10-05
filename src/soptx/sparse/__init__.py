# 移植自 brighthe/fealpy ``fealpy/sparse/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""跨后端的稀疏张量: ``COOTensor``、``CSRTensor`` 及类 scipy 的构造函数.

稀疏张量的形状分为 "稠密维" (``values`` 的前导轴, 用于批量) 与 "稀疏维" (由索引
表示), ``shape = dense_shape + sparse_shape``.
"""

from typing import overload, Optional, Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT
from ..backend import Size
from .sparse_tensor import SparseTensor
from .coo_tensor import COOTensor
from .csr_tensor import CSRTensor

from .ops import spdiags, speye



@overload
def coo_matrix(arg1: _DT, /, itype=None) -> COOTensor: ...
@overload
def coo_matrix(arg1: SparseTensor, /) -> COOTensor: ...
@overload
def coo_matrix(arg1: Size, /, *, itype=None, dtype=None, device=None) -> COOTensor: ...
@overload
def coo_matrix(arg1: Tuple[_DT, Tuple[_DT, ...]], /, *,
               shape: Optional[Size] = None) -> COOTensor: ...
def coo_matrix(arg1, /, *,
               shape: Optional[Size] = None,
               itype=None, dtype=None, device=None) -> COOTensor:
    """构造 COO (坐标, 又称 ijv 或三元组) 格式的稀疏张量, 不带批量维, 用法仿 scipy.

    支持以下几种构造方式:

    ``coo_matrix(D)``
        由稠密张量 ``D`` 的非零元构造.
    ``coo_matrix(S)``
        由另一个稀疏张量构造, 等价于 ``S.tocoo()``.
    ``coo_matrix((M, ...))``
        构造形状为 ``(M, ...)`` 的空张量.
    ``coo_matrix((data, (i, j, ...)), shape=(M, ...))``
        由非零值与各稀疏维的索引构造, 二维时 ``A[i[k], j[k]] = data[k]``;
        不给 ``shape`` 时由索引推断.

    Parameters
    ----------
    arg1 : TensorLike, SparseTensor or tuple
        构造来源, 见上.
    shape : tuple of int, optional
        稀疏维形状, 只用于 ``(data, (i, ...))`` 方式.
    itype : dtype, optional
        索引的整数类型.
    dtype : dtype, optional
        空张量的数值类型.
    device : device, optional
        空张量的设备.

    Returns
    -------
    COOTensor
        COO 格式稀疏张量.

    Raises
    ------
    TypeError
        参数组合不合法.
    """
    if isinstance(arg1, _DT):
        indices_tuple = bm.nonzero(arg1)
        indices = bm.stack(indices_tuple, axis=0)
        if itype is not None:
            indices = bm.astype(indices, itype)
        values = bm.copy(arg1[indices_tuple])
        return COOTensor(indices, values, arg1.shape)

    elif isinstance(arg1, (COOTensor, CSRTensor)):
        return arg1.tocoo()

    elif isinstance(arg1, (tuple, list)):
        if isinstance(arg1[0], int):
            ndim = len(arg1)
            if itype is None:
                itype = bm.int64
            indices = bm.empty((ndim, 0), dtype=itype, device=device)
            values = bm.empty((0,), dtype=dtype, device=device)
            return COOTensor(indices, values, spshape=arg1)

        elif isinstance(arg1[0], _DT) or arg1[0] is None:
            assert len(arg1) == 2
            values = arg1[0] # 非零元
            indices = bm.stack(arg1[1], axis=0)
            return COOTensor(indices, values, shape)

    raise TypeError(f"Error: Illegal combination of parameters")


@overload
def csr_matrix(arg1: _DT, /, itype=None) -> CSRTensor: ...
@overload
def csr_matrix(arg1: SparseTensor, /) -> CSRTensor: ...
@overload
def csr_matrix(arg1: Size, /, *, itype=None, dtype=None, device=None) -> CSRTensor: ...
@overload
def csr_matrix(arg1: Tuple[_DT, Tuple[_DT, _DT]], /, *,
               shape: Optional[Size] = None) -> CSRTensor: ...
@overload
def csr_matrix(arg1: Tuple[_DT, _DT, _DT], /, *,
               shape: Optional[Size] = None) -> CSRTensor: ...
def csr_matrix(arg1,
               shape: Optional[Size] = None,
               itype=None, dtype=None, device=None) -> CSRTensor:
    """构造 CSR (压缩稀疏行) 格式的稀疏矩阵, 不带批量维, 用法仿 scipy.

    支持以下几种构造方式:

    ``csr_matrix(D)``
        由二维稠密张量 ``D`` 的非零元构造.
    ``csr_matrix(S)``
        由另一个稀疏张量构造, 等价于 ``S.tocsr()``.
    ``csr_matrix((M, N))``
        构造形状为 ``(M, N)`` 的空矩阵.
    ``csr_matrix((data, (row, col)), shape=(M, N))``
        由非零值与行列索引构造, ``a[row[k], col[k]] = data[k]``.
    ``csr_matrix((data, indices, indptr), shape=(M, N))``
        标准 CSR 表示: 第 ``i`` 行的列索引为 ``indices[indptr[i]:indptr[i+1]]``,
        对应的值为 ``data[indptr[i]:indptr[i+1]]``.

    Parameters
    ----------
    arg1 : TensorLike, SparseTensor or tuple
        构造来源, 见上.
    shape : tuple of int, optional
        矩阵形状, 只用于由元组构造的方式.
    itype : dtype, optional
        索引的整数类型, 默认 ``bm.int64``.
    dtype : dtype, optional
        空矩阵的数值类型.
    device : device, optional
        空矩阵的设备.

    Returns
    -------
    CSRTensor
        CSR 格式稀疏矩阵.

    Raises
    ------
    ValueError
        参数组合不合法.
    """
    if itype is None:
        itype = bm.int64

    if isinstance(arg1, _DT): # 由稠密张量构造
        indices_tuple = bm.nonzero(arg1)
        indices = bm.stack(indices_tuple, axis=0)
        if itype is not None:
            indices = bm.astype(indices, itype)
        values = bm.copy(arg1[indices_tuple])
        return COOTensor(indices, values, arg1.shape).tocsr()

    elif isinstance(arg1, SparseTensor): # 由另一个稀疏张量构造
        return arg1.tocsr()

    elif isinstance(arg1, (tuple, list)):
        if isinstance(arg1[0], int): # 构造空稀疏张量
            assert len(arg1) == 2
            if itype is None:
                itype = bm.int64
            indptr = bm.zeros((arg1[0]+1,), dtype=itype, device=device)
            indices = bm.empty((0,), dtype=itype, device=device)
            data = bm.empty((0,), dtype=dtype, device=device)
            return CSRTensor(indptr, indices, data, spshape=arg1)

        elif isinstance(arg1[0], _DT) or arg1[0] is None:
            if len(arg1) == 2: # 由类 COO 格式构造
                values = arg1[0]
                indices = bm.stack(arg1[1], axis=0)
                return COOTensor(indices, values, shape).tocsr()
            elif len(arg1) == 3: # 直接由 CSR 数据构造
                data, indices, indptr = tuple(arg1)
                return CSRTensor(indptr, indices, data, shape)

    raise ValueError(f"Error: Illegal combination of parameters")


# NOTE: 稀疏张量的接口一览

# 1. 数据获取:
# itype, dtype, nnz,
# shape, dense_shape, sparse_shape,
# ndim, dense_ndim, sparse_ndim,
# size,
# values_context,
# (COO) indices, values,
# (CSR) crow, col, values

# 2. 数据类型与设备管理:
# astype, device_put

# 3. 格式转换:
# to_dense (=toarray), tocsr, tocoo

# 4. 对象转换:
# to_scipy, from_scipy,

# 5. 变形与操作:
# copy, coalesce, reshape, flatten, ravel,
# tril, triu,
# concat

# 6. 算术运算:
# add, sub, mul, div, pow, neg, matmul
