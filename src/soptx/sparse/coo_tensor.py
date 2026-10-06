# 移植自 brighthe/fealpy ``fealpy/sparse/coo_tensor.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""COO (坐标) 格式稀疏张量."""

from typing import Optional, Union, overload, Tuple, Sequence
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .sparse_tensor import SparseTensor
from .utils import (
    flatten_indices, check_shape_match, check_spshape_match
)
from ._spspmm import spspmm_coo
from ._spmm import spmm_coo


class COOTensor(SparseTensor):
    """COO (坐标) 格式稀疏张量.

    Parameters
    ----------
    indices : TensorLike
        非零元索引, 形状 ``(D, nnz)``, ``D`` 为稀疏维数.
    values : TensorLike or None
        非零元的值, 形状 ``(..., nnz)``, 前导轴为稠密维; None 表示值均为 1 的
        模式张量.
    spshape : tuple of int, optional
        稀疏维形状, 默认取各维最大索引加 1.
    is_coalesced : bool, optional
        索引是否已合并 (无重复且有序); True 时 ``coalesce`` 直接返回自身.

    Raises
    ------
    TypeError
        ``indices`` 不是张量, 或 ``values`` 既不是张量也不是 None.
    ValueError
        ``indices`` 不是二维, ``values`` 末轴长度与 ``nnz`` 不符, 或 ``spshape``
        长度与稀疏维数不符.
    """
    def __init__(self, indices: TensorLike, values: Optional[TensorLike],
                 spshape: Optional[Size] = None, *,
                 is_coalesced: Optional[bool] = None):
        self._indices = indices
        self._values = values
        self.is_coalesced = is_coalesced
        self._check(indices, values)

        if spshape is None:
            self._spshape = tuple(bm.tolist(bm.max(indices, axis=1) + 1))
        else:
            # 总维数应等于稀疏维数加稠密维数
            if len(spshape) != indices.shape[0]:
                raise ValueError(
                    f"length of sparse shape ({len(spshape)}) "
                    f"must match the size of indices in dim-0 ({indices.shape[0]})"
                )
            self._spshape = tuple(spshape)

    def _check(self, indices: TensorLike, values: Optional[TensorLike]):
        if not isinstance(indices, TensorLike):
            raise TypeError(f"indices must be a Tensor, but got {type(indices)}")
        if indices.ndim != 2:
            raise ValueError(f"indices must be a 2D tensor, but got {indices.ndim}D")

        if isinstance(values, TensorLike):
            if values.ndim < 1:
                raise ValueError(f"values must be at least 1D, but got {values.ndim}D")

            # values 的末轴须与 indices 的末轴 (非零元个数) 一致.
            if values.shape[-1] != indices.shape[1]:
                raise ValueError(f"values must have the same size as indices ({indices.shape[1]}) "
                                 "in the last dimension (number of non-zero elements), "
                                 f"but got {values.shape[-1]}")
        elif values is None:
            pass
        else:
            raise TypeError(f"values must be a Tensor or None, but got {type(values)}")

    def __repr__(self) -> str:
        return f"COOTensor(indices={self._indices}, values={self._values}, shape={self.shape})"

    ### 1. 数据获取 ###
    @property
    def device(self):
        """索引张量所在的设备 (cpu / cuda:0 等)."""
        return self._indices.device

    @property
    def itype(self):
        """索引的整数类型."""
        return self._indices.dtype

    @property
    def nnz(self):
        """非零元个数 (含重复索引)."""
        return self._indices.shape[1]

    @property
    def indices(self) -> TensorLike:
        """非零元索引, 形状 ``(D, nnz)``."""
        return self._indices

    @property
    def values(self) -> Optional[TensorLike]:
        """非零元的值, 形状 ``(..., nnz)``; 模式张量为 None."""
        return self._values

    @property
    def row(self):
        """行索引, 沿用 scipy 的命名."""
        return self._indices[0]

    @property
    def col(self):
        """列索引, 沿用 scipy 的命名."""
        return self._indices[1]

    @property
    def data(self):
        """非零元的值, 沿用 scipy 的命名."""
        return self._values

    @property
    def nonzero_slice(self) -> Tuple[Union[slice, TensorLike]]:
        """在同形状稠密张量上取出各非零位置的下标元组, 稠密维取全切片."""
        slicing = [self._indices[i] for i in range(self.sparse_ndim)]
        return (slice(None),) * self.dense_ndim + tuple(slicing)

    ### 2. 数据类型与设备管理 ###
    def astype(self, dtype=None, /, *, copy=True):
        """转换 ``values`` 的数据类型; 模式张量转换为值全为 1 的张量."""
        if self._values is None:
            values = bm.ones(self.nnz, dtype=dtype)
        else:
            values = bm.astype(self._values, dtype, copy=copy)

        return COOTensor(self._indices, values, self._spshape,
                         is_coalesced=self.is_coalesced)

    def device_put(self, device=None, /):
        """把索引与值移到指定设备."""
        return COOTensor(bm.device_put(self._indices, device),
                         bm.device_put(self._values, device),
                         self._spshape,
                         is_coalesced=self.is_coalesced)

    ### 3. 格式转换 ###
    def to_dense(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        """转为稠密张量, 重复索引的值相加; 参数见 ``SparseTensor.to_dense``."""
        if self.values is None:
            dtype = bm.float64 if (dtype is None) else dtype
            context = {"dtype": dtype, "device": bm.get_device(self.indices)}
            src = bm.full((1,) * (self.dense_ndim + 1), fill_value, **context)
            src = bm.broadcast_to(src, self.dense_shape + (self.nnz,))
        else:
            src = self.values if (dtype is None) else bm.astype(self.values, dtype)
            context = {"dtype": src.dtype, "device": bm.get_device(src)}

        dense_tensor = bm.zeros(self.dense_shape + (prod(self._spshape),), **context)
        flattened = flatten_indices(self._indices, self._spshape)[0]
        dense_tensor = bm.index_add(dense_tensor, flattened, src, axis=-1)

        return dense_tensor.reshape(self.shape)

    def tocoo(self, *, copy=False):
        """返回自身, ``copy=True`` 时返回副本."""
        if copy:
            return self.copy()
        return self

    def tocsr(self, *, copy=False):
        """转为 CSR 格式 (只支持两个稀疏维).

        按行重排非零元, 不合并重复索引, 行内列的顺序不保证有序; 需要时先调用
        ``coalesce``.

        Parameters
        ----------
        copy : bool, optional
            是否复制 ``values``. 默认 False.

        Returns
        -------
        CSRTensor
            CSR 格式稀疏张量.
        """
        from .csr_tensor import CSRTensor
        # try:
        #     crow, col, values = bm.coo_tocsr(self.indices, self.values, self.sparse_shape)
        #     return CSRTensor(crow, col, values, spshape=self._spshape)
        # except (AttributeError, NotImplementedError):
        #     pass

        count = bm.bincount(self._indices[0], minlength=self._spshape[0])
        crow = bm.cumsum(count, axis=0)
        crow = bm.concat([bm.tensor([0], **bm.context(crow)), crow])
        order = bm.argsort(self._indices[0])
        new_col = bm.copy(self._indices[-1, order])

        if self.values is None:
            new_values = None
        else:
            new_values = self.values[..., order]
            new_values = bm.copy(new_values) if copy else new_values

        return CSRTensor(crow, new_col, new_values, spshape=self._spshape)

    ### 4. 对象转换 ###
    def to_scipy(self):
        """转为 ``scipy.sparse.coo_matrix``.

        Raises
        ------
        ValueError
            带有稠密维 (批量).
        """
        from scipy.sparse import coo_matrix
        if self.dense_ndim != 0:
            raise ValueError("Only COOTensor with 0 dense dimension "
                             "can be converted to scipy sparse matrix")


        return coo_matrix(
            (bm.to_numpy(self.values), bm.to_numpy(self.indices)),
            shape = self.sparse_shape
        )

    @classmethod
    def from_scipy(cls, mat, /):
        """由 ``scipy.sparse.coo_matrix`` 构造."""
        indices = bm.stack([bm.from_numpy(mat.row), bm.from_numpy(mat.col)], axis=0)
        values = bm.from_numpy(mat.data)
        return cls(indices, values, mat.shape)

    ### 5. 变形与操作 ###
    def copy(self):
        """深拷贝索引与值."""
        if self._values is None:
            return COOTensor(bm.copy(self._indices), None, self._spshape)
        return COOTensor(bm.copy(self._indices), bm.copy(self._values), self._spshape)

    def coalesce(self, accumulate: bool=True) -> 'COOTensor':
        """按字典序排序索引并合并重复项, 参数见 ``SparseTensor.coalesce``."""
        if self.is_coalesced or self.nnz == 0:
            return self

        order = bm.lexsort(tuple(reversed(self._indices)))
        sorted_indices = self._indices[:, order]
        unique_mask = bm.concat([
            bm.ones((1, ), dtype=bool, device=bm.get_device(sorted_indices)),
            bm.any(sorted_indices[:, 1:] - sorted_indices[:, :-1], axis=0)
        ], axis=0)
        new_indices = bm.copy(sorted_indices[..., unique_mask])

        if self._values is not None:
            add_index = bm.cumsum(unique_mask, axis=0) - 1
            sorted_values = self._values[..., order]
            new_values = bm.zeros_like(sorted_values[..., unique_mask])
            new_values = bm.index_add(new_values, add_index, sorted_values, axis=-1)

        else:
            if accumulate:
                unique_location = bm.concat([
                    bm.nonzero(unique_mask)[0],
                    bm.tensor([len(unique_mask)], **bm.context(self._indices))
                ], axis=0)
                new_values = unique_location[1:] - unique_location[:-1]

            else:
                new_values = None

        return COOTensor(new_indices, new_values, self.sparse_shape, is_coalesced=True)

    @overload
    def reshape(self, shape: Size, /) -> 'COOTensor': ...
    @overload
    def reshape(self, *shape: int) -> 'COOTensor': ...
    def reshape(self, *shape) -> 'COOTensor':
        """改变稀疏维形状. 尚未实现, 调用即抛 ``NotImplementedError``; 展平可用 ``ravel``."""
        raise NotImplementedError("COOTensor.reshape 尚未实现, 展平可用 ravel()")

    def ravel(self):
        """把稀疏维展平为一维, 共享 ``values``."""
        spshape = self.sparse_shape
        new_indices = flatten_indices(self._indices, spshape)
        return COOTensor(new_indices, self._values, (prod(spshape),))

    def flatten(self):
        """把稀疏维展平为一维, 复制 ``values``."""
        spshape = self.sparse_shape
        new_indices = flatten_indices(self._indices, spshape)
        if self._values is None:
            values = None
        else:
            values = bm.copy(self._values)
        return COOTensor(new_indices, values, (prod(spshape),))

    @property
    def T(self):
        """交换最后两个稀疏维, 共享 ``values``.

        Raises
        ------
        ValueError
            稀疏维数小于 2.
        """
        _indices = self._indices
        _spshape = self._spshape

        if self.sparse_ndim == 2:
            new_indices = bm.stack([_indices[1], _indices[0]], axis=0)
            shape = tuple(reversed(_spshape))
        elif self.sparse_ndim >= 3:
            new_indices = bm.concat([_indices[:-2], _indices[-1:], _indices[-2:-1]], axis=0)
            shape = _spshape[:-2] + (_spshape[-1], _spshape[-2])
        else:
            raise ValueError("sparse ndim must be 2 or greater to be transposed, "
                             f"but got {self.sparse_ndim}")
        return COOTensor(new_indices, self._values, shape)

    def partial(self, index: Union[TensorLike, slice], /):
        """按非零元的编号、掩码或切片取出部分非零元, 稀疏形状不变."""
        new_indices = bm.copy(self.indices[:, index])
        new_values = self.values

        if new_values is not None:
            new_values = bm.copy(new_values[..., index])

        return COOTensor(new_indices, new_values, self._spshape)

    def tril(self, k: int = 0) -> 'COOTensor':
        """取最后两个稀疏维中第 ``k`` 条对角线及以下的非零元."""
        indices = self.indices
        tril_loc = (indices[-2] + k) >= indices[-1]
        return self.partial(tril_loc)

    def triu(self, k: int = 0) -> 'COOTensor':
        """取最后两个稀疏维中第 ``k`` 条对角线及以上的非零元."""
        indices = self.indices
        triu_loc = (indices[-1] - k) >= indices[-2]
        return self.partial(triu_loc)

    @classmethod
    def concat(cls, coo_tensors: Sequence['COOTensor'], /, *, axis: int=0) -> 'COOTensor':
        """沿稀疏维 ``axis`` 拼接多个 COO 张量 (须带 ``values``).

        Raises
        ------
        ValueError
            列表为空.
        """
        if len(coo_tensors) == 0:
            raise ValueError("coo_tensors cannot be empty")

        if len(coo_tensors) == 1:
            return coo_tensors[0]

        indices_list = []
        values_list = []
        prev_len = 0

        for coo in coo_tensors:
            indices = bm.copy(coo.indices)
            indices = bm.index_add(
                indices, bm.array([axis], device=indices.device), prev_len,
                axis=0
            )
            indices_list.append(indices)
            values_list.append(bm.copy(coo.values))
            prev_len += coo.sparse_shape[axis]

        new_indices = bm.concat(indices_list, axis=1)
        del indices_list
        new_values = bm.concat(values_list, axis=-1)
        del values_list
        spshape = list(coo_tensors[-1].sparse_shape)
        spshape[axis] = prev_len
        return cls(new_indices, new_values, spshape)

    ### 6. 算术运算 ###
    def neg(self) -> 'COOTensor':
        """取负; 模式张量返回自身."""
        if self._values is None:
            return self
        else:
            return COOTensor(self._indices, -self._values, self.sparse_shape)

    @overload
    def add(self, other: Union[Number, 'COOTensor'], alpha: Number=1) -> 'COOTensor': ...
    @overload
    def add(self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other: Union[Number, 'COOTensor', TensorLike], alpha: Number=1) -> Union['COOTensor', TensorLike]:
        """计算 ``self + alpha * other``.

        与 COO 张量相加时拼接两者的非零元 (不合并重复索引); 与稠密张量相加时返回
        稠密张量; 与数相加时只加到已有非零元上.

        Parameters
        ----------
        other : int, float, COOTensor or TensorLike
            加数.
        alpha : int or float, optional
            加数的系数, 默认 1.

        Returns
        -------
        COOTensor or TensorLike
            ``other`` 为稠密张量时返回稠密张量, 否则返回 COO 张量.

        Raises
        ------
        TypeError
            ``other`` 的类型不受支持.
        ValueError
            形状不匹配, 或一方有 ``values`` 而另一方没有.

        Notes
        -----
        模式张量 (``values`` 为 None) 与稠密张量相加时, 每个非零位置按 1 计入.
        """
        if isinstance(other, COOTensor):
            check_shape_match(self.shape, other.shape)
            check_spshape_match(self.sparse_shape, other.sparse_shape)
            new_indices = bm.concat((self._indices, other._indices), axis=1)
            if self._values is None:
                if other._values is None:
                    new_values = None
                else:
                    raise ValueError("self has no value while other does")
            else:
                if other._values is None:
                    raise ValueError("self has value while other does not")
                new_values = bm.concat((self._values, other._values*alpha), axis=-1)
            return COOTensor(new_indices, new_values, self.sparse_shape)

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            output = other * alpha
            context = bm.context(output)
            output = output.reshape(self.dense_shape + (prod(self._spshape),))
            flattened = flatten_indices(self._indices, self._spshape)[0]

            if self._values is None:
                src = bm.ones((1,) * (self.dense_ndim + 1), **context)
                src = bm.broadcast_to(src, self.dense_shape + (self.nnz,))
            else:
                src = self._values
            output = bm.index_add(output, flattened, src, axis=-1)

            return output.reshape(self.shape)

        elif isinstance(other, (int, float)):
            new_values = self._values + alpha * other
            return COOTensor(bm.copy(self._indices), new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in addition")

    def mul(self, other: Union[Number, 'COOTensor', TensorLike]) -> 'COOTensor': # TODO: 补齐与 COO 张量相乘
        """逐元素乘法; 结果与原张量共享索引.

        Raises
        ------
        NotImplementedError
            ``other`` 为 COO 张量.
        ValueError
            模式张量乘以数.
        TypeError
            ``other`` 的类型不受支持.
        """
        if isinstance(other, COOTensor):
            raise NotImplementedError

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])
            if self._values is not None:
                new_values = bm.multiply(self._values, new_values)
            return COOTensor(self._indices, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            if self._values is None:
                raise ValueError("Cannot multiply COOTensor without value with scalar")
            new_values = self._values * other
            return COOTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in multiplication")

    def div(self, other: Union[Number, TensorLike]) -> 'COOTensor':
        """逐元素除法; 结果与原张量共享索引.

        Raises
        ------
        ValueError
            模式张量不能做除法.
        TypeError
            ``other`` 的类型不受支持.
        """
        if self._values is None:
            raise ValueError("Cannot divide COOTensor without value")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])
            new_values = bm.divide(self._values, new_values)
            return COOTensor(self._indices, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values / other
            return COOTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in division")

    def pow(self, other: Union[TensorLike, Number]) -> 'COOTensor':
        """逐元素乘方; 结果与原张量共享索引.

        Raises
        ------
        ValueError
            模式张量不能做乘方.
        TypeError
            ``other`` 的类型不受支持.
        """
        if self._values is None:
            raise ValueError("Cannot power COOTensor without value with tensor")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.power(self._values, other[self.nonzero_slice])
            return COOTensor(self._indices, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values ** other
            return COOTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f'Unsupported type {type(other).__name__} in power')

    @overload
    def matmul(self, other: 'COOTensor') -> 'COOTensor': ...
    @overload
    def matmul(self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other: Union['COOTensor', TensorLike]):
        """矩阵乘法.

        Parameters
        ----------
        other : COOTensor or TensorLike
            稀疏张量, 或稠密张量: 一维为矩阵--向量乘, 二维为矩阵--矩阵乘; 也支持
            ``(*B, M, K)`` 与 ``(*B, K, N)`` 的批量矩阵乘.

        Returns
        -------
        CSRTensor or TensorLike
            与稀疏张量相乘时返回 CSR 张量 (而非 COO), 与稠密张量相乘时返回稠密张量.

        Raises
        ------
        ValueError
            任一方为模式张量.
        TypeError
            ``other`` 的类型不受支持.
        """
        if isinstance(other, COOTensor):
            if (self.values is None) or (other.values is None):
                raise ValueError("Matrix multiplication between COOTensor without "
                                 "value is not implemented now")
            if hasattr(bm, 'csr_spspmm'):
                from .csr_tensor import CSRTensor
                mat1 = self.tocsr()
                mat2 = other.tocsr()
                crow, col, values, spshape = bm.csr_spspmm(
                    mat1.crow, mat1.col, mat1.values, mat1.sparse_shape,
                    mat2.crow, mat2.col, mat2.values, mat2.sparse_shape
                )
                return CSRTensor(crow, col, values, spshape)
            else:
                indices, values, spshape = spspmm_coo(
                    self.indices, self.values, self.sparse_shape,
                    other.indices, other.values, other.sparse_shape,
                )
            return COOTensor(indices, values, spshape).coalesce().tocsr()

        elif isinstance(other, TensorLike):
            if self.values is None:
                raise ValueError()
            if hasattr(bm, 'coo_spmm'):
                return bm.coo_spmm(self._indices, self._values, self._spshape, other)
            else:
                return spmm_coo(self.indices, self.values, self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")
