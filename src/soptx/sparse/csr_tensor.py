# 移植自 brighthe/fealpy ``fealpy/sparse/csr_tensor.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""CSR (压缩稀疏行) 格式稀疏矩阵."""

from typing import Optional, Union, overload, List,Tuple
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .sparse_tensor import SparseTensor
from .utils import (
    flatten_indices,
    check_shape_match, check_spshape_match
)
from ._spspmm import spspmm_csr
from ._spmm import spmm_csr
from .coo_tensor import COOTensor

class CSRTensor(SparseTensor):
    """CSR (压缩稀疏行) 格式稀疏矩阵, 稀疏维固定为两维.

    Parameters
    ----------
    crow : TensorLike
        压缩行指针, 形状 ``(nrow + 1, )``, 第 ``i`` 行的非零元位于
        ``[crow[i], crow[i+1])``.
    col : TensorLike
        非零元的列索引, 形状 ``(nnz, )``.
    values : TensorLike or None
        非零元的值, 形状 ``(..., nnz)``, 前导轴为稠密维; None 表示值均为 1 的
        模式矩阵.
    spshape : tuple of int, optional
        稀疏维形状 ``(nrow, ncol)``, 默认由 ``crow`` 长度与最大列索引推断.

    Raises
    ------
    ValueError
        ``crow`` 或 ``col`` 不是一维, ``spshape`` 不是二元组或与 ``crow`` 不符,
        ``values`` 末轴长度与 ``nnz`` 不符, 或 ``values`` 既不是张量也不是 None.
    """
    def __init__(self, crow: TensorLike, col: TensorLike, values: Optional[TensorLike],
                 spshape: Optional[Size]=None) -> None:
        self._crow = crow
        self._col = col
        self._values = values

        if spshape is None:
            nrow = crow.shape[0] - 1
            ncol = bm.max(col) + 1
            self._spshape = (nrow, ncol)
        else:
            self._spshape = tuple(spshape)

        self._check(crow, col, values, self._spshape)

    def _check(self, crow: TensorLike, col: TensorLike, values: Optional[TensorLike], spshape: Size):
        if crow.ndim != 1:
            raise ValueError(f"crow must be a 1-D tensor, but got {crow.ndim}")
        if col.ndim != 1:
            raise ValueError(f"col must be a 1-D tensor, but got {col.ndim}")
        if len(spshape) != 2:
                raise ValueError(f"spshape must be a 2-tuple for CSR format, but got {spshape}")

        if spshape[0] != crow.shape[0] - 1:
            raise ValueError(f"crow.shape[0] - 1 must be equal to spshape[0], "
                             f"but got {crow.shape[0] - 1} and {spshape[0]}")

        if isinstance(values, TensorLike):
            if values.ndim < 1:
                raise ValueError(f"values must be at least 1-D, but got {values.ndim}")

            if values.shape[-1] != col.shape[-1]:
                raise ValueError(f"values must have the same size as col ({col.shape[-1]}) "
                                 "in the last dimension (number of non-zero elements), "
                                 f"but got {values.shape[-1]}")
        elif values is None:
            pass
        else:
            raise ValueError(f"values must be a Tensor or None, but got {type(values)}")

    def __repr__(self) -> str:
        return f"CSRTensor(crow={self._crow}, col={self._col}, "\
               + f"values={self._values}, shape={self.shape})"

    ### 1. 数据获取 ###
    @property
    def device(self):
        """列索引张量所在的设备 (cpu / cuda:0 等)."""
        return self._col.device

    @property
    def itype(self):
        """索引的整数类型."""
        return self._col.dtype

    @property
    def nnz(self):
        """非零元个数 (含重复索引)."""
        return self._col.shape[-1]

    @property
    def crow(self) -> TensorLike:
        """压缩行指针."""
        return self._crow

    @property
    def col(self):
        """非零元的列索引."""
        return self._col

    @property
    def values(self) -> Optional[TensorLike]:
        """非零元的值, 形状 ``(..., nnz)``; 模式矩阵为 None."""
        return self._values

    @property
    def row(self):
        """由行指针展开得到的各非零元行索引."""
        count = self._crow[1:] - self._crow[:-1]
        nrow = self._crow.shape[0] - 1
        kargs = bm.context(self._crow)
        return bm.repeat(bm.arange(nrow, **kargs), count)

    @property
    def indptr(self):
        """压缩行指针, 沿用 scipy 的命名."""
        return self._crow

    @property
    def indices(self):
        """列索引, 沿用 scipy 的命名."""
        return self._col

    @property
    def data(self):
        """非零元的值, 沿用 scipy 的命名."""
        return self._values

    @property
    def nonzero_slice(self) -> Tuple[Union[slice, TensorLike]]:
        """在同形状稠密矩阵上取出各非零位置的 ``(row, col)`` 下标元组."""
        return self.row, self._col

    ### 2. 数据类型与设备管理 ###
    def astype(self, dtype=None, /, *, copy=True):
        """转换 ``values`` 的数据类型; 模式矩阵转换为值全为 1 的矩阵."""
        if self._values is None:
            values = bm.ones(self.nnz, dtype=dtype)
        else:
            values = bm.astype(self._values, dtype, copy=copy)

        return CSRTensor(self._crow, self._col, values, self._spshape)

    def device_put(self, device=None, /):
        """把行指针、列索引与值移到指定设备."""
        return CSRTensor(bm.device_put(self._crow, device),
                         bm.device_put(self._col, device),
                         bm.device_put(self._values, device),
                         self._spshape)

    ### 3. 格式转换 ###
    def to_dense(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        """转为稠密矩阵, 重复索引的值相加; 参数见 ``SparseTensor.to_dense``."""
        if self.values is None:
            dtype = bm.float64 if (dtype is None) else dtype
            context = {"dtype": dtype, "device": bm.get_device(self.indices)}
            src = bm.full((1,) * (self.dense_ndim + 1), fill_value, **context)
            src = bm.broadcast_to(src, self.dense_shape + (self.nnz,))
        else:
            src = self.values if (dtype is None) else bm.astype(self.values, dtype)
            context = {"dtype": src.dtype, "device": bm.get_device(src)}

        index_context = {'dtype': self._crow.dtype, 'device': bm.get_device(self._crow)}

        count = self._crow[1:] - self._crow[:-1]
        nrow = self._crow.shape[0] - 1
        row = bm.repeat(bm.arange(nrow, **index_context), count)
        indices = bm.stack([row, self._col], axis=0)

        dense_tensor = bm.zeros(self.dense_shape + (prod(self._spshape),), **context)
        flattened = flatten_indices(indices, self._spshape)[0]
        dense_tensor = bm.index_add(dense_tensor, flattened, src, axis=-1)

        return dense_tensor.reshape(self.shape)

    def tocoo(self, *, copy=False):
        """转为 COO 格式; ``copy=True`` 时复制 ``values``."""
        from .coo_tensor import COOTensor
        indices = bm.stack(self.nonzero_slice, axis=0)
        new_values = bm.copy(self._values) if copy else self._values
        return COOTensor(indices, new_values, self.sparse_shape)

    def tocsr(self, *, copy=False):
        """返回自身, ``copy=True`` 时返回副本."""
        if copy:
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                             bm.copy(self._values), self._spshape)
        return self

    ### 4. 对象转换 ###
    def to_petsc(self):
        """转为 PETSc AIJ 矩阵 (需要 petsc4py).

        Raises
        ------
        ValueError
            带有稠密维 (批量).
        """
        from petsc4py import PETSc

        if self.dense_ndim != 0:
            raise ValueError("Only CSRTensor with 0 dense dimension "
                             "can be converted to PETSc sparse matrix")

        return PETSc.Mat().createAIJ(
                size=self._spshape, csr=(self._crow, self._col, self._values))

    def to_scipy(self):
        """转为 ``scipy.sparse.csr_matrix``.

        Raises
        ------
        ValueError
            带有稠密维 (批量).
        """
        from scipy.sparse import csr_matrix

        if self.dense_ndim != 0:
            raise ValueError("Only CSRTensor with 0 dense dimension "
                             "can be converted to scipy sparse matrix")

        return csr_matrix(
            (bm.to_numpy(self._values), bm.to_numpy(self._col), bm.to_numpy(self._crow)),
            shape = self._spshape
        )

    @classmethod
    def from_scipy(cls, mat, /):
        """由 ``scipy.sparse.csr_matrix`` 构造."""
        crow = bm.from_numpy(mat.indptr)
        col = bm.from_numpy(mat.indices)
        values = bm.from_numpy(mat.data)
        return cls(crow, col, values, mat.shape)

    ### 5. 变形与操作 ###
    def copy(self):
        """深拷贝行指针、列索引与值."""
        if self._values is None:
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                             None, self._spshape)
        return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                         bm.copy(self._values), self._spshape)

    def coalesce(self, accumulate: bool=True) -> 'CSRTensor':
        """合并重复的 ``(row, col)`` 项 (值相加), 返回行内列有序的规范 CSR.

        Parameters
        ----------
        accumulate : bool, optional
            模式矩阵时是否把重复次数作为新的值. 默认 True.

        Returns
        -------
        CSRTensor
            规范 CSR 矩阵.

        Notes
        -----
        numpy 后端且 ``values`` 为一维时交给 scipy 的 ``sum_duplicates``, 其余情况
        走与后端无关的排序合并. 后者由 Kyle 提供, Edwin 提议并入 FEALPy 的
        ``CSRTensor``, 使稀疏矩阵相加返回规范 CSR; 该路径出错时应直接报告, 而不是
        在求解器一侧另做稀疏格式的变通.
        """
        nrow, ncol = self.sparse_shape
        if self.nnz == 0:
            values = self._values
            if values is not None:
                values = bm.zeros(values.shape[:-1] + (0,), **bm.context(values))
            return CSRTensor(
                bm.zeros((nrow + 1,), **bm.context(self._crow)),
                bm.zeros((0,), **bm.context(self._col)),
                values,
                self._spshape,
            )

        backend = bm.get_current_backend("CSRTensor.coalesce")
        if (
            getattr(backend, "backend_name", None) == "numpy"
            and self._values is not None
            and self._values.ndim == 1
        ):
            mat = self.to_scipy()
            mat.sum_duplicates()
            return CSRTensor.from_scipy(mat)

        count = self._crow[1:] - self._crow[:-1]
        row = bm.repeat(
            bm.arange(nrow, dtype=self._crow.dtype, device=bm.get_device(self._crow)),
            count,
        )
        flat = bm.astype(row, bm.int64) * ncol + bm.astype(self._col, bm.int64)
        order = bm.argsort(flat)
        flat = flat[order]

        group_start = bm.ones(
            (self.nnz,), dtype=bm.bool, device=bm.get_device(self._col)
        )
        group_start = bm.set_at(group_start, slice(1, None), flat[1:] != flat[:-1])
        unique_flat = flat[group_start]
        group_id = bm.cumsum(group_start, axis=0) - 1

        new_row = unique_flat // ncol
        new_col = unique_flat % ncol
        counts_per_row = bm.bincount(new_row, minlength=nrow)
        counts_per_row = bm.astype(counts_per_row, self._crow.dtype)
        new_crow = bm.concat(
            [
                bm.zeros((1,), **bm.context(self._crow)),
                bm.cumsum(counts_per_row, axis=0),
            ],
            axis=0,
        )
        new_col = bm.astype(new_col, self._col.dtype)

        if self._values is None:
            if accumulate:
                new_values = bm.bincount(group_id, minlength=unique_flat.shape[0])
            else:
                new_values = None
        else:
            sorted_values = self._values[..., order]
            new_values = bm.zeros(
                self._values.shape[:-1] + (unique_flat.shape[0],),
                **bm.context(self._values),
            )
            new_values = bm.index_add(new_values, group_id, sorted_values, axis=-1)

        return CSRTensor(new_crow, new_col, new_values, self._spshape)

    @overload
    def reshape(self, shape: Size, /) -> 'CSRTensor': ...
    @overload
    def reshape(self, *shape: int) -> 'CSRTensor': ...
    def reshape(self, *shape) -> 'CSRTensor':
        """改变形状. 尚未实现: 函数体为空, 返回 None."""
        pass

    def ravel(self) -> 'CSRTensor':
        """展平稀疏维. 尚未实现: 函数体为空, 返回 None."""
        pass

    def flatten(self) -> 'CSRTensor':
        """展平稀疏维并复制. 尚未实现: 函数体为空, 返回 None."""
        pass

    @property
    def T(self):
        """转置, 经 COO 格式中转."""
        A = self.tocoo()
        return A.T.tocsr()

    def partial(self, index: Union[TensorLike, slice]):
        """按非零元的编号、掩码或切片取出部分非零元, 形状不变, 行指针随之重算."""
        crow = self.crow
        ZERO = bm.zeros([1], dtype=crow.dtype, device=bm.get_device(crow))
        new_col = bm.copy(self.col[..., index])
        is_selected = bm.zeros((self.nnz,), dtype=bm.bool, device=bm.get_device(new_col))
        is_selected = bm.set_at(is_selected, index, True)
        selected_cum = bm.concat([ZERO, bm.cumsum(is_selected, axis=0)], axis=0)
        new_nnz_per_row = selected_cum[crow[1:]] - selected_cum[crow[:-1]]
        new_crow = bm.concat([ZERO, bm.cumsum(new_nnz_per_row, axis=0)], axis=0)

        new_values = self.values

        if new_values is not None:
            new_values = bm.copy(new_values[..., index])

        return CSRTensor(new_crow, new_col, new_values, self._spshape)

    def tril(self, k: int = 0) -> 'CSRTensor':
        """取第 ``k`` 条对角线及以下的非零元."""
        tril_loc = (self.row + k) >= self.col
        return self.partial(tril_loc)

    def triu(self, k: int = 0) -> 'CSRTensor':
        """取第 ``k`` 条对角线及以上的非零元."""
        tril_loc = (self.col - k) >= self.row
        return self.partial(tril_loc)

    def sum(self, axis=0):
        """按行或按列求和.

        Parameters
        ----------
        axis : {0, 1}, optional
            0 (默认) 返回各行之和 ``(nrow, )``, 1 返回各列之和 ``(ncol, )``.
            注意与 numpy 的约定相反.

        Returns
        -------
        TensorLike or None
            求和结果; ``axis`` 取其他值时返回 None.
        """
        kargs = bm.context(self._values)
        if axis == 0: # 各行之和
            return self@bm.ones(self._spshape[1], **kargs)
        elif axis == 1: # 各列之和
            r = bm.zeros(self._spshape[1], **kargs)
            r = bm.index_add(r, self._col, self._values)
            return r

    ### 6. 算术运算 ###
    def neg(self) -> 'CSRTensor':
        """取负; 模式矩阵返回自身."""
        if self._values is None:
            return self
        else:
            return CSRTensor(self._crow, self._col, -self._values, self._spshape)

    @overload
    def add(self, other: Union[Number, 'CSRTensor'], alpha: Number=1) -> 'CSRTensor': ...
    @overload
    def add(self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other: Union[Number, 'CSRTensor', TensorLike], alpha: Number=1) -> Union['CSRTensor', TensorLike]:
        """计算 ``self + alpha * other``.

        与 CSR 矩阵相加时逐行合并两者的非零元并 ``coalesce`` 为规范 CSR, 一方为
        模式矩阵时其值按 1 计; 与稠密张量相加时返回稠密张量; 与数相加时只加到已有
        非零元上.

        Parameters
        ----------
        other : int, float, CSRTensor or TensorLike
            加数.
        alpha : int or float, optional
            加数的系数, 默认 1.

        Returns
        -------
        CSRTensor or TensorLike
            ``other`` 为稠密张量时返回稠密张量, 否则返回 CSR 矩阵.

        Raises
        ------
        TypeError
            ``other`` 的类型不受支持.
        ValueError
            形状不匹配.

        Notes
        -----
        模式矩阵 (``values`` 为 None) 与稠密张量相加的分支有误
        (``dense_ndim + (nnz,)`` 为 int 与 tuple 相加), 会抛 ``TypeError``.
        """
        self_indices = bm.stack(self.nonzero_slice, axis=0)
        if isinstance(other, CSRTensor):
            check_shape_match(self.shape, other.shape)
            check_spshape_match(self.sparse_shape, other.sparse_shape)

            nrow = self.sparse_shape[0]
            self_count = self._crow[1:] - self._crow[:-1]
            other_count = other._crow[1:] - other._crow[:-1]
            new_count = self_count + other_count
            new_crow = bm.concat(
                [
                    bm.zeros((1,), **bm.context(self._crow)),
                    bm.cumsum(new_count, axis=0),
                ],
                axis=0,
            )
            total_nnz = self.nnz + other.nnz

            row_context = bm.context(self._crow)
            self_row = bm.repeat(bm.arange(nrow, **row_context), self_count)
            other_row = bm.repeat(bm.arange(nrow, **row_context), other_count)
            self_local = bm.arange(self.nnz, **row_context) - bm.repeat(
                self._crow[:-1], self_count
            )
            other_local = bm.arange(other.nnz, **row_context) - bm.repeat(
                other._crow[:-1], other_count
            )
            self_pos = new_crow[self_row] + self_local
            other_pos = new_crow[other_row] + self_count[other_row] + other_local

            new_col = bm.zeros((total_nnz,), **bm.context(self._col))
            new_col = bm.index_add(new_col, self_pos, self._col, axis=0)
            new_col = bm.index_add(new_col, other_pos, other._col, axis=0)

            if self._values is None and other._values is None:
                unit_context = {"device": bm.get_device(self._col)}
                self_values = bm.ones((self.nnz,), **unit_context)
                other_values = bm.ones((other.nnz,), **unit_context)
            elif self._values is None:
                shape = other._values.shape[:-1] + (self.nnz,)
                self_values = bm.ones(shape, device=bm.get_device(other._values))
                other_values = other._values
            elif other._values is None:
                shape = self._values.shape[:-1] + (other.nnz,)
                self_values = self._values
                other_values = bm.ones(shape, device=bm.get_device(self._values))
            else:
                self_values = self._values
                other_values = other._values

            other_values = other_values * alpha
            probe = bm.sum(self_values) + bm.sum(other_values) * 0
            value_context = bm.context(probe)
            self_values = self_values + bm.zeros(self_values.shape, **value_context)
            other_values = other_values + bm.zeros(other_values.shape, **value_context)
            new_values = bm.zeros(
                self_values.shape[:-1] + (total_nnz,),
                **value_context,
            )
            new_values = bm.index_add(new_values, self_pos, self_values, axis=-1)
            new_values = bm.index_add(
                new_values,
                other_pos,
                other_values,
                axis=-1,
            )
            # Kyle/Edwin 的修改: CSR 相加须返回规范 CSR. 这依赖 Kyle 的重复项合并
            # 路径, 使矩阵相加之后的对角线提取、基于 nnz 的检查与迭代求解器的输入
            # 保持一致.
            return CSRTensor(new_crow, new_col, new_values, self.sparse_shape).coalesce()

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            output = other * alpha
            context = bm.context(output)
            output = output.reshape(self.dense_shape + (prod(self._spshape),))
            flattened = flatten_indices(self_indices, self._spshape)[0]

            if self._values is None:
                src = bm.ones((1,) * (self.dense_ndim + 1), **context)
                src = bm.broadcast_to(src, self.dense_ndim + (self.nnz,))
            else:
                src = self._values
            output = bm.index_add(output, flattened, src, axis=-1)

            return output.reshape(self.shape)

        elif isinstance(other, (int, float)):
            new_values = self._values + alpha * other
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col), new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in addition")

    def mul(self, other: Union[Number, 'CSRTensor', TensorLike]) -> 'CSRTensor':
        """逐元素乘法; 与数或稠密张量相乘时结果与原矩阵共享索引.

        Raises
        ------
        ValueError
            模式矩阵乘以数.
        TypeError
            ``other`` 的类型不受支持.

        Notes
        -----
        与 CSR 矩阵相乘的分支尚未实现, 函数体为空, 会静默返回 None.
        """
        if isinstance(other, CSRTensor):
            pass

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])

            if self._values is not None:
                bm.multiply(self._values, new_values, out=new_values)

            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            if self._values is None:
                raise ValueError("Cannot multiply CSRTensor without value with scalar")
            new_values = self._values * other

            return CSRTensor(self._crow,self._col, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in multiplication")

    def div(self, other: Union[Number, TensorLike]) -> 'CSRTensor':
        """逐元素除法; 结果与原矩阵共享索引.

        一维张量的长度等于行数时按行广播, 等于列数时按列广播 (方阵时按行).

        Raises
        ------
        ValueError
            模式矩阵不能做除法, 或形状不匹配.
        TypeError
            ``other`` 的类型不受支持.
        """
        if self._values is None:
                raise ValueError("Cannot divide CSRTensor without value")

        if isinstance(other, TensorLike):
            if len(other.shape) == 1: #TODO: 处理方阵 (行数等于列数) 时的歧义
                if other.shape[0] == self.shape[0]:
                    other = bm.broadcast_to(other[:, None], self.shape)
                elif other.shape[0] == self.shape[1]:
                    other = bm.broadcast_to(other[None, :], self.shape)
            check_shape_match(other.shape, self.shape)
            new_values = bm.copy(other[self.nonzero_slice])
  
            bm.divide(self._values, new_values, out=new_values)
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values / other
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in division")

    def pow(self, other: Union[TensorLike, Number]) -> 'CSRTensor':
        """逐元素乘方; 结果与原矩阵共享索引.

        Raises
        ------
        ValueError
            模式矩阵不能做乘方.
        TypeError
            ``other`` 的类型不受支持.
        """
        if self._values is None:
            raise ValueError("Cannot power CSRTensor without value with tensor")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])

            new_values = bm.power(self._values, new_values)
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values ** other
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        else:
            raise TypeError(f'Unsupported type {type(other).__name__} in power')

    @overload
    def matmul(self, other: 'CSRTensor') -> 'CSRTensor': ...
    @overload
    def matmul(self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other: Union['CSRTensor', TensorLike]):
        """矩阵乘法.

        Parameters
        ----------
        other : CSRTensor or TensorLike
            CSR 矩阵, 或稠密张量: 一维为矩阵--向量乘, 二维为矩阵--矩阵乘; 也支持
            ``(*B, M, K)`` 与 ``(*B, K, N)`` 的批量矩阵乘.

        Returns
        -------
        CSRTensor or TensorLike
            与 CSR 矩阵相乘时返回 CSR 矩阵, 与稠密张量相乘时返回稠密张量.

        Raises
        ------
        ValueError
            任一方为模式矩阵.
        TypeError
            ``other`` 的类型不受支持.
        """
        if isinstance(other, CSRTensor):
            if (self.values is None) or (other.values is None):
                raise ValueError("Matrix multiplication between CSRTensor without "
                                 "value is not implemented now")
            if hasattr(bm, 'csr_spspmm'):
                crow, col, values, spshape = bm.csr_spspmm(
                    self.crow, self.col, self.values, self.sparse_shape,
                    other.crow, other.col, other.values, other.sparse_shape
                )
            else:
                crow, col,values, spshape = spspmm_csr(
                    self._crow,self._col ,self._values, self.sparse_shape,
                    other._crow, other._col,other._values, other.sparse_shape,
                )
            return CSRTensor(crow, col, values, spshape)

        elif isinstance(other, TensorLike):
            if self.values is None:
                raise ValueError()
            if hasattr(bm, 'csr_spmm'):
                return bm.csr_spmm(self._crow, self._col, self._values, self._spshape, other)
            else:
                return spmm_csr(self._crow, self._col,self._values,self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")


    def find(self):
        """找出存储值不为 0 的项.

        Returns
        -------
        tuple
            ``(行索引, 列索引, 值)``.
        """
        nz_mask = self.values != 0
        return self.row[nz_mask], self.col[nz_mask], self.values[nz_mask]

    def diags(self) -> 'CSRTensor':
        """取出对角线上的非零元.

        Returns
        -------
        CSRTensor
            只含对角元的同形状 CSR 矩阵 (不是对角线向量).
        """
        diags_loc = (self.row) == self.col
        return self.partial(diags_loc)

    def col_min(self):
        """各列非零元的最小值.

        Returns
        -------
        TensorLike
            形状 ``(ncol, )``.

        Notes
        -----
        结果以 0 为初值, 全为正值的列返回 0; 依赖 numpy 的 ``minimum.at``,
        其他后端不可用.
        """
        M = bm.zeros(self._spshape[1], dtype=self._values.dtype)
        bm.minimum.at(M, self._col, self._values)
        
        return M

    def __getitem__(self, index):
        if isinstance(index, Tuple):
            crow_index, col_index = index 
        else:
            crow_index = index
            col_index = None

        if col_index is not None:
            if isinstance(col_index, slice):
                start = col_index.start if col_index.start is not None else 0
                stop = col_index.stop if col_index.stop is not None else self._spshape[1]
                step = col_index.step if col_index.step is not None else 1
                new_shape = (stop - start + step - 1) // step
                col_index = bm.arange(start, stop, step)
                if new_shape == self._spshape[1]:
                    new_crow = self._crow
                    new_col = self._col
                    new_values = self._values
                    new_col_shape = self._spshape[1] 
            elif isinstance(col_index, (List, TensorLike)):
                new_shape = len(col_index)
            elif isinstance(col_index, int):
                new_shape = 1
            else:
                raise TypeError(f'index must be a slice or int, but got {type(index)}')

            kwargs = bm.context(self._col)
            nrz = self.crow[1:] - self.crow[:-1]
            isfindnode = bm.zeros((self._spshape[1],), **kwargs)
            isfindnode = bm.add_at(isfindnode, col_index, 1) == 1
            row = bm.repeat(bm.arange(self._spshape[0]), nrz)
            new_row = row[isfindnode[self._col]]
            new_row = bm.concat((new_row, bm.full((1,), self._spshape[0] - 1, **kwargs)))
            new_crow = bm.concat((bm.zeros((1,), **kwargs), bm.cumsum(bm.bincount(new_row))))
            new_crow[-1] = new_crow[-1] - 1
            new_values = self._values[isfindnode[self._col]]
            if isinstance(col_index, int):
                new_col = bm.zeros((self._col[isfindnode[self._col]].shape[0],), dtype=bm.int64)
            else:
                a = bm.searchsorted(col_index, self._col[isfindnode[self._col]]) 
                new_col = bm.arange(new_shape)[a]
            new_col_shape = new_shape
        else:
            new_crow = self._crow
            new_col = self._col
            new_values = self._values
            new_col_shape = self._spshape[1]

        if isinstance(crow_index, slice):
            start = crow_index.start if crow_index.start is not None else 0
            stop = crow_index.stop if crow_index.stop is not None else new_crow.shape[0] - 1
            step = crow_index.step if crow_index.step is not None else 1
            new_row_shape = (stop - start + step - 1) // step
            if new_row_shape == new_crow.shape[0] - 1:
                return CSRTensor(new_crow, new_col, new_values, spshape=(new_row_shape, new_col_shape)) 
        elif isinstance(crow_index, (List, TensorLike)):
            new_row_shape = len(crow_index)
        elif isinstance(crow_index, int):
            new_row_shape = 1
        else:
            raise TypeError(f'index must be a slice or int, but got {type(index)}')

        kwargs = bm.context(self.crow)
        nrz = new_crow[1:] - new_crow[:-  1]
        isfindnode = bm.zeros((new_crow.shape[0] - 1,), **kwargs)
        isfindnode = bm.add_at(isfindnode, crow_index, 1)
        findnode = bm.repeat(isfindnode == 1, nrz) 
        new_col = new_col[findnode]
        new_values = new_values[findnode]
        new_crow = bm.concat((bm.zeros((1,), **kwargs), bm.cumsum(nrz[isfindnode == 1])))
        
        return CSRTensor(new_crow, new_col, new_values, spshape=(new_row_shape, new_col_shape))
    
    def sum_duplicates(self):
        """``coalesce`` 的别名, 沿用 scipy 的命名."""
        # Kyle/Edwin 的修改: 保留 scipy 风格的 sum_duplicates, 作为规范 CSR 合并
        # 实现的别名. coalesce 现用的重复项合并算法由 Kyle 提供, Edwin 为大规模
        # 稀疏 FVM/FEM 系统提议并入.
        return self.coalesce()
