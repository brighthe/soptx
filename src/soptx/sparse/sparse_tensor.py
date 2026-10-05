# 移植自 brighthe/fealpy ``fealpy/sparse/sparse_tensor.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""稀疏张量基类."""

from typing import Optional, Union, overload, Dict, Sequence, Any, TypeVar, Type
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .utils import _dense_ndim, _dense_shape

_Self = TypeVar('_Self', bound='SparseTensor')


class SparseTensor():
    """稀疏张量基类, 声明各格式共有的接口并实现与格式无关的部分.

    ``values`` 的前导轴为稠密维 (批量), 末轴对应各非零元; ``values`` 为 None
    时表示只有稀疏结构、元素值均为 1 的 "模式" 张量.
    """
    _values: Optional[TensorLike]
    _spshape: Size

    ### 1. 数据获取 ###
    # NOTE: 以下属性由各子类的索引系统提供.
    @property
    def itype(self):
        """索引的整数类型."""
        raise NotImplementedError
    @property
    def nnz(self) -> int:
        """非零元个数."""
        raise NotImplementedError

    def values_context(self) -> Dict[str, Any]:
        """``values`` 的 dtype 与 device 构造参数; ``values`` 为 None 时为空字典."""
        if self._values is None:
            return {}
        return bm.context(self._values)

    @property
    def device(self):
        """所在设备."""
        raise NotImplementedError

    @property
    def ftype(self):
        """``values`` 的浮点类型, 无 ``values`` 时为 None."""
        return None if self._values is None else self._values.dtype

    @property
    def dtype(self):
        """``values`` 的数据类型, 无 ``values`` 时为 None."""
        return None if self._values is None else self._values.dtype

    @property
    def shape(self):
        """完整形状, 稠密维在前."""
        return self.dense_shape + self.sparse_shape
    @property
    def dense_shape(self):
        """稠密维 (批量) 形状."""
        return _dense_shape(self._values)
    @property
    def sparse_shape(self):
        """稀疏维形状."""
        return self._spshape

    @property
    def ndim(self):
        """总维数."""
        return self.dense_ndim + self.sparse_ndim
    @property
    def dense_ndim(self):
        """稠密维数."""
        return _dense_ndim(self._values)
    @property
    def sparse_ndim(self):
        """稀疏维数."""
        return len(self._spshape)

    def size(self, dim: Optional[int]=None) -> int:
        """视为稠密张量时的元素个数; 给出 ``dim`` 时为该轴长度."""
        if dim is None:
            return prod(self.shape)
        else:
            return self.shape[dim]

    ### 2. 数据类型与设备管理 ###
    def astype(self: _Self, dtype=None, /, *, copy=True) -> _Self:
        """转换 ``values`` 的数据类型, 由子类实现."""
        raise NotImplementedError

    def device_put(self: _Self, device=None, /) -> _Self:
        """移到指定设备, 由子类实现."""
        raise NotImplementedError

    ### 3. 格式转换 ###
    def to_dense(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        """转为稠密张量, 返回新对象, 由子类实现.

        Parameters
        ----------
        fill_value : int, float or bool, optional
            ``values`` 为 None 时非零位置填入的值, 默认 1.
        dtype : dtype, optional
            ``values`` 为 None 时的元素类型, 默认 float64.

        Returns
        -------
        TensorLike
            稠密张量.
        """
        raise NotImplementedError

    def toarray(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        """``to_dense`` 的别名."""
        return self.to_dense(fill_value=fill_value, dtype=dtype)

    def tocoo(self, *, copy=False):
        """转为 COO 格式, 由子类实现."""
        raise NotImplementedError

    def tocsr(self, *, copy=False):
        """转为 CSR 格式, 由子类实现."""
        raise NotImplementedError

    ### 4. 对象转换 ###
    def to_scipy(self):
        """转为 scipy 稀疏矩阵, 由子类实现."""
        raise NotImplementedError

    def from_scipy(cls, mat, /):
        """由 scipy 稀疏矩阵构造, 由子类实现."""
        raise NotImplementedError

    ### 5. 变形与操作 ###
    def copy(self: _Self) -> _Self:
        """深拷贝, 由子类实现."""
        raise NotImplementedError

    def coalesce(self: _Self, accumulate: bool=True) -> _Self:
        """合并重复索引 (对应的值相加), 返回新稀疏张量; 已合并时返回自身. 由子类实现.

        Parameters
        ----------
        accumulate : bool, optional
            ``values`` 为 None 时, 是否把各索引的出现次数作为新的值. 默认 True.

        Returns
        -------
        SparseTensor
            合并后的稀疏张量.
        """
        raise NotImplementedError

    def reshape(self: _Self, *shape) -> _Self:
        """改变稀疏维形状, 由子类实现."""
        raise NotImplementedError

    def ravel(self: _Self) -> _Self:
        """把稀疏维展平为一维, 尽量返回视图.

        Returns
        -------
        SparseTensor
            形状 ``(*dense_shape, -1)``.
        """
        return self.reshape(-1)

    def flatten(self: _Self) -> _Self:
        """把稀疏维展平为一维, 返回副本. 由子类实现.

        Returns
        -------
        SparseTensor
            形状 ``(*dense_shape, -1)``.
        """
        raise NotImplementedError

    @property
    def T(self: _Self) -> _Self:
        """转置 (交换最后两个稀疏维), 由子类实现."""
        raise NotImplementedError

    def tril(self, k: int=0) -> _Self:
        """取第 ``k`` 条对角线及以下的部分, 由子类实现."""
        raise NotImplementedError

    def triu(self, k: int=0) -> _Self:
        """取第 ``k`` 条对角线及以上的部分, 由子类实现."""
        raise NotImplementedError

    @classmethod
    def concat(cls: Type[_Self], tensors: Sequence[_Self], /, *, axis: int=0) -> _Self:
        """沿 ``axis`` 拼接多个同格式稀疏张量, 由子类实现."""
        raise NotImplementedError

    ### 6. 算术运算 ###
    def neg(self: _Self) -> _Self:
        """取负, 由子类实现."""
        raise NotImplementedError

    @overload
    def add(self: _Self, other: Union[Number, _Self], alpha: Number=1) -> _Self: ...
    @overload
    def add(self: _Self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other, alpha: Number=1):
        """计算 ``self + alpha * other``, 由子类实现; 与稠密张量相加时返回稠密张量."""
        raise NotImplementedError

    def mul(self: _Self, other: Union[Number, _Self, TensorLike]) -> _Self:
        """逐元素乘法, 由子类实现."""
        raise NotImplementedError

    def div(self: _Self, other: Union[Number, TensorLike]) -> _Self:
        """逐元素除法, 由子类实现."""
        raise NotImplementedError

    def pow(self: _Self, other: Union[TensorLike, Number]) -> _Self:
        """逐元素乘方, 由子类实现."""
        raise NotImplementedError

    @overload
    def matmul(self: _Self, other: _Self) -> _Self: ...
    @overload
    def matmul(self: _Self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other):
        """矩阵乘法, 由子类实现; 与稠密张量相乘时返回稠密张量."""
        raise NotImplementedError

    def __pos__(self: _Self): return self
    def __neg__(self: _Self): return self.neg()

    @overload
    def __add__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __add__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __add__(self, other):
        return self.add(other)

    @overload
    def __radd__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __radd__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __radd__(self, other):
        return self.add(other)

    @overload
    def __sub__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __sub__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __sub__(self, other):
        return self.add(-other)

    @overload
    def __rsub__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __rsub__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __rsub__(self, other):
        return self.neg().add(other)

    def __mul__(self: _Self, other: Union[_Self, TensorLike, Number]) -> _Self:
        return self.mul(other)
    __rmul__ = __mul__

    def __truediv__(self: _Self, other: Union[TensorLike, Number]) -> _Self:
        return self.div(other)

    def __pow__(self: _Self, other: Union[TensorLike, Number]) -> _Self:
        return self.pow(other)

    @overload
    def __matmul__(self: _Self, other: _Self) -> _Self: ...
    @overload
    def __matmul__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __matmul__(self, other):
        return self.matmul(other)
