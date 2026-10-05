# 移植自 brighthe/fealpy ``fealpy/functionspace/function.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""有限元函数: 空间与自由度值的组合."""

from typing import Generic, Union, TypeVar, Optional
from functools import partial

from ..backend import backend_manager as bm
from ..backend import TensorLike, Number
from ..typing import Index, _S

_FS = TypeVar('_FS')


class Function(Generic[_FS]):
    """有限元函数, 由所属空间与自由度值数组组成.

    以重心坐标调用即求值 (``uh(bcs)`` 等价于 ``space.value(uh.array, bcs)``);
    下标读写作用于自由度数组; 名字以 ``value`` 结尾的空间方法 (如
    ``grad_value``) 可直接作为属性调用, 已绑定自由度数组; 其余未定义的属性转发给
    自由度数组. 与数或张量的四则运算作用于自由度数组, 返回同一空间中的新函数.

    Parameters
    ----------
    space : FunctionSpace
        所属空间.
    array : TensorLike
        自由度值, 形状 ``(..., GDOF)``.
    coordtype : str, optional
        函数接受的坐标类型.
    """
    def __init__(self, space: _FS, array: TensorLike, coordtype: Optional[str]=None) -> None:
        self.space = space
        self.array = array
        self.coordtype = coordtype

    def __call__(self, bcs: TensorLike, index: Index=_S):
        return self.space.value(self.array, bcs, index=index)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.space}, {self.array})'

    def __getitem__(self, index: Index):
        return self.array[index]

    def __setitem__(self, index: Index, value: TensorLike):
        if bm.backend_name == 'jax':
            self.array = self.array.at[index].set(value)
        else:
            self.array[index] = value

    def __getattr__(self, item: str):
        if item in {'space', 'array', 'coordtype'}:
            return object.__getattribute__(self, item)

        if item.endswith('value') and hasattr(self.space, item):
            attr = getattr(self.space, item)
            if callable(attr):
                func = partial(attr, self.array)
                func.coordtype = attr.coordtype
                return func
            else:
                return attr

        return getattr(self.array, item)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(self.space, -self.array, self.coordtype)

    def __add__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array + other, self.coordtype)
    __radd__ = __add__

    def __iadd__(self, other: Union[TensorLike, Number]):
        self.array += other
        return self

    def __sub__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array - other, self.coordtype)

    def __rsub__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, other - self.array, self.coordtype)

    def __isub__(self, other: Union[TensorLike, Number]):
        self.array -= other
        return self

    def __mul__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array * other, self.coordtype)
    __rmul__ = __mul__

    def __imul__(self, other: Union[TensorLike, Number]):
        self.array *= other
        return self

    def __truediv__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array / other, self.coordtype)

    def __rtruediv__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, other / self.array, self.coordtype)

    def __itruediv__(self, other: Union[TensorLike, Number]):
        self.array /= other
        return self

    def __matmul__(self, other: TensorLike) -> TensorLike:
        return self.array @ other

    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return other @ self.array
