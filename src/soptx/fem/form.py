# 移植自 brighthe/fealpy ``fealpy/fem/form.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""变分形式的积分子容器基类.

``Form`` 按组保存积分子, 并逐组产出单元局部张量与自由度映射; 全局装配由子类
``BilinearForm`` 与 ``LinearForm`` 完成.
"""

from typing import (
    Sequence, overload, Iterable, Dict, Tuple, Optional, Union, TypeVar, Generic,
    Callable
)

from ..typing import TensorLike, Size, Index
from ..backend import backend_manager as bm
from ..functionspace import FunctionSpace as _FS
from .integrator import Integrator, GroupIntegrator

from abc import ABC

import logging

logger = logging.getLogger(__name__)

_I = TypeVar('_I', bound=Integrator)
_IT = TypeVar('_IT')
Self = TypeVar('Self')
# NOTE: _Splitter 产出的分块与 Index 不同, 只有两种:
# (1) TensorLike: 实体编号, 或实体的布尔掩码.
# (2) slice: 实体的切片.
_SplitterInterface = Callable[[_FS, Integrator], Iterable[_IT]]
_Splitter = Union[_SplitterInterface[TensorLike], _SplitterInterface[slice]]


class Form(Generic[_I], ABC):
    """变分形式基类: 按组保存积分子, 产出局部张量.

    Parameters
    ----------
    *space : FunctionSpace
        一个或多个函数空间, 也可传入单个空间序列.
    batch_size : int, optional
        批量维长度, 0 表示不带批量维. 默认 0.

    Attributes
    ----------
    integrators : dict of str to Integrator
        组名到积分子的映射.
    splitters : dict of str to callable or None
        组名到分块器的映射, None 表示整体装配.
    batch_size : int
        批量维长度.
    sparse_shape : tuple of int
        全局张量形状 (不含批量维), 由子类的 ``_get_sparse_shape`` 给出.

    Raises
    ------
    ValueError
        没有给出任何空间.
    """
    _spaces: Tuple[_FS, ...]
    integrators: Dict[str, _I]
    # chunk_sizes: Dict[str, int]
    splitters: Dict[str, _Splitter]
    batch_size: int
    sparse_shape: Tuple[int, ...]

    @overload
    def __init__(self, space: _FS, /, *, batch_size: int=0): ...
    @overload
    def __init__(self, space: Tuple[_FS, ...], /, *, batch_size: int=0): ...
    @overload
    def __init__(self, *space: _FS, batch_size: int=0): ...
    def __init__(self, *space, batch_size: int=0):
        if len(space) == 0:
            raise ValueError("No space is given.")
        if isinstance(space[0], Sequence):
            space = space[0]
        self._spaces = space
        self.integrators = {}
        self.splitters = {}
        # self.chunk_sizes = {}
        self._cursor = 0
        self.batch_size = batch_size

        self._values_ravel_shape = (-1,) if self.batch_size == 0 else (self.batch_size, -1)
        self.sparse_shape = self._get_sparse_shape()

    def copy(self):
        """浅拷贝: 共享空间与积分子, ``sparse_shape`` 取反序 (供转置使用)."""
        new_obj = self.__class__(self._spaces, batch_size=self.batch_size)
        new_obj.integrators.update(self.integrators)
        # new_obj.chunk_sizes.update(self.chunk_sizes)
        new_obj.splitters.update(self.splitters)
        new_obj._values_ravel_shape = self._values_ravel_shape
        new_obj.sparse_shape = tuple(reversed(self.sparse_shape))
        return new_obj

    def __len__(self) -> int:
        return len(self.integrators)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self._spaces}"

    def _get_sparse_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError('Please implement the _get_sparse_shape method '
                                  'to generate the shape of the form.')

    @property
    def shape(self) -> Size:
        """全局张量形状, 带批量维时批量维在前."""
        if self.batch_size == 0:
            return self.sparse_shape
        return (self.batch_size,) + self.sparse_shape

    @property
    def space(self):
        """单空间时返回该空间, 多空间时返回空间元组."""
        if len(self._spaces) == 1:
            return self._spaces[0]
        else:
            return self._spaces

    @overload
    def add_integrator(self: Self, I: _I, /, *, region: Optional[Index] = None, splitter: Union[_Splitter, int, None] = None, group: str = ...) -> Self: ...
    @overload
    def add_integrator(self: Self, I: Sequence[_I], /, *, region: Optional[Index] = None, splitter: Union[_Splitter, int, None] = None, group: str = ...) -> Self: ...
    @overload
    def add_integrator(self: Self, *I: _I, region: Optional[Index] = None, splitter: Union[_Splitter, int, None] = None, group: str = ...) -> Self: ...
    def add_integrator(self, *I, region: Optional[Index] = None, splitter=None, group=None):
        """把积分子作为一组加入.

        Parameters
        ----------
        *I : Integrator
            一个或多个积分子, 也可传入积分子序列. 多个积分子合成一个
            ``GroupIntegrator``.
        region : Index, optional
            积分区域 (实体编号或掩码), 设置到积分子上.
        splitter : callable or int, optional
            分块器; 整数表示按该块长均匀分块 (``UniformSplitter``). 默认 None,
            即整体装配. 注意 SOPTX 的积分子不接受 ``indices`` 参数, 分块装配目前
            不可用, 见 ``docs/known-issues/README.md``.
        group : str, optional
            组名, 默认自动编号. 组名已存在时与原积分子相加.

        Returns
        -------
        Form
            本对象, 便于链式调用.
        """
        if len(I) == 0:
            logger.info("add_integrator() is called with no arguments.")
            return self

        if len(I) == 1 and isinstance(I[0], Sequence):
            I = tuple(I[0])

        if len(I) == 1:
            I = I[0]
            if region is not None:
                I.set_region(region)
        else:
            I = GroupIntegrator(*I, region=region)

        if isinstance(splitter, int):
            splitter = UniformSplitter(splitter)

        return self._add_integrator_impl(I, group, splitter)

    @overload
    def __lshift__(self: Self, other: Integrator) -> Self: ...
    def __lshift__(self, other):
        """``form << integrator``: 把积分子作为新的一组加入."""
        if isinstance(other, Integrator):
            return self._add_integrator_impl(other, None)
        else:
            return NotImplemented

    def _add_integrator_impl(self, I: _I, group: Optional[str] = None, splitter: Optional[_Splitter] = None):
        group = f'_group_{self._cursor}' if group is None else group
        self._cursor += 1

        if group in self.integrators:
            self.integrators[group] += I
            if splitter is not None:
                self.splitters[group] = splitter
        else:
            self.integrators[group] = I
            self.splitters[group] = splitter

        return self

    def _assembly_kernel(self, group: str, /, indices=None):
        integrator = self.integrators[group]
        if indices is None:
            value = integrator.assembly(self.space)
            etg = integrator.to_global_dof(self.space)
        else:
            value = integrator.assembly(self.space, indices=indices)
            etg = integrator.to_global_dof(self.space, indices=indices)
        if not isinstance(etg, (tuple, list)):
            etg = (etg, )
        return value, etg

    def assembly_local_iterative(self):
        """逐组 (分块时逐块) 产出局部张量与自由度映射.

        Yields
        ------
        tuple
            ``(局部张量, 自由度映射元组)``; 自由度映射元组依次对应各空间.
        """
        for key, int_ in self.integrators.items():
            splitter = self.splitters[key]
            if splitter is None:
                logger.debug(f"(ASSEMBLY LOCAL FULL) {key}")
                yield self._assembly_kernel(key)
            else:
                logger.debug(f"(ASSEMBLY LOCAL ITER) {key}")
                for indices in splitter(self.space, int_):
                    yield self._assembly_kernel(key, indices)


class UniformSplitter():
    """按固定块长把积分实体切成连续切片的分块器.

    Parameters
    ----------
    chunk_size : int
        每块的实体数.
    """
    def __init__(self, chunk_size: int, /): # 分块方法的参数
        self.chunk_size = chunk_size

    def __call__(self, space, integrator: Integrator): # 接收空间与积分子
        if isinstance(space, (list, tuple)):
            space = space[0]
        size = integrator.size(space.mesh)
        start = 0

        while start < size:
            stop = start + self.chunk_size
            if stop >= size:
                stop = size
            logger.debug(f"(FORM ITER) {stop}/{size}")
            yield slice(start, stop, 1)
            start = stop


# NOTE: 已弃用.
# `_assembly_kernel` 方法的迭代工具.
class IntegralIter():
    """按实体编号或分段点逐块计算积分子的迭代器 (已弃用).

    Parameters
    ----------
    integrator : Integrator
        积分子.
    indices_or_segments : TensorLike or iterable of TensorLike
        一维张量时视为分段点, 否则视为各块的实体编号.
    """
    def __init__(self, integrator: Integrator, /, indices_or_segments: Union[Iterable[TensorLike], TensorLike]):
        self.integrator = integrator
        self.indices_or_segments = indices_or_segments

    def kernel(self, space: Union[_FS, Tuple[_FS, ...]], /, indices: Index):
        """计算一块实体上的局部张量与自由度映射元组."""
        etg = self.integrator.to_global_dof(space, indices=indices)
        if not isinstance(etg, (tuple, list)):
            etg = (etg, )
        return self.integrator(space, indices=indices), etg

    def __call__(self, spaces: Tuple[_FS, ...]):
        if isinstance(self.indices_or_segments, TensorLike):
            return self._call_impl_segments(spaces, self.indices_or_segments)
        elif isinstance(self.indices_or_segments, Iterable):
            return self._call_impl_indices(spaces, self.indices_or_segments)
        else:
            raise TypeError(f"Unsupported indices or segments.")

    def _call_impl_indices(self, spaces: Tuple[_FS, ...], /, indices: Iterable[TensorLike]):
        for index in indices:
            yield self.kernel(spaces, index)

    def _call_impl_segments(self, spaces: Tuple[_FS, ...], /, segments: TensorLike):
        assert segments.ndim == 1
        start = 0
        stop = 0
        length = segments.shape[0] + 1

        for i in range(length):
            logger.debug(f"(FORM ITER) {i+1}/{length}")
            stop = segments[i] if (i + 1 < length) else None
            slicing = slice(start, stop, 1)
            yield self.kernel(spaces, slicing)
            start = stop

    @classmethod
    def split(cls, integrator: Integrator, /, chunk_size=0):
        """按块长 ``chunk_size`` 生成分段点并构造迭代器."""
        size = integrator.get_region().shape[0]
        if chunk_size >= size:
            segments = bm.empty((0,), dtype=bm.int64)
        else:
            segments = bm.arange(chunk_size, size, chunk_size, dtype=bm.int64)
        return cls(integrator, segments)
