# 移植自 brighthe/fealpy ``fealpy/fem/form.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""变分形式的积分子容器基类.

``Form`` 按组保存积分子, 并逐组产出单元局部张量与自由度映射; 全局装配由子类
``BilinearForm`` 与 ``LinearForm`` 完成.
"""

from typing import (
    Sequence, overload, Dict, Tuple, Optional, TypeVar, Generic,
)

from ..typing import Size, Index
from ..functionspace import FunctionSpace as _FS
from .integrator import Integrator, GroupIntegrator

from abc import ABC

import logging

logger = logging.getLogger(__name__)

_I = TypeVar('_I', bound=Integrator)
Self = TypeVar('Self')


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
        self._cursor = 0
        self.batch_size = batch_size

        self._values_ravel_shape = (-1,) if self.batch_size == 0 else (self.batch_size, -1)
        self.sparse_shape = self._get_sparse_shape()

    def copy(self):
        """浅拷贝: 共享空间与积分子, ``sparse_shape`` 取反序 (供转置使用)."""
        new_obj = self.__class__(self._spaces, batch_size=self.batch_size)
        new_obj.integrators.update(self.integrators)
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
    def add_integrator(self: Self, I: _I, /, *, region: Optional[Index] = None, group: str = ...) -> Self: ...
    @overload
    def add_integrator(self: Self, I: Sequence[_I], /, *, region: Optional[Index] = None, group: str = ...) -> Self: ...
    @overload
    def add_integrator(self: Self, *I: _I, region: Optional[Index] = None, group: str = ...) -> Self: ...
    def add_integrator(self, *I, region: Optional[Index] = None, group=None):
        """把积分子作为一组加入.

        Parameters
        ----------
        *I : Integrator
            一个或多个积分子, 也可传入积分子序列. 多个积分子合成一个
            ``GroupIntegrator``.
        region : Index, optional
            积分区域 (实体编号或掩码), 设置到积分子上.
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

        return self._add_integrator_impl(I, group)

    @overload
    def __lshift__(self: Self, other: Integrator) -> Self: ...
    def __lshift__(self, other):
        """``form << integrator``: 把积分子作为新的一组加入."""
        if isinstance(other, Integrator):
            return self._add_integrator_impl(other, None)
        else:
            return NotImplemented

    def _add_integrator_impl(self, I: _I, group: Optional[str] = None):
        group = f'_group_{self._cursor}' if group is None else group
        self._cursor += 1

        if group in self.integrators:
            self.integrators[group] += I
        else:
            self.integrators[group] = I

        return self

    def _assembly_kernel(self, group: str, /):
        integrator = self.integrators[group]
        value = integrator.assembly(self.space)
        etg = integrator.to_global_dof(self.space)
        if not isinstance(etg, (tuple, list)):
            etg = (etg, )
        return value, etg

    def assembly_local_iterative(self):
        """逐组产出局部张量与自由度映射.

        Yields
        ------
        tuple
            ``(局部张量, 自由度映射元组)``; 自由度映射元组依次对应各空间.
        """
        for key in self.integrators:
            logger.debug(f"(ASSEMBLY LOCAL) {key}")
            yield self._assembly_kernel(key)
