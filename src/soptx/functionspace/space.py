# 移植自 brighthe/fealpy ``fealpy/functionspace/space.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""函数空间基类."""

from typing import Union, Callable, Optional, Any

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, Number, _S, Size
from .function import Function
from .utils import zero_dofs


class FunctionSpace():
    r"""函数空间基类.

    声明子类须提供的接口 (基函数、取值、自由度计数与映射、插值), 并实现与具体空间
    无关的自由度数组与有限元函数构造.
    """
    ftype: Any
    itype: Any
    device: Any

    # 基函数
    def basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        """积分点处的基函数值, 由子类实现."""
        raise NotImplementedError
    def grad_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        """积分点处的基函数梯度, 由子类实现."""
        raise NotImplementedError
    def hess_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        """积分点处的基函数 Hessian, 由子类实现."""
        raise NotImplementedError

    # 取值
    def value(self, uh: TensorLike, p: TensorLike, index: Index=_S) -> TensorLike:
        """有限元函数在积分点处的值, 由子类实现."""
        raise NotImplementedError
    def grad_value(self, uh: TensorLike, p: TensorLike, index: Index=_S) -> TensorLike:
        """有限元函数在积分点处的梯度, 由子类实现."""
        raise NotImplementedError

    # 计数
    def number_of_global_dofs(self) -> int:
        """全局自由度个数, 由子类实现."""
        raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int:
        """每个实体上的局部自由度个数, 由子类实现."""
        raise NotImplementedError

    # 实体与自由度的关系
    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """单元到全局自由度的映射, 由子类实现."""
        raise NotImplementedError
    def face_to_dof(self, index: Index=_S) -> TensorLike:
        """面到全局自由度的映射, 由子类实现."""
        raise NotImplementedError

    # 插值
    def interpolate(self, source: Union[Callable[..., TensorLike], TensorLike, Number],
                    uh: TensorLike, dim: Optional[int]=None, index: Index=_S) -> TensorLike:
        """把函数或数据插值到空间中, 由子类实现."""
        raise NotImplementedError

    def array(self, batch: Union[int, Size, None]=None, *, dtype=None, device=None) -> TensorLike:
        """创建全零的自由度值数组.

        Parameters
        ----------
        batch : int or tuple of int, optional
            批量维形状; None 或 0 表示不带批量维.
        dtype : dtype, optional
            浮点类型, 默认取空间的 ``ftype``.
        device : device, optional
            设备.

        Returns
        -------
        TensorLike
            形状 ``(*batch, GDOF)`` 的全零数组.
        """
        GDOF = self.number_of_global_dofs()
        if (batch is None) or (batch == 0):
            batch = tuple()

        elif isinstance(batch, int):
            batch = (batch, )

        shape = batch + (GDOF, )

        if dtype is None:
            dtype = self.ftype 

        return bm.zeros(shape, dtype=dtype, device=device)

    def function(self, array: Optional[TensorLike]=None,
                batch: Union[int, Size, None]=None, *,
                coordtype='barycentric', 
                dtype=None, device=None):
        """创建空间中的有限元函数.

        Parameters
        ----------
        array : TensorLike, optional
            自由度值; 为 None 时用 ``array`` 创建全零数组.
        batch : int or tuple of int, optional
            ``array`` 为 None 时的批量维形状.
        coordtype : str, optional
            函数接受的坐标类型, 默认 ``'barycentric'``.
        dtype, device : optional
            ``array`` 为 None 时的浮点类型与设备, 默认取空间的 ``ftype`` 与 ``device``.

        Returns
        -------
        Function
            有限元函数.
        """
        if array is None:
            if dtype is None:
                dtype = self.ftype
            if device is None:
                device = self.device
            array = self.array(batch=batch, dtype=dtype, device=device)

        return Function(self, array, coordtype)
