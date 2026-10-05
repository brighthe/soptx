# 移植自 brighthe/fealpy ``fealpy/utils/utils.py`` @ f474a5775, 仅保留 SOPTX 用到的 4 个函数.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""积分子系数的求值与形状工具."""

from typing import Optional, Union

from ..backend import backend_manager as bm
from ..typing import TensorLike, CoefLike
from ..mesh.mesh_base import HomogeneousMesh

__all__ = [
    'process_coef_func',
    'is_scalar',
    'is_tensor',
    'fill_axis',
]


def process_coef_func(
    coef: Optional[CoefLike],
    bcs: Optional[TensorLike]=None,
    mesh: Optional[HomogeneousMesh]=None,
    etype: Optional[Union[int, str]]=None,
    index: Optional[TensorLike]=None,
    n: Optional[TensorLike]=None
):
    r"""求出系数在积分点处的值; 系数不是函数时原样返回.

    Parameters
    ----------
    coef : CoefLike or None
        系数, 可以是标量、张量或函数. 函数按 ``coordtype`` 属性区分:
        ``'barycentric'`` (缺省) 以 ``coef(bcs, index=index)`` 调用,
        否则先把 ``bcs`` 映射为直角坐标 ``ps`` 再以 ``coef(ps)`` 调用;
        ``n`` 给出且函数恰有两个参数时以 ``coef(ps, n)`` 调用.
    bcs : TensorLike, optional
        积分点的重心坐标, 系数为函数时必须给出.
    mesh : HomogeneousMesh, optional
        网格, 用于把重心坐标映射为直角坐标.
    etype : int or str, optional
        积分所在的实体类型, 系数为函数时必须给出.
    index : TensorLike, optional
        参与积分的实体编号, 系数为函数时必须给出.
    n : TensorLike, optional
        积分点处的法向量, 只传给接受两个参数的直角坐标函数.

    Returns
    -------
    CoefLike
        系数为函数时是其在积分点处的值, 否则是 ``coef`` 本身.

    Raises
    ------
    RuntimeError
        系数为函数但缺少 ``index``、``bcs`` 或 ``etype``, 或重心坐标函数缺少
        齐次网格.
    """
    if callable(coef):
        if index is None:
            raise RuntimeError('The index should be provided for coef functions.')
        if bcs is None:
            raise RuntimeError('The bcs should be provided for coef functions.')
        if etype is None:
            raise RuntimeError('The etype should be provided for coef functions.')
        if getattr(coef, 'coordtype', 'barycentric') == 'barycentric':
            if (mesh is None) or (not isinstance(mesh, HomogeneousMesh)):
                raise RuntimeError('The mesh should be provided for cartesian coef functions.'
                                   'Note that only homogeneous meshes are supported here.')

            coef_val = coef(bcs, index=index)
        else:
            ps = mesh.bc_to_point(bcs, index=index)
            ##TODO: 适应不同情况的 coef, coef 的接口应该是 coef(ps, n) 或者 coef(ps)
            import inspect
            if (n is not None) & (len(inspect.signature(coef).parameters) == 2):
                coef_val = coef(ps, n)
            else:
                coef_val = coef(ps)
    else:
        coef_val = coef
    return coef_val


def is_scalar(input: Union[int, float, complex, TensorLike]) -> bool:
    """判断输入是否为标量: Python 数, 或只含一个元素的张量."""
    if isinstance(input, TensorLike):
        return bm.size(input) == 1
    else:
        return isinstance(input, (int, float, complex))


def is_tensor(input: Union[int, float, complex, TensorLike]) -> bool:
    """判断输入是否为至少含两个元素的张量."""
    if isinstance(input, TensorLike):
        return bm.size(input) >= 2
    return False


def fill_axis(input: TensorLike, ndim: int):
    """在末尾补长度为 1 的轴, 使张量维数达到 ``ndim``.

    Parameters
    ----------
    input : TensorLike
        输入张量.
    ndim : int
        目标维数.

    Returns
    -------
    TensorLike
        维数为 ``ndim`` 的张量; 维数已相等时返回输入本身.

    Raises
    ------
    RuntimeError
        输入维数大于 ``ndim``.
    """
    diff = ndim - input.ndim

    if diff > 0:
        return bm.reshape(input, input.shape + (1, ) * diff)
    elif diff == 0:
        return input
    else:
        raise RuntimeError(f"The dimension of the input should be smaller than {ndim}, "
                           f"but got shape {tuple(input.shape)}.")
