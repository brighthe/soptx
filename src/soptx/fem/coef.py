# 移植自 brighthe/fealpy ``fealpy/utils/utils.py`` @ f474a5775, 仅保留 SOPTX 用到的 4 个函数.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

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
    r"""Fetch the result Tensor if `coef` is a function."""
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
            ##TODO:适应不同情况的coef, coef的接口应该是coef(ps, n)或者coef(ps)
            import inspect
            if (n is not None) & (len(inspect.signature(coef).parameters) == 2):
                coef_val = coef(ps, n)
            else:
                coef_val = coef(ps)
    else:
        coef_val = coef
    return coef_val


def is_scalar(input: Union[int, float, complex, TensorLike]) -> bool:
    if isinstance(input, TensorLike):
        return bm.size(input) == 1
    else:
        return isinstance(input, (int, float, complex))


def is_tensor(input: Union[int, float, complex, TensorLike]) -> bool:
    if isinstance(input, TensorLike):
        return bm.size(input) >= 2
    return False


def fill_axis(input: TensorLike, ndim: int):
    diff = ndim - input.ndim

    if diff > 0:
        return bm.reshape(input, input.shape + (1, ) * diff)
    elif diff == 0:
        return input
    else:
        raise RuntimeError(f"The dimension of the input should be smaller than {ndim}, "
                           f"but got shape {tuple(input.shape)}.")
