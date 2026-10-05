# 移植自 brighthe/fealpy ``fealpy/functionspace/utils.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""函数空间的自由度与张量基工具."""

from typing import Optional, Tuple, Union
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size


def zero_dofs(gdofs: int, dims: Union[Size, int, None]=None, *, dtype=None):
    """创建全零的自由度数组, 形状 ``(gdofs, *dims)``; ``dims`` 为 None 或 0 时为 ``(gdofs, )``."""
    kwargs = {'dtype': dtype}

    if dims is None:
        shape = (gdofs, )
    elif isinstance(dims, int):
        shape = (gdofs, ) if dims == 0 else (gdofs, dims)
    else:
        shape = (gdofs, *dims)

    return bm.zeros(shape, **kwargs)


def flatten_indices(shape: Size, permute: Size) -> TensorLike:
    """按轴置换后展平的顺序, 给出原张量各元素的展平序号.

    Parameters
    ----------
    shape : tuple of int
        原张量形状.
    permute : tuple of int
        轴置换, 含义同 ``permute_dims``.

    Returns
    -------
    TensorLike
        形状为 ``shape`` 的整数张量, 元素为该位置在置换后展平序列中的序号.
    """
    permuted_shape = [shape[d] for d in permute]
    numel = prod(permuted_shape)
    # 置换后的序号
    permuted_indices = bm.arange(numel, dtype=bm.int64).reshape(permuted_shape)
    # 置换前的序号
    inv_permute = [None, ] * len(permute)
    for d in range(len(permute)):
        inv_permute[permute[d]] = d

    return bm.permute_dims(permuted_indices, inv_permute)


def to_tensor_dof(to_dof: TensorLike, dof_numel: int, gdof: int, dof_priority: bool=True) -> TensorLike:
    """把实体到标量自由度的映射扩展为实体到张量自由度的映射.

    Parameters
    ----------
    to_dof : TensorLike or tuple of TensorLike
        实体到标量自由度的映射 ``(NE, ldof)``; 变阶空间为 ``(cell2dof,
        cell2dofLocation)`` 压缩格式.
    dof_numel : int
        每个标量自由度的分量数.
    gdof : int
        标量自由度总数.
    dof_priority : bool, optional
        为 True 时自由度优先排列 (同一分量的全部自由度相邻), 否则分量优先.
        默认 True.

    Returns
    -------
    TensorLike or tuple of TensorLike
        实体上张量自由度的全局编号 ``(NE, ldof * dof_numel)``; 压缩格式输入时
        返回同格式的元组.

    Notes
    -----
    压缩格式分支固定按 2 个分量、自由度优先展开, 不使用 ``dof_numel`` 与
    ``dof_priority``.
    """
    if isinstance(to_dof, tuple):
        scell2dof, scell2dofLocation = to_dof[0], to_dof[1]
        sgdof = bm.max(scell2dof) + 1
        context = bm.context(scell2dof)
        NC = scell2dofLocation.shape[0] - 1
        ldof = scell2dofLocation[1:] - scell2dofLocation[:-1]
        cell2dofLocation = scell2dofLocation*2
        cell2dof = bm.zeros(2*scell2dof.shape[0], **context)
        scell2dof = list(bm.split(scell2dof,scell2dofLocation[1:-1]))
        #if dof_priority:
        for i in range(NC):
            idx =  bm.arange(cell2dofLocation[i],
                             cell2dofLocation[i]+ldof[i], **context)
            cell2dof[idx] = scell2dof[i] 
            idx2 = idx + ldof[i]
            cell2dof[idx2] = scell2dof[i] + gdof 
        return cell2dof, cell2dofLocation
        #else:
        #    for i in range(NC):
        #        idx =  bm.arange(scell2dofLocation[i],
        #                         scell2dofLocation[i]+ldof[i], **context)*2
        #        cell2dof[idx] = scell2dof[i]*2 
        #        idx2 = idx + 1
        #        cell2dof[idx2] = scell2dof[i]*2 + 1 
        #    return cell2dof, cell2dofLocation
    else:
        context = bm.context(to_dof)
        indices = bm.arange(gdof*dof_numel, **context)
        num_entity = to_dof.shape[0]

        if dof_priority:
            indices = indices.reshape(dof_numel, gdof)
            indices = indices[:, to_dof] # (dof_numel, entity, ldof)
            indices = bm.swapaxes(indices, 0, 1) # (entity, dof_numel, ldof)
        else:
            indices = indices.reshape(gdof, dof_numel)
            indices = indices[to_dof, :] # (entity, ldof, dof_numel)

        return indices.reshape(num_entity, -1)


def tensor_basis(shape: Size, *, dtype=None, device=None) -> TensorLike:
    """生成由 0 与 1 组成的张量基.

    Parameters
    ----------
    shape : tuple of int
        每个张量基的形状.
    dtype, device : optional
        数据类型与设备.

    Returns
    -------
    TensorLike
        形状 ``(numel, *shape)``, 第 ``k`` 个基只在展平后的第 ``k`` 个位置为 1.
    """
    numel = prod(shape)
    return bm.eye(numel, dtype=dtype, device=device).reshape((numel,) + shape)


def normal_strain(gphi: TensorLike, indices: TensorLike, *, out: Optional[TensorLike]=None) -> TensorLike:
    """组装正应变部分的应变--位移矩阵.

    Parameters
    ----------
    gphi : TensorLike
        标量基函数梯度, 形状 ``(..., ldof, GD)``.
    indices : TensorLike
        各分量在展平自由度中的位置, 形状 ``(ldof, GD)``.
    out : TensorLike, optional
        输出张量, 默认新建.

    Returns
    -------
    TensorLike
        形状 ``(..., GD, GD*ldof)``.

    Raises
    ------
    ValueError
        ``out`` 的形状不符.
    """
    kwargs = {'dtype': gphi.dtype}
    if hasattr(gphi, 'device'):
        kwargs['device'] = gphi.device

    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: TensorLike, indices: TensorLike, *, out: Optional[TensorLike]=None) -> TensorLike:
    """组装剪应变部分的应变--位移矩阵.

    Parameters
    ----------
    gphi : TensorLike
        标量基函数梯度, 形状 ``(..., ldof, GD)``.
    indices : TensorLike
        各分量在展平自由度中的位置, 形状 ``(ldof, GD)``.
    out : TensorLike, optional
        输出张量, 默认新建.

    Returns
    -------
    TensorLike
        形状 ``(..., NNZ, GD*ldof)``, 其中 ``NNZ = GD*(GD-1)//2``, 按 ``(i, j)``,
        ``i < j`` 的字典序排列.

    Raises
    ------
    ValueError
        ``GD < 2`` 或 ``out`` 的形状不符.
    """
    kwargs = {'dtype': gphi.dtype}
    if hasattr(gphi, 'device'):
        kwargs['device'] = gphi.device

    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD-1))//2
    new_shape = gphi.shape[:-2] + (NNZ, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    cursor = 0
    for i in range(0, GD-1):
        for j in range(i+1, GD):
            out[..., cursor, indices[:, i]] = gphi[..., :, j]
            out[..., cursor, indices[:, j]] = gphi[..., :, i]
            cursor += 1

    return out
