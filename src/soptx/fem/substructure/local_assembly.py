"""仅依赖细单元矩阵与自由度映射的子结构局部刚度组装."""

from dataclasses import dataclass
from operator import index as integer_index
from typing import Any

from soptx.backend import backend_manager as bm


@dataclass(frozen=True)
class LocalAssemblyMap:
    """同构子结构共享的矩阵散加映射.

    Attributes
    ----------
    n_cells : int
        每个子结构的细单元数.
    n_element_dofs : int
        每个细单元的自由度数.
    n_local_dofs : int
        每个子结构的自由度数.
    scatter_index : TensorLike
        按 (单元, 行, 列) 展平的局部矩阵位置, 与输入矩阵同设备.
        构造后调用方不得原地修改, 本对象不持有材料、积分子或 prototype.
    """

    n_cells: int
    n_element_dofs: int
    n_local_dofs: int
    scatter_index: Any


def build_local_assembly_map(cell2dof: Any, n_local_dofs: int, *, backend: Any = None) -> LocalAssemblyMap:
    """由局部单元自由度编号构造可复用的矩阵散加映射.

    Parameters
    ----------
    cell2dof : array_like
        形状为 (NC, n_edof) 的整数编号, 顺序须与单元矩阵一致.
    n_local_dofs : int
        子结构自由度数.
    backend : backend proxy or None
        与输入相容的后端, None 使用当前后端.

    Returns
    -------
    LocalAssemblyMap
        仅包含尺寸和编号的组装映射.

    Raises
    ------
    ValueError
        编号形状、类型或范围不合法.
    """
    backend = bm.get_current_backend() if backend is None else backend
    n_local_dofs = integer_index(n_local_dofs)
    if (getattr(cell2dof, 'ndim', None) != 2 or min(cell2dof.shape) <= 0
            or cell2dof.dtype not in (backend.int32, backend.int64)
            or n_local_dofs <= 0):
        raise ValueError("cell2dof 须为非空二维整数编号, n_local_dofs 须为正数.")
    if bool(backend.any(cell2dof < 0)) or bool(backend.any(cell2dof >= n_local_dofs)):
        raise ValueError("cell2dof 超出局部自由度范围.")
    dofs = backend.astype(cell2dof, backend.int64)
    slots = dofs[:, :, None] * n_local_dofs + dofs[:, None, :]
    return LocalAssemblyMap(
        int(dofs.shape[0]), int(dofs.shape[1]), n_local_dofs,
        backend.reshape(slots, (-1,)),
    )


def assemble_local_stiffness(KE_chunk: Any, assembly_map: LocalAssemblyMap, *, backend: Any = None) -> Any:
    """将细单元刚度批次散加为各子结构刚度, 不执行材料插值或积分.

    Parameters
    ----------
    KE_chunk : array_like
        形状为 (b, NC, n_edof, n_edof), 单元和行列顺序与映射一致.
    assembly_map : LocalAssemblyMap
        与输入后端和设备相容的组装映射.
    backend : backend proxy or None
        后端代理, None 使用当前后端.

    Returns
    -------
    array_like
        形状为 (b, n_local_dofs, n_local_dofs) 的局部刚度.

    Raises
    ------
    ValueError
        单元矩阵形状与映射不符.

    Notes
    -----
    不修改输入、不保存矩阵批次. 批次偏移与 bincount 顺序沿用原局部装配.
    """
    expected = (assembly_map.n_cells, assembly_map.n_element_dofs,
                assembly_map.n_element_dofs)
    if getattr(KE_chunk, 'ndim', None) != 4 or tuple(KE_chunk.shape[1:]) != expected:
        raise ValueError(
            f"KE_chunk 须具有形状 (b, {expected[0]}, {expected[1]}, {expected[2]}), "
            f"得到 {getattr(KE_chunk, 'shape', None)}."
        )
    backend = bm.get_current_backend() if backend is None else backend
    n_chunk, n_dof = KE_chunk.shape[0], assembly_map.n_local_dofs
    offsets = backend.arange(n_chunk, dtype=backend.int64,
                             device=backend.get_device(KE_chunk)) * (n_dof * n_dof)
    indices = backend.reshape(offsets[:, None] + assembly_map.scatter_index[None, :], (-1,))
    accumulated = backend.bincount(
        indices, weights=backend.reshape(KE_chunk, (-1,)),
        minlength=n_chunk * n_dof * n_dof,
    )
    return backend.reshape(accumulated, (n_chunk, n_dof, n_dof))
