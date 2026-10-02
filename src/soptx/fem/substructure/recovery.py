"""由接口解和局部缩聚结果恢复完整全局位移."""

from __future__ import annotations

from typing import Any, List

from fealpy.backend import backend_manager as bm

from .layout import HasGlobalDofs, StructuredSubstructureLayout
from .reduction_adapter import normalize_local_reduction

def recover_full_displacement(
    layout: StructuredSubstructureLayout,
    sub_meshes: List[Any],
    condensors: Any,
    system: HasGlobalDofs,
    interface_displacement: Any,
) -> Any:
    """由接口位移和局部缩聚结果恢复完整的全局位移向量.

    Parameters
    ----------
    layout : StructuredSubstructureLayout
        整体有限元布局和局部到全局自由度映射.
    sub_meshes : list
        子结构列表.
    condensors : Any
        缩聚器列表或单个批量缩聚结果, 形状约定见
        ``normalize_local_reduction``.
    system : HasGlobalDofs
        接口系统或至少具有 ``global_dofs`` 属性的视图.
    interface_displacement : Any
        接口自由度上的位移, 形状 ``(n_interface,)``.

    Returns
    -------
    Any
        全局位移向量, 形状 ``(total_full_dofs,)``.

    Raises
    ------
    ValueError
        子结构与缩聚结果不匹配, 或位移长度与接口自由度数不一致.

    Notes
    -----
    内部自由度不被任何两个子结构共享, 因此写回时不存在重复索引;
    接口自由度直接由 ``interface_displacement`` 一次写入.
    """
    if not sub_meshes:
        raise ValueError("sub_meshes 不能为空.")
    u_b = bm.asarray(interface_displacement, dtype=bm.float64)
    if len(u_b) != len(system.global_dofs):
        raise ValueError(
            f"interface_displacement 的长度必须等于接口自由度数 "
            f"{len(system.global_dofs)}; 当前为 {len(u_b)}."
        )

    n_b = int(sub_meshes[0].n_b)
    b_interface = layout.interface_indices(sub_meshes, system.global_dofs)
    _, recover = normalize_local_reduction(condensors, len(sub_meshes), n_b)

    U_full: Any = bm.zeros((layout.total_full_dofs,), dtype=bm.float64)
    U_full = bm.set_at(U_full, system.global_dofs, u_b)

    # (B, n_b) -> (B, n_i): 批量缩聚器一次完成, 缩聚器列表逐个恢复后堆叠.
    u_sub_i = recover(u_b[b_interface])

    i_global = bm.stack(
        [
            layout.get_substructure_global_dofs(pos, sub_mesh)[sub_mesh.i_dofs]
            for pos, sub_mesh in zip(
                layout._substructure_positions(sub_meshes), sub_meshes
            )
        ],
        axis=0,
    )
    return bm.set_at(
        U_full, bm.reshape(i_global, (-1,)), bm.reshape(u_sub_i, (-1,))
    )

