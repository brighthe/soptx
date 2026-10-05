"""由接口解和局部缩聚结果恢复完整全局位移."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from fealpy.backend import backend_manager as bm

from .layout import HasGlobalDofs, StructuredSubstructureLayout
from .reduction_adapter import normalize_local_reduction


def recover_full_displacement_batches(
    layout: StructuredSubstructureLayout,
    sub_meshes: Sequence[Any],
    displacement_batches: Iterable[Any],
) -> Any:
    """由流式产出的各子结构边界与内部位移拼出全尺度位移向量.

    Parameters
    ----------
    layout : StructuredSubstructureLayout
        整体有限元布局和局部到全局自由度映射.
    sub_meshes : sequence
        按批次编号排列的全部子结构.
    displacement_batches : iterable
        连续覆盖全部子结构的批次迭代器, 每项提供 ``start``, ``end``,
        ``boundary`` 与 ``internal`` 属性, 形状分别为 ``(end - start, n_b)``
        与 ``(end - start, n_i)``; 通常来自
        ``iter_exact_internal_displacement_batches``.

    Returns
    -------
    Any
        全局位移向量, 形状 ``(total_full_dofs,)``.

    Raises
    ------
    ValueError
        子结构为空, 批次区间不连续, 或未完整覆盖子结构.

    Notes
    -----
    每批按子结构全局自由度一次写入边界与内部分量. 相邻子结构共享的接口
    自由度会被写入多次, 协调接口空间下这些值相同, 因此结果与整批
    ``recover_full_displacement`` 一致. 跨批只保留全尺度向量本身.
    """
    if not sub_meshes:
        raise ValueError("sub_meshes 不能为空.")
    n_sub_total = len(sub_meshes)
    positions = layout._substructure_positions(sub_meshes)

    U_full: Any = bm.zeros((layout.total_full_dofs,), dtype=bm.float64)
    next_start = 0
    for batch in displacement_batches:
        start = int(batch.start)
        end = int(batch.end)
        if start != next_start or end <= start or end > n_sub_total:
            raise ValueError(
                "displacement_batches 必须以连续半开区间完整覆盖子结构; "
                f"期望 start={next_start}, 当前区间为 [{start}, {end})."
            )
        global_dofs = bm.stack(
            [
                layout.get_substructure_global_dofs(pos, sub_mesh)
                for pos, sub_mesh in zip(
                    positions[start:end], sub_meshes[start:end]
                )
            ],
            axis=0,
        )
        b_dofs = sub_meshes[start].b_dofs
        i_dofs = sub_meshes[start].i_dofs
        U_full = bm.set_at(
            U_full,
            bm.reshape(global_dofs[:, b_dofs], (-1,)),
            bm.reshape(bm.asarray(batch.boundary), (-1,)),
        )
        U_full = bm.set_at(
            U_full,
            bm.reshape(global_dofs[:, i_dofs], (-1,)),
            bm.reshape(bm.asarray(batch.internal), (-1,)),
        )
        next_start = end

    if next_start != n_sub_total:
        raise ValueError(
            "displacement_batches 未完整覆盖全部子结构; "
            f"已覆盖 {next_start}, 总数为 {n_sub_total}."
        )
    return U_full

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

