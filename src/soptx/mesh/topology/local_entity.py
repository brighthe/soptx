# 移植自 brighthe/fealpy ``fealpy/mesh/topology/local_entity.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""提取局部子实体的完整出现并确定其规范定向.

``extract_local_entity_occurrences`` 按同类源分区及其参数化的
:class:`LocalEntityGroup` 布局, 生成以网格块全局节点编号表示的子实体连接行;
``canonicalize_local_entity_occurrences`` 再从允许的定向中选出顶点元组字典序最小
者, 并返回相应的完整子节点置换.

这两步刻意不涉及存储与关系: 不分配分区 id, 不对全局实体去重, 也不建立关联关系.
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

from ...backend import bm, Tensor
from ..storage import EntitySector

if TYPE_CHECKING:
    from ..schema.entity_schema import EntitySchema

__all__ = [
    "CanonicalLocalEntityOccurrence",
    "LocalEntityOccurrence",
    "canonicalize_local_entity_occurrences",
    "extract_local_entity_occurrences",
]


class LocalEntityOccurrence(NamedTuple):
    """绑定到一个具体子 Schema 的局部子实体完整出现.

    ``schema`` 是各行共享的不可变子 :class:`EntitySchema`; ``indices`` 为网格块
    全局节点编号, 形状 ``(出现次数, schema.number_of_nodes())``. 各行按子实体完整
    的规范局部节点顺序排列, 不省略高阶插值节点.
    """

    schema: EntitySchema
    indices: Tensor


class CanonicalLocalEntityOccurrence(NamedTuple):
    """一组同类局部出现经规范定向后的各行.

    ``indices`` 与输入同形状, 为按子实体规范完整节点顺序排列的网格块全局节点编号;
    ``canonical_vertices`` 为相应的顶点子行, 用于判定拓扑身份;
    ``vertex_permutation`` 与 ``node_permutation`` 逐行记录为达到规范形式所选用的
    定向.
    """

    schema: EntitySchema
    indices: Tensor
    canonical_vertices: Tensor
    vertex_permutation: Tensor
    node_permutation: Tensor


def extract_local_entity_occurrences(
    sector: EntitySector,
    top_dim: int,
) -> tuple[LocalEntityOccurrence, ...]:
    """提取某一拓扑维数的全部局部子实体出现.

    这是源 :class:`EntitySector` 与参数化的
    :class:`~soptx.mesh.schema.LocalEntityGroup` 协议之间的拓扑构造桥梁; 不对全局
    实体去重, 不分配派生分区 id, 也不建立关系, 这些由后续拓扑阶段负责.

    对 :meth:`EntitySchema.local_entity_groups` 返回的每一组, 用该组在父实体中的
    局部节点列收集源连接, 再整形为 "每个父实体每次局部出现一行". 因此返回的各行
    包含子 Schema 完整的局部节点布局, 含高阶节点.

    Parameters
    ----------
    sector : EntitySector
        同类源分区, ``indices`` 宽度等于其 Schema 的完整局部节点数.
    top_dim : int
        目标拓扑维数, 取值于闭区间 ``[0, sector.schema.top_dim]``.

    Returns
    -------
    tuple of LocalEntityOccurrence
        父 Schema 返回的每个 :class:`LocalEntityGroup` 对应一项; 父实体在同一维数
        上有多个局部组时, 多项可能共用同一个子 Schema.

    Raises
    ------
    TypeError
        ``sector`` 不是 :class:`EntitySector`, 或 ``top_dim`` 不是普通整数.
    ValueError
        ``top_dim`` 超出父 Schema 支持的区间.
    NotImplementedError
        分区为变长 (``indptr is not None``), 无法使用固定的局部节点布局.
    """
    if not isinstance(sector, EntitySector):
        raise TypeError("sector must be an EntitySector instance")
    if type(top_dim) is not int:
        raise TypeError("top_dim must be a plain integer")
    if sector.indptr is not None:
        raise NotImplementedError(
            "local entity occurrence extraction requires a homogeneous "
            "sector with indptr=None"
        )

    schema = sector.schema
    groups = schema.local_entity_groups(top_dim)
    source_indices = sector.indices
    source_dtype = source_indices.dtype
    source_device = bm.get_device(source_indices)

    occurrences: list[LocalEntityOccurrence] = []
    for group in groups:
        local_node_indices = bm.tensor(
            group.local_node_indices,
            dtype=source_dtype,
            device=source_device,
        )
        gathered = source_indices[:, local_node_indices]
        child_width = group.schema.number_of_nodes()
        gathered = bm.reshape(gathered, (-1, child_width))
        occurrences.append(LocalEntityOccurrence(group.schema, gathered))

    return tuple(occurrences)


def canonicalize_local_entity_occurrences(
    occurrence: LocalEntityOccurrence,
) -> CanonicalLocalEntityOccurrence:
    """为每一行局部出现选定规范定向.

    对每个子 Schema, 用 :meth:`EntitySchema.vertex_permutations` 枚举允许的顶点
    自同构; 对原始子局部节点顺序施加相应的完整 :meth:`EntitySchema.node_permutation`
    得到候选行, 逐行选取规范顶点元组字典序最小的候选. 顶点元组相同时保留第一个
    候选, 使结果确定.

    Parameters
    ----------
    occurrence : LocalEntityOccurrence
        由 :func:`extract_local_entity_occurrences` 生成的一组同类局部出现.

    Returns
    -------
    CanonicalLocalEntityOccurrence
        ``indices`` 与输入同形状; 逐行的 ``vertex_permutation`` 与
        ``node_permutation`` 可在之后用于把源局部出现映射到派生的规范实体.

    Raises
    ------
    TypeError
        ``occurrence`` 不是 :class:`LocalEntityOccurrence`.
    ValueError
        ``occurrence.indices`` 不是二维, 或宽度与子 Schema 的完整局部节点数不符.
    """
    if not isinstance(occurrence, LocalEntityOccurrence):
        raise TypeError("occurrence must be a LocalEntityOccurrence")
    if len(occurrence.indices.shape) != 2:
        raise ValueError("occurrence.indices must be rank-2")

    schema = occurrence.schema
    source = occurrence.indices
    child_width = schema.number_of_nodes()
    if source.shape[-1] != child_width:
        raise ValueError(
            f"occurrence width must be {child_width}, got {source.shape[-1]}"
        )

    source_dtype = source.dtype
    source_device = bm.get_device(source)
    vertex_permutations = schema.vertex_permutations()
    vertex_positions = schema.local_vertices()
    row_count = source.shape[0]

    allowed_vertex = bm.tensor(
        vertex_permutations,
        dtype=source_dtype,
        device=source_device,
    )
    allowed_node = bm.tensor(
        tuple(schema.node_permutation(vp) for vp in vertex_permutations),
        dtype=source_dtype,
        device=source_device,
    )

    candidates = source[:, allowed_node]  # (NO, NP, child_width)
    candidate_vertices = candidates[:, :, vertex_positions]  # (NO, NP, NV)

    permutation_count = allowed_vertex.shape[0]
    best_permutation = bm.zeros((row_count,), dtype=source_dtype)
    best_vertices = candidate_vertices[:, 0, :]

    for permutation_index in range(1, permutation_count):
        candidate = candidate_vertices[:, permutation_index, :]
        candidate_is_less = bm.zeros((row_count,), dtype=bm.bool)
        still_equal = bm.ones((row_count,), dtype=bm.bool)

        for vertex_index in range(candidate.shape[-1]):
            column_is_less = candidate[:, vertex_index] < best_vertices[:, vertex_index]
            column_is_equal = candidate[:, vertex_index] == best_vertices[:, vertex_index]
            candidate_is_less = candidate_is_less | (
                still_equal & column_is_less
            )
            still_equal = still_equal & column_is_equal

        replace = candidate_is_less
        best_permutation = bm.where(
            replace,
            bm.full_like(best_permutation, permutation_index),
            best_permutation,
        )
        best_vertices = bm.where(
            replace[:, None],
            candidate,
            best_vertices,
        )

    row_ids = bm.arange(row_count, dtype=source_dtype, device=source_device)
    canonical = candidates[row_ids, best_permutation]
    chosen_vertex = allowed_vertex[best_permutation]
    chosen_node = allowed_node[best_permutation]

    return CanonicalLocalEntityOccurrence(
        schema=schema,
        indices=canonical,
        canonical_vertices=best_vertices,
        vertex_permutation=chosen_vertex,
        node_permutation=chosen_node,
    )
