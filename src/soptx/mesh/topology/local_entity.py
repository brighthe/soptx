# 移植自 brighthe/fealpy ``fealpy/mesh/topology/local_entity.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Extract and canonically orient complete local subentity occurrences.

``extract_local_entity_occurrences`` converts a homogeneous source sector and
its parameterized :class:`LocalEntityGroup` layout into block-global child
connectivity rows.  ``canonicalize_local_entity_occurrences`` then selects the
allowed orientation whose vertex tuple is lexicographically smallest and
returns the corresponding complete child-node permutation.

Both steps are deliberately storage/relation-free: they do not assign sector
ids, deduplicate global entities, or create incidence relations.
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
    """Complete local occurrences bound to one concrete child Schema.

    ``schema`` is the immutable child :class:`EntitySchema` shared by all
    rows, and ``indices`` contains block-global node ids with shape
    ``(number_of_occurrences, schema.number_of_nodes())``.  Rows follow the
    child's complete canonical local-node order; higher-order interpolation
    nodes are not omitted.
    """

    schema: EntitySchema
    indices: Tensor


class CanonicalLocalEntityOccurrence(NamedTuple):
    """Canonically oriented rows of one homogeneous local occurrence group.

    ``indices`` has the same shape as the input occurrence and contains
    block-global node ids in the child's canonical full-node order.
    ``canonical_vertices`` contains the corresponding vertex sub-rows used for
    topological identity.  ``vertex_permutation`` and ``node_permutation``
    record, row by row, the supported orientation chosen to reach the
    canonical form.
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
    """Extract all local subentity occurrences for one topological dimension.

    This is the topology-construction bridge between a source
    :class:`EntitySector` and the parameterized
    :class:`~fealpy.mesh.schema.LocalEntityGroup` protocol.  It does not
    deduplicate global entities, assign derived sector ids, or construct
    relations; those responsibilities belong to later topology stages.

    For each group returned by
    :meth:`EntitySchema.local_entity_groups`, the source connectivity is
    gathered with the group's parent-local node columns and reshaped into one
    occurrence row per parent entity and local occurrence.  Consequently the
    returned rows contain the child Schema's complete local-node layout,
    including higher-order nodes.

    Parameters:
        sector: A homogeneous source sector whose ``indices`` width equals its
            Schema's complete local-node count.
        top_dim: Target topological dimension in the closed interval
            ``[0, sector.schema.top_dim]``.

    Returns:
        One result per :class:`LocalEntityGroup` returned by the parent
        Schema.  Multiple results may therefore share the same child Schema
        when the parent exposes multiple local groups at one dimension.

    Raises:
        TypeError: If ``sector`` is not an :class:`EntitySector` or
            ``top_dim`` is not a plain integer.
        ValueError: If ``top_dim`` is outside the parent Schema's supported
            interval.
        NotImplementedError: If the sector is variable-cardinality
            (``indptr is not None``) and cannot use a fixed local-node layout.
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
    """Choose a canonical orientation for every local occurrence row.

    For each child Schema, the allowed vertex automorphisms are enumerated
    through :meth:`EntitySchema.vertex_permutations`.  A candidate canonical
    row is obtained by applying the corresponding full
    :meth:`EntitySchema.node_permutation` to the original child-local node
    order.  The row-wise candidate with the lexicographically smallest
    canonical vertex tuple is selected.  Equal canonical vertex tuples keep
    the first candidate, making the result deterministic.

    Parameters:
        occurrence: A homogeneous local occurrence group produced by
            :func:`extract_local_entity_occurrences`.

    Returns:
        A canonical occurrence whose ``indices`` shape equals the input shape.
        Its per-row ``vertex_permutation`` and ``node_permutation`` can be
        used later to map source local occurrences to derived canonical
        entities.

    Raises:
        TypeError: If ``occurrence`` is not a
            :class:`LocalEntityOccurrence`.
        ValueError: If ``occurrence.indices`` is not rank-2 or its width does
            not match the child Schema's complete local-node count.
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
