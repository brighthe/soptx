# 移植自 brighthe/fealpy ``fealpy/mesh/topology/builder.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Construct per-root derived sectors and canonical incidence relations.

This module implements the topology half of the FEALPy mesh four-layer model.
Topology construction is rooted in one :class:`EntitySector`; it never merges
distinct root semantic sectors and never creates, trims, merges, or renumbers
the canonical ``"node"`` sector owned by :class:`MeshBlock`.

For conforming Lagrange meshes, each local subentity uses two identity levels:

``(child Schema Python type, canonical vertex tuple)``
    identifies a topological entity;

``(concrete Schema id, canonical full-node tuple)``
    validates conforming consistency for that topological identity.

The implementation extracts occurrences with :func:`extract_local_entity_occurrences`,
canonicalizes orientations with :func:`canonicalize_local_entity_occurrences`,
and stores both forward and reverse canonical :class:`EntityRelation` pairs.
"""


__all__ = [
    "TopologyBuilder",
    "TopRelationConnector",
    "TopRelationInferer",
]

from collections.abc import Iterable, Iterator
from typing import NamedTuple, TYPE_CHECKING

from ...backend import bm
from ...backend import Tensor
from ..schema import (
    HexahedronSchema,
    PrismSchema,
    PyramidSchema,
    PolygonSchema,
    QuadrilateralSchema,
    EdgeSchema,
    TetrahedronSchema,
    TriangleSchema,
)
from ..storage import MeshBlock, EntitySector, Relation
from .local_entity import (
    CanonicalLocalEntityOccurrence,
    LocalEntityOccurrence,
    canonicalize_local_entity_occurrences,
    extract_local_entity_occurrences,
)

if TYPE_CHECKING:
    from ..schema.entity_schema import EntitySchema

_LEGACY_SCHEMA_TYPES = {
    "node": None,
    "edge": EdgeSchema,
    "tri": TriangleSchema,
    "quad": QuadrilateralSchema,
    "tet": TetrahedronSchema,
    "prism": PrismSchema,
    "pyramid": PyramidSchema,
    "hex": HexahedronSchema,
}


def _unique_unordered_rows_across(*arrays: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Unique rows across 2D tensors after row-wise canonicalization.

    Rows are treated as unordered sets by sorting each row first.
    """
    if not arrays:
        raise ValueError("at least one array is required")

    for arr in arrays:
        if len(arr.shape) != 2:
            raise ValueError("only 2D tensors are supported")

    total = bm.concat(arrays, axis=0)  # (total_rows, ncols)
    canonical_total = bm.sort(total, axis=1)

    indices = bm.lexsort(tuple(reversed(canonical_total.T)), axis=0)  # sorted <-> original
    sorted_canonical = canonical_total[indices]

    diff_flag = bm.any(sorted_canonical[1:] != sorted_canonical[:-1], axis=1)
    true = bm.ones((1,), dtype=bm.bool, device=diff_flag.device)
    diff_flag = bm.concat([true, diff_flag])

    # choose representative rows in their original ordering
    unique = total[indices[diff_flag]]
    sorted_to_unique = bm.cumulative_sum(diff_flag, axis=0) - 1  # sorted -> unique

    original_to_sorted = bm.empty_like(indices)
    original_to_sorted[indices] = bm.arange(
        len(indices),
        dtype=original_to_sorted.dtype,
        device=original_to_sorted.device,
    )
    total_to_unique = sorted_to_unique[original_to_sorted]  # original -> unique

    array_indptr = [0]
    for arr in arrays:
        array_indptr.append(array_indptr[-1] + len(arr))

    arr_to_unique = tuple(
        total_to_unique[array_indptr[i]:array_indptr[i + 1]]
        for i in range(len(arrays))
    )

    return unique, arr_to_unique


def _unique_ordered_rows(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Unique exact 2D rows in lexicographic order and map inputs to uniques."""
    if len(rows.shape) != 2:
        raise ValueError("only 2D tensors are supported")
    if rows.shape[0] == 0:
        return rows, bm.zeros(
            (0,),
            dtype=rows.dtype,
            device=bm.get_device(rows),
        )

    indices = bm.lexsort(tuple(reversed(rows.T)), axis=0)
    sorted_rows = rows[indices]
    diff_flag = bm.any(sorted_rows[1:] != sorted_rows[:-1], axis=1)
    true = bm.ones((1,), dtype=bm.bool, device=diff_flag.device)
    first_flag = bm.concat([true, diff_flag])

    unique = rows[indices[first_flag]]
    sorted_to_unique = bm.cumulative_sum(first_flag, axis=0) - 1

    original_to_sorted = bm.empty_like(indices)
    original_to_sorted[indices] = bm.arange(
        len(indices),
        dtype=original_to_sorted.dtype,
        device=original_to_sorted.device,
    )
    inverse = sorted_to_unique[original_to_sorted]
    return unique, inverse


def get_total_face(cell: Tensor, local_face: list[list[int]]) -> Tensor:
    total_face = cell[:, local_face]
    NFC = len(local_face[0])
    return bm.reshape(total_face, (-1, NFC))


def _lower_entities(
    schema: type["EntitySchema"],
    excluded: set[str] | None = None,
) -> dict[str, list[list[int]]]:
    """Return the OFace entries that point to lower-dimensional schemas."""
    excluded = set() if excluded is None else excluded
    return {
        name: local_indices
        for name, local_indices in schema.OFace.items()
        if name not in excluded and _LEGACY_SCHEMA_TYPES[name].top_dim < schema.top_dim
    }


def _derived_sector_id(
    root: EntitySector,
    child_schema: "EntitySchema",
    *,
    use_legacy_name: bool,
) -> str:
    """Return a root-scoped derived sector id.

    A single classic root whose sector id equals its Schema name keeps the
    child Schema's short name, preserving the historical single-type layout.
    Other roots use ``f"{root.id}_{child_schema.name}"`` so distinct semantic
    sectors cannot share a derived id.
    """
    if use_legacy_name:
        return child_schema.name
    return f"{root.id}_{child_schema.name}"


def _derived_sector_id_for_root_id(
    root_id: str,
    child_schema: "EntitySchema",
    *,
    use_legacy_name: bool,
) -> str:
    """Return the derived id associated with a root's child Schema."""
    if use_legacy_name:
        return child_schema.name
    return f"{root_id}_{child_schema.name}"


def _ordered_rows_to_existing(
    existing: Tensor,
    new: Tensor,
    target_id: str,
) -> Tensor:
    """Map exact canonical rows in ``new`` to an existing canonical sector.

    ``existing`` must already be the canonical, unique, lexicographically
    ordered row set of its target sector.  The function builds a combined
    ordered-unique index, verifies that the first ``len(existing)`` rows still
    map to themselves, and returns target ids for the ``new`` rows.
    """
    existing_count = int(existing.shape[0])
    new_count = int(new.shape[0])
    if new_count == 0:
        return bm.zeros((0,), dtype=existing.dtype)
    if existing_count == 0:
        raise ValueError(f"target sector {target_id!r} is empty")

    combined = bm.concat([existing, new], axis=0)
    _, inverse = _unique_ordered_rows(combined)
    expected = bm.arange(
        existing_count,
        dtype=inverse.dtype,
        device=bm.get_device(inverse),
    )
    if bool(bm.any(inverse[:existing_count] != expected)):
        raise ValueError(
            f"target sector {target_id!r} is not a canonical superset of "
            "the requested occurrences"
        )
    return inverse[existing_count:]


def _validate_conforming_occurrences(
    canonical_by_schema: dict["EntitySchema", CanonicalLocalEntityOccurrence],
) -> None:
    """Validate the double-key conforming rule within one root sector.

    The topological key is ``(type(child Schema), canonical vertex tuple)``.
    On the first occurrence the concrete Schema id and canonical full-node
    tuple are recorded.  Later occurrences must have the same concrete Schema
    id and the same canonical full nodes; otherwise the mesh is nonconforming.
    """
    seen: dict[
        tuple[type["EntitySchema"], tuple[int, ...]],
        tuple[str, tuple[int, ...]],
    ] = {}

    for schema, canonical in canonical_by_schema.items():
        schema_type = type(schema)
        vertices = bm.to_numpy(canonical.canonical_vertices).tolist()
        full_nodes = bm.to_numpy(canonical.indices).tolist()
        for vertex_row, full_row in zip(vertices, full_nodes):
            vertex_key = tuple(int(value) for value in vertex_row)
            full_key = tuple(int(value) for value in full_row)
            identity_key = (schema_type, vertex_key)

            if identity_key not in seen:
                seen[identity_key] = (schema.id, full_key)
                continue

            expected_schema_id, expected_full = seen[identity_key]
            if schema.id != expected_schema_id:
                raise NotImplementedError(
                    "p-nonconforming or Schema-incompatible subentities are "
                    "not supported for one root sector"
                )
            if full_key != expected_full:
                raise ValueError(
                    "illegal conforming mesh: the same canonical vertices "
                    "have different canonical full nodes"
                )


class ConstructResult(NamedTuple):
    face_type: str
    face: Tensor
    cell_to_face: tuple[Tensor, ...]


class TopologyBuilder:
    """Build per-root derived sectors and canonical incidence relations.

    ``construct`` is the stable entry point.  It handles one or more root cell
    sectors independently:

    - extract complete child occurrences;
    - canonicalize allowed orientations;
    - validate the double-key conforming rules;
    - deduplicate exact canonical full-node rows;
    - bind derived sectors with ``source_cell_sector_id``;
    - build root/derived/node relations and materialize reverse pairs.

    The class does not parse sector ids into cell/face/edge roles.  Role
    interpretation belongs to :class:`MeshView`, based on its root anchor and
    the stored provenance and relations.
    """

    @classmethod
    def construct_lower_dims(
        cls,
        cells: Iterable[Tensor],
        local_face_dicts: Iterable[dict[str, list[list[int]]]],
    ) -> Iterator[ConstructResult]:
        """
        Construct lower-dimensional elements.

        Parameters:
            cells (Iterable[Tensor]):
                A sequence of cells, containing tensors in the shape of (NC, NVF).
            local_face_dicts (Iterable[dict[str, list[list[int]]]]):
                A sequence of local face dictionaries. Keys are used to tag the
                faces, and values are local face indices.

        Returns:
            Iterator[ConstructResult]:
                An iterator of ConstructResult, which contains the face type name,
                the unique face array, and the cell-to-face mapping for each input
                cell.
        """
        # NOTE: {face_kind: ([total_face,], [NFC,])}
        face_table: dict[str, tuple[list[Tensor], list[int]]] = {}

        for cell, local_face_dict in zip(cells, local_face_dicts):
            for face_kind, local_face in local_face_dict.items():
                if face_kind not in face_table:
                    face_table[face_kind] = ([], [])

                face_table[face_kind][0].append(get_total_face(cell, local_face))
                face_table[face_kind][1].append(len(local_face))

        for face_kind, (total_face_list, nfc_list) in face_table.items():
            face, js = _unique_unordered_rows_across(*total_face_list)
            cell2faces = tuple(
                bm.reshape(j, (-1, NFC))
                for NFC, j in zip(nfc_list, js)
            )

            yield ConstructResult(face_kind, face, cell2faces)

    @classmethod
    def _construct_from_blocks(
        cls,
        storage: MeshBlock,
        blocks: list[EntitySector],
        excluded: set[str],
    ) -> list[EntitySector]:
        """Construct one OFace layer from ``blocks`` and return touched sectors."""
        constructed: list[EntitySector] = []

        for const_result in cls.construct_lower_dims(
            [block.indices for block in blocks],
            [_lower_entities(block.schema, excluded) for block in blocks],
        ):
            face_type_name, face_array, cell2face_from_each_cell = const_result

            if face_type_name not in storage.sectors:
                storage.add_sector(
                    EntitySector(
                        id=face_type_name,
                        schema=_LEGACY_SCHEMA_TYPES[face_type_name](),
                        indices=face_array,
                    ),
                    root=False,
                )
                face_array_to_sector = None
            else:
                old_face_array = storage.sectors[face_type_name].indices
                merged, (old_to_merged, face_array_to_sector) = _unique_unordered_rows_across(
                    old_face_array, face_array
                )
                if len(merged) != len(old_face_array) or bool(
                    bm.any(old_to_merged != bm.arange(
                        len(old_face_array),
                        dtype=old_to_merged.dtype,
                        device=old_to_merged.device,
                    ))
                ):
                    raise ValueError(
                        f"constructing {face_type_name!r} would renumber an existing sector; "
                        "construct all lower-dimensional OFace entries from roots first"
                    )
                storage.sectors[face_type_name].indices = merged

            face_block = storage.get_sector(face_type_name)
            constructed.append(face_block)

            for cell2face, block in zip(cell2face_from_each_cell, blocks):
                if face_array_to_sector is not None:
                    cell2face = face_array_to_sector[cell2face]
                storage.add_relation(Relation(
                    src_sector_id=block.id,
                    tgt_sector_id=face_type_name,
                    tgt_indices=cell2face,
                ))

        return constructed

    @classmethod
    def construct(
        cls,
        storage: MeshBlock,
        src_name: str | None = None,
        exclude: list[str] | None = None,
    ):
        """Construct lower-dimensional sectors per root cell sector.

        The operation is in-place on ``storage``.  Each root sector is handled
        independently: occurrences are extracted from its parameterized local
        entity groups, canonically oriented, deduplicated, and bound to a
        derived sector with ``source_cell_sector_id`` set to the root id.
        Nested canonical relations are built for derived sectors, and reverse
        pairs are materialized.  Distinct root semantic sectors are never
        merged.

        This method is not idempotent.  It assumes the requested derived
        sectors and relations do not already exist; rebuild paths must clear
        stale sectors/relations before invoking it.

        Parameters:
            storage (MeshBlock): The mesh storage object to modify.
            src_name (str, optional): The name of the source block to start from.
                If None, all root blocks are used. Default is None.
            exclude (list[str], optional): A list of lower-dimensional shape names
                to exclude from construction. Default is None.

        Returns:
            list[EntitySector]: Newly constructed lower-dimensional sectors.
        """
        excluded = set() if exclude is None else set(exclude)
        if src_name is None:
            current_blocks = [
                storage.get_sector(name)
                for name in storage.root_cell_sector_ids
            ]
        else:
            current_blocks = [storage.get_sector(src_name)]

        if not current_blocks:
            return []

        use_legacy_names = (
            len(storage.root_cell_sector_ids) == 1
            and current_blocks[0].id == current_blocks[0].schema.name
        )
        constructed: list[EntitySector] = []
        polygon_roots = [
            root for root in current_blocks if isinstance(root.schema, PolygonSchema)
        ]
        fixed_roots = [
            root for root in current_blocks if not isinstance(root.schema, PolygonSchema)
        ]
        for root in polygon_roots:
            constructed.extend(
                cls._construct_polygon_root(
                    storage,
                    root,
                    excluded,
                    use_legacy_name=use_legacy_names,
                )
            )
        for root in fixed_roots:
            constructed.extend(
                cls._construct_derived_sectors_for_root(
                    storage,
                    root,
                    excluded,
                    use_legacy_name=use_legacy_names,
                )
            )
        for root in fixed_roots:
            cls._construct_nested_relations_for_root(
                storage,
                root,
                excluded,
                use_legacy_name=use_legacy_names,
            )
        cls._materialize_reverse_relations(storage)
        return constructed

    @classmethod
    def _construct_polygon_root(
        cls,
        storage: MeshBlock,
        root: EntitySector,
        excluded: set[str],
        *,
        use_legacy_name: bool,
    ) -> list[EntitySector]:
        """Construct cyclic edges and ragged incidence for one polygon root."""
        indptr = root.indptr
        if indptr is None:
            raise ValueError("PolygonSchema root sectors require indptr")

        vertices = root.indices
        dtype = vertices.dtype
        device = bm.get_device(vertices)
        counts = indptr[1:] - indptr[:-1]
        cell_count = int(counts.shape[0])
        occurrence_count = int(vertices.shape[0])
        cell_ids = bm.repeat(
            bm.arange(cell_count, dtype=dtype, device=device),
            counts,
        )
        occurrence_ids = bm.arange(occurrence_count, dtype=dtype, device=device)
        local_ids = occurrence_ids - bm.repeat(indptr[:-1], counts)

        if occurrence_count:
            order = bm.lexsort((vertices, cell_ids), axis=0)
            sorted_cells = cell_ids[order]
            sorted_vertices = vertices[order]
            duplicate_vertex = bm.logical_and(
                sorted_cells[1:] == sorted_cells[:-1],
                sorted_vertices[1:] == sorted_vertices[:-1],
            )
            if bool(bm.any(duplicate_vertex)):
                raise ValueError("polygon cells must not repeat a vertex")

        successor = bm.concat([vertices[1:], vertices[:1]], axis=0)
        if occurrence_count:
            successor = bm.set_at(
                successor,
                indptr[1:] - 1,
                vertices[indptr[:-1]],
            )
        total_edge = bm.stack([vertices, successor], axis=1)
        if bool(bm.any(total_edge[:, 0] == total_edge[:, 1])):
            raise ValueError("polygon cells must not contain degenerate edges")

        canonical_edge = bm.sort(total_edge, axis=1)
        edge_indices, cell_to_edge = _unique_ordered_rows(canonical_edge)
        edge_count = int(edge_indices.shape[0])

        if occurrence_count:
            _, incidence_counts = bm.unique_counts(cell_to_edge)
            if bool(bm.any(incidence_counts > 2)):
                raise ValueError(
                    "classic PolygonMesh requires at most two cells per edge"
                )

            orientation = bm.all(
                total_edge == edge_indices[cell_to_edge],
                axis=1,
            )
            orientation_sum = bm.zeros(
                (edge_count,),
                dtype=bm.int32,
                device=device,
            )
            orientation_sum = bm.index_add(
                orientation_sum,
                cell_to_edge,
                bm.astype(orientation, bm.int32),
            )
            if bool(bm.any(
                bm.logical_and(incidence_counts == 2, orientation_sum != 1)
            )):
                raise ValueError(
                    "neighboring polygon cells must traverse shared edges "
                    "in opposite directions"
                )

            edge_order = bm.argsort(cell_to_edge, stable=True)
            sorted_edge = cell_to_edge[edge_order]
            first_flag = bm.concat([
                bm.ones((1,), dtype=bm.bool, device=device),
                sorted_edge[1:] != sorted_edge[:-1],
            ])
            last_flag = bm.concat([
                sorted_edge[1:] != sorted_edge[:-1],
                bm.ones((1,), dtype=bm.bool, device=device),
            ])
            first_occurrence = edge_order[first_flag]
            last_occurrence = edge_order[last_flag]
            edge_to_cell = bm.stack([
                cell_ids[first_occurrence],
                cell_ids[last_occurrence],
                local_ids[first_occurrence],
                local_ids[last_occurrence],
            ], axis=1)
            oriented_edge = total_edge[first_occurrence]
            classic_orientation = bm.all(
                total_edge == oriented_edge[cell_to_edge],
                axis=1,
            )
        else:
            orientation = bm.zeros((0,), dtype=bm.bool, device=device)
            edge_to_cell = bm.zeros((0, 4), dtype=dtype, device=device)
            oriented_edge = edge_indices
            classic_orientation = orientation

        constructed: list[EntitySector] = []
        edge_id: str | None = None
        if "edge" not in excluded:
            edge_schema = EdgeSchema()
            edge_id = _derived_sector_id(
                root,
                edge_schema,
                use_legacy_name=use_legacy_name,
            )
            edge = EntitySector(
                id=edge_id,
                schema=edge_schema,
                indices=edge_indices,
                source_cell_sector_id=root.id,
                attributes={
                    "_polygon_edge_to_cell": edge_to_cell,
                    "_polygon_oriented_edge": oriented_edge,
                },
            )
            storage.add_sector(edge)
            storage.add_relation(Relation(
                src_sector_id=root.id,
                tgt_sector_id=edge_id,
                src_indices=cell_ids,
                tgt_indices=cell_to_edge,
            ))
            storage.add_relation(Relation(
                src_sector_id=edge_id,
                tgt_sector_id="node",
                tgt_indices=edge_indices,
            ))
            constructed.append(edge)

        if "node" not in excluded:
            storage.add_relation(Relation(
                src_sector_id=root.id,
                tgt_sector_id="node",
                src_indices=cell_ids,
                tgt_indices=vertices,
            ))

        root.attributes["_polygon_local_edge_indices"] = local_ids
        root.attributes["_polygon_canonical_edge_sign"] = orientation
        root.attributes["_polygon_cell_to_edge_sign"] = classic_orientation
        if edge_id is not None:
            root.attributes["_polygon_edge_sector_id"] = edge_id
        return constructed

    @classmethod
    def _construct_derived_sectors_for_root(
        cls,
        storage: MeshBlock,
        root: EntitySector,
        excluded: set[str],
        *,
        use_legacy_name: bool,
    ) -> list[EntitySector]:
        """Build all lower-dimensional sectors for one root sector.

        Occurrences are grouped by concrete child Schema, canonically
        oriented, validated for conforming consistency, and deduplicated by
        exact canonical full-node rows.  Each resulting sector records its
        ``source_cell_sector_id`` and receives direct root→derived and
        root→node relations.
        """
        occurrences_by_schema: dict["EntitySchema", list[LocalEntityOccurrence]] = {}
        for top_dim in range(1, root.schema.top_dim):
            for occurrence in extract_local_entity_occurrences(root, top_dim):
                if occurrence.schema.name in excluded:
                    continue
                occurrences_by_schema.setdefault(occurrence.schema, []).append(occurrence)

        canonical_by_schema: dict[
            "EntitySchema",
            CanonicalLocalEntityOccurrence,
        ] = {}
        for child_schema, occurrences in occurrences_by_schema.items():
            combined = bm.concat(
                [occurrence.indices for occurrence in occurrences],
                axis=0,
            )
            canonical_by_schema[child_schema] = canonicalize_local_entity_occurrences(
                LocalEntityOccurrence(child_schema, combined)
            )

        _validate_conforming_occurrences(canonical_by_schema)

        constructed: list[EntitySector] = []
        root_count = int(root.indices.shape[0])
        for child_schema, canonical in canonical_by_schema.items():
            occurrences = occurrences_by_schema[child_schema]
            unique_indices, inverse = _unique_ordered_rows(canonical.indices)

            cell_to_derived_parts: list[Tensor] = []
            offset = 0
            for occurrence in occurrences:
                count = int(occurrence.indices.shape[0])
                local_count = count // root_count
                part_inverse = inverse[offset:offset + count]
                cell_to_derived_parts.append(
                    bm.reshape(part_inverse, (root_count, local_count))
                )
                offset += count
            cell_to_derived = bm.concat(cell_to_derived_parts, axis=1)

            derived_id = _derived_sector_id(
                root,
                child_schema,
                use_legacy_name=use_legacy_name,
            )
            if derived_id in storage.sectors:
                existing = storage.get_sector(derived_id)
                raise ValueError(
                    f"derived sector {derived_id!r} for root {root.id!r} "
                    "already exists in the block"
                )
            derived = EntitySector(
                id=derived_id,
                schema=child_schema,
                indices=unique_indices,
                source_cell_sector_id=root.id,
            )
            storage.add_sector(derived)
            storage.add_relation(Relation(
                src_sector_id=root.id,
                tgt_sector_id=derived_id,
                tgt_indices=cell_to_derived,
            ))
            constructed.append(derived)

        if "node" not in excluded:
            storage.add_relation(Relation(
                src_sector_id=root.id,
                tgt_sector_id="node",
                tgt_indices=root.indices,
            ))

        return constructed

    @classmethod
    def _construct_nested_relations_for_root(
        cls,
        storage: MeshBlock,
        root: EntitySector,
        excluded: set[str],
        *,
        use_legacy_name: bool,
    ) -> None:
        """Build nested relations among sectors derived from one root.

        Derived sectors are processed from highest to lowest dimension.  For
        each source sector, its child occurrences are extracted and matched to
        the already-built root-scoped target sectors.  This produces
        face→edge, edge→node, and analogous lower-dimensional chains without
        merging sectors across different roots.
        """
        derived = [
            sector
            for sector in storage.sectors.values()
            if sector.source_cell_sector_id == root.id
        ]
        derived.sort(
            key=lambda sector: sector.schema.top_dim,
            reverse=True,
        )

        for source in derived:
            if source.schema.top_dim == 0:
                continue
            occurrences_by_schema: dict[
                "EntitySchema",
                list[LocalEntityOccurrence],
            ] = {}
            for top_dim in range(1, source.schema.top_dim):
                for occurrence in extract_local_entity_occurrences(source, top_dim):
                    if occurrence.schema.name in excluded:
                        continue
                    occurrences_by_schema.setdefault(
                        occurrence.schema,
                        [],
                    ).append(occurrence)

            source_count = int(source.indices.shape[0])
            for child_schema, occurrences in occurrences_by_schema.items():
                combined = bm.concat(
                    [occurrence.indices for occurrence in occurrences],
                    axis=0,
                )
                canonical = canonicalize_local_entity_occurrences(
                    LocalEntityOccurrence(child_schema, combined)
                )
                target_id = _derived_sector_id_for_root_id(
                    root.id,
                    child_schema,
                    use_legacy_name=use_legacy_name,
                )
                target = storage.get_sector(target_id)
                occurrence_to_target = _ordered_rows_to_existing(
                    target.indices,
                    canonical.indices,
                    target_id,
                )

                source_to_target_parts: list[Tensor] = []
                offset = 0
                for occurrence in occurrences:
                    count = int(occurrence.indices.shape[0])
                    local_count = count // source_count
                    part = occurrence_to_target[offset:offset + count]
                    source_to_target_parts.append(
                        bm.reshape(part, (source_count, local_count))
                    )
                    offset += count
                source_to_target = bm.concat(source_to_target_parts, axis=1)

                storage.add_relation(Relation(
                    src_sector_id=source.id,
                    tgt_sector_id=target_id,
                    tgt_indices=source_to_target,
                ))

            if "node" not in excluded:
                storage.add_relation(Relation(
                    src_sector_id=source.id,
                    tgt_sector_id="node",
                    tgt_indices=source.indices,
                ))

    @classmethod
    def _materialize_reverse_relations(cls, storage: MeshBlock) -> None:
        """Materialize the canonical reverse for every stored relation.

        The inverse is added only when the reverse ordered pair is absent.
        Since :class:`MeshBlock` enforces one relation per ordered pair, the
        result is deterministic and preserves transpose/incidence consistency.
        """
        for (src_id, tgt_id), relation in list(storage.relations.items()):
            reverse_id = (tgt_id, src_id)
            if reverse_id in storage.relations:
                continue
            storage.add_relation(relation.inverse())

    @classmethod
    def construct_nested_relations(
        cls,
        storage: MeshBlock,
        src_name: str | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Optionally construct relations among already-created lower entities."""
        excluded = set() if exclude is None else set(exclude)
        if src_name is None:
            current_blocks = [
                storage.get_sector(name)
                for root_name in storage.root_cell_sector_ids
                for name in _lower_entities(storage.get_sector(root_name).schema, excluded)
                if name in storage.sectors
            ]
        else:
            current_blocks = [storage.get_sector(src_name)]

        while current_blocks:
            current_blocks = cls._construct_from_blocks(storage, current_blocks, excluded)


class TopRelationConnector:
    @classmethod
    def _new_to_existing(
        cls,
        existing: Tensor,
        new: Tensor,
        tgt_name: str,
    ) -> Tensor:
        merged, (existing_to_merged, new_to_merged) = _unique_unordered_rows_across(existing, new)
        if len(merged) != len(existing) or bool(
            bm.any(existing_to_merged != bm.arange(
                len(existing),
                dtype=existing_to_merged.dtype,
                device=existing_to_merged.device,
            ))
        ):
            raise ValueError(
                f"connecting to {tgt_name!r} requires target entities that are not in the existing sector"
            )
        return new_to_merged

    @classmethod
    def connect(
        cls,
        storage: MeshBlock,
        src_name: str,
        tgt_name: str,
    ) -> Relation:
        """Connect src -> tgt mapping for existing tgt sector.
        This is an in-place operation that modifies the ``storage`` object.

        A temporary tgt layer is built from src only. The temporary tgt entities
        are matched back to the existing tgt sector, then the temporary relation
        is remapped into the storage numbering.

        Parameters:
            storage (MeshBlock): The mesh storage object to modify.
            src_name (str): The name of the source block.
            tgt_name (str): The name of the target block.

        Returns:
            Relation: The constructed relation from src to tgt.
        """
        if tgt_name not in storage.sectors:
            raise ValueError(f"target sector {tgt_name!r} does not exist")

        source = storage.get_sector(src_name)
        if tgt_name not in source.schema.OFace:
            raise ValueError(f"{tgt_name!r} is not an OFace entry of {src_name!r}")
        local_entities = source.schema.OFace[tgt_name]
        construct_result = next(TopologyBuilder.construct_lower_dims(
            [source.indices],
            [{tgt_name: local_entities}],
        ))
        new_tgt = construct_result.face
        src_to_new_tgt = construct_result.cell_to_face[0]
        new_to_existing = cls._new_to_existing(
            storage.get_sector(tgt_name).indices,
            new_tgt,
            tgt_name,
        )
        src_to_tgt = new_to_existing[src_to_new_tgt]
        relation = Relation(
            src_sector_id=src_name,
            tgt_sector_id=tgt_name,
            tgt_indices=src_to_tgt,
        )
        storage.add_relation(relation)
        return relation


class TopRelationInferer:
    _pattern_select_cache: dict[tuple[str, str, int], Tensor] = {}

    @staticmethod
    def _dim(storage: MeshBlock, name: str) -> int:
        return storage.get_sector(name).schema.top_dim

    @classmethod
    def _select_first_occurrence_positions(cls, row: Tensor) -> Tensor:
        if len(row) == 0:
            return bm.asarray([], dtype=bm.int32)

        _, first_indices, _, _ = bm.unique_all(row)
        order = bm.argsort(first_indices, axis=0)
        return first_indices[order]

    @classmethod
    def _deduplicate_homogeneous_by_pattern(
        cls,
        tgt_indices: Tensor,
        pattern_key: tuple[str, str, int],
    ) -> Tensor:
        if len(tgt_indices) == 0:
            if len(tgt_indices.shape) >= 2:
                return bm.reshape(tgt_indices, (0, tgt_indices.shape[1]))
            return bm.reshape(tgt_indices, (0, 0))

        if len(tgt_indices.shape) != 2:
            raise ValueError("inferred relation is not homogeneous")

        row_width = tgt_indices.shape[1]
        select_pos = cls._pattern_select_cache.get(pattern_key)

        if select_pos is None:
            select_pos = cls._select_first_occurrence_positions(tgt_indices[0])
            cls._pattern_select_cache[pattern_key] = select_pos
        elif len(select_pos) > 0 and bool(bm.any(select_pos >= row_width)):
            select_pos = cls._select_first_occurrence_positions(tgt_indices[0])
            cls._pattern_select_cache[pattern_key] = select_pos

        return tgt_indices[:, select_pos]

    @classmethod
    def _compose_homogeneous(cls, src_to_mid: Relation, mid_to_tgt: Relation) -> Tensor:
        if src_to_mid.src_indices is not None or mid_to_tgt.src_indices is not None:
            raise ValueError("only homogeneous relations are supported for inference")

        composed = mid_to_tgt.tgt_indices[src_to_mid.tgt_indices]
        return bm.reshape(composed, (len(src_to_mid.tgt_indices), -1))

    @classmethod
    def _merge_candidates(
        cls,
        candidates: list[Tensor],
        src_name: str,
        tgt_name: str,
        mid_dim: int
    ) -> Tensor:
        if not candidates:
            raise ValueError("no candidates to merge")

        if len(candidates) == 1:
            merged = candidates[0]
        else:
            merged = bm.concat(candidates, axis=1)
        return cls._deduplicate_homogeneous_by_pattern(
            tgt_indices=merged,
            pattern_key=(src_name, tgt_name, mid_dim)
        )

    @classmethod
    def _iter_adjacent_children(
        cls,
        storage: MeshBlock,
        mid_name: str,
    ) -> Iterator[tuple[str, Relation]]:
        parent_dim = cls._dim(storage, mid_name)
        for (src, tgt), relation in storage.relations.items():
            if src != mid_name:
                continue
            if cls._dim(storage, tgt) != parent_dim - 1:
                continue
            yield tgt, relation

    @classmethod
    def _infer_from(cls, storage: MeshBlock, src_name: str, dst_name: str) -> None:
        src_dim = cls._dim(storage, src_name)
        dst_dim = cls._dim(storage, dst_name)
        if src_dim <= dst_dim:
            return

        if src_dim - 1 < dst_dim:
            raise ValueError(f"no adjacent lower-dimensional relation found for {src_name!r}")

        for mid_dim in range(src_dim - 1, dst_dim, -1): # in [src_dim-1, dst_dim+1]
            # 1) Get all `src -> parent` relations
            src_to_mids: list[tuple[str, Relation]] = []

            if mid_dim == src_dim - 1: # get the highest-dimensional's children by its schema name
                src_to_mids = list(cls._iter_adjacent_children(storage, src_name))
            else: # get the rest by their dimension
                for (rel_src, rel_tgt), relation in storage.relations.items():
                    if rel_src != src_name:
                        continue
                    if cls._dim(storage, rel_tgt) != mid_dim:
                        continue
                    src_to_mids.append((rel_tgt, relation))

            if not src_to_mids:
                raise ValueError(f"cannot infer relation from {src_name!r} to {dst_name!r}")

            # 2) Get all `mid -> dst` relations and compose them with
            #   `src -> mid` to get `src -> dst` candidates.
            candidates_of_dst: dict[str, list[Tensor]] = {}
            for mid_name, src_to_mid in src_to_mids:
                for dst_name, mid_to_dst in cls._iter_adjacent_children(storage, mid_name):
                    if (src_name, dst_name) in storage.relations:
                        composed = storage.relations[(src_name, dst_name)].tgt_indices
                    else:
                        composed = cls._compose_homogeneous(src_to_mid, mid_to_dst)
                    candidates_of_dst.setdefault(dst_name, []).append(composed)

            for dst_name, candidates in candidates_of_dst.items():
                merged = cls._merge_candidates(candidates, src_name, dst_name, mid_dim)
                storage.add_relation(Relation(
                    src_sector_id=src_name,
                    tgt_sector_id=dst_name,
                    tgt_indices=merged,
                ))

        if (src_name, dst_name) not in storage.relations:
            raise ValueError(f"cannot infer relation from {src_name!r} to {dst_name!r}")

    @classmethod
    def infer(cls, storage: MeshBlock, src_name: str, dst_name: str) -> None:
        src_dim = cls._dim(storage, src_name)
        dst_dim = cls._dim(storage, dst_name)

        if src_dim <= dst_dim:
            raise ValueError(f"expect src dimension > dst dimension, got {src_name!r} -> {dst_name!r}")

        cls._infer_from(storage, src_name, dst_name)
