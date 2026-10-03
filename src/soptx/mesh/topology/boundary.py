# 移植自 brighthe/fealpy ``fealpy/mesh/topology/boundary.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Infer boundary masks from canonical sector-pair relations.

Boundary queries use ``EntitySector.id`` rather than Schema names or parsed
ontology strings.  For a single root, boundary is defined by incidence counts
against codimension-one derived sectors.  For multiple roots of different
topological dimension, block-wide boundary remains rooted in the highest root
dimension until anchored :class:`MeshView` context is introduced in P4.
"""

from dataclasses import dataclass
from functools import cached_property

from ...backend import bm
from ...backend import Tensor
from ..storage import MeshBlock

__all__ = ["BoundaryInfo", "BoundaryInferencer"]


@dataclass(frozen=True)
class BoundaryInfo:
    mask: Tensor

    @cached_property
    def index(self) -> Tensor:
        return bm.nonzero(self.mask)[0]


class BoundaryInferencer:
    """Infer block-wide boundary masks from canonical relations.

    Codimension-one entities are boundary entities when they are adjacent to
    exactly one top-dimensional source sector.  Higher/lower-dimensional
    boundary masks are then propagated through existing forward relations.
    """

    @staticmethod
    def _top_dimension(storage: MeshBlock) -> int:
        if not storage.root_cell_sector_ids:
            return -1
        return max(storage.get_sector(name).schema.top_dim for name in storage.root_cell_sector_ids)

    @staticmethod
    def _accumulate_target_counts(num_target: int, relation_targets: list[Tensor]) -> Tensor:
        count = bm.zeros((num_target,), dtype=bm.int32)

        for tgt_indices in relation_targets:
            flat = bm.reshape(tgt_indices, (-1,))
            if len(flat) == 0:
                continue

            unique_idx, unique_count = bm.unique_counts(flat) # type: ignore
            count[unique_idx] += unique_count

        return count

    @classmethod
    def infer_codim1(
        cls,
        storage: MeshBlock,
        codim1_names: list[str],
        binfo: dict[str, BoundaryInfo],
    ) -> dict[str, BoundaryInfo]:
        """Infer boundary info for all codimension-1 entity blocks.

        Boundary rule:
            Count adjacent top-dimensional entities. A codim-1 entity is on
            boundary iff adjacency count equals 1.
        """
        top_dim = cls._top_dimension(storage)
        if top_dim < 0:
            return {}

        for codim1_name in codim1_names:
            if codim1_name in binfo:
                continue
            sector = storage.get_sector(codim1_name)
            if sector.schema.top_dim != top_dim - 1:
                raise ValueError(
                    f"Entity block {codim1_name!r} is not codimension-1, "
                    f"top_dim={sector.schema.top_dim}, expected {top_dim - 1}"
                )
            num_target = len(sector.indices)
            relation_targets: list[Tensor] = []
            mask = None

            for src_sec in storage.sectors.values():
                if src_sec.schema.top_dim != top_dim:
                    continue
                key = (src_sec.id, codim1_name)
                if key not in storage.relations:
                    continue
                relation = storage.relations[key]
                relation_targets.append(relation.tgt_indices)

            count = cls._accumulate_target_counts(num_target, relation_targets)
            mask = bm.equal(count, bm.ones_like(count))
            binfo[codim1_name] = BoundaryInfo(mask=mask)

        return binfo

    @classmethod
    def infer_top_dim(
        cls,
        storage: MeshBlock,
        top_names: list[str],
        binfo: dict[str, BoundaryInfo]
    ) -> dict[str, BoundaryInfo]:
        """Infer boundary info for top-dimensional entity blocks.

        Boundary rule:
            A top-dimensional entity is on boundary iff it references at least
            one boundary codim-1 entity.
        """
        top_dim = cls._top_dimension(storage)
        if top_dim < 0:
            return {}

        for top_name in top_names:
            num_cell = len(storage.get_sector(top_name).indices)
            cell_mask = bm.zeros((num_cell,), dtype=bm.bool)

            for (src_name, tgt_name), relation in storage.relations.items():
                if src_name != top_name:
                    continue
                if tgt_name not in binfo:
                    continue

                boundary_mask = binfo[tgt_name].mask
                hit = boundary_mask[relation.tgt_indices]

                if relation.src_indices is not None:
                    hit_count = bm.zeros((num_cell,), dtype=bm.int32)
                    hit_count = bm.index_add(
                        hit_count,
                        relation.src_indices,
                        bm.astype(hit, bm.int32),
                    )
                    local_mask = hit_count > 0
                elif len(hit.shape) == 1:
                    local_mask = hit
                else:
                    local_mask = bm.any(hit, axis=1)

                cell_mask = bm.logical_or(cell_mask, local_mask)

            binfo[top_name] = BoundaryInfo(mask=cell_mask)

        return binfo

    @classmethod
    def infer_all(
        cls,
        storage: MeshBlock,
        names: list[str],
        binfo: dict[str, BoundaryInfo]
    ) -> dict[str, BoundaryInfo]:
        """Infer boundary info for all entity blocks."""
        top_dim = cls._top_dimension(storage)
        codim1 = top_dim - 1
        if top_dim < 0:
            return {}

        for name in names:
            sector = storage.get_sector(name)
            if sector.schema.top_dim == codim1:
                raise ValueError(f"{name!r} is codimension-1, use infer_codim1() instead")
            num_entity = len(storage.get_sector(name).indices)
            mask = bm.zeros((num_entity,), dtype=bm.bool)

            for (src_name, tgt_name), relation in storage.relations.items():
                if tgt_name != name:
                    continue
                if storage.get_sector(src_name).schema.top_dim != codim1:
                    continue
                if src_name not in binfo:
                    continue

                parent_index = binfo[src_name].index
                if len(parent_index) == 0:
                    continue

                child_index = bm.reshape(relation.tgt_indices[parent_index], (-1,))
                if len(child_index) == 0:
                    continue

                mask[child_index] = True

            binfo[name] = BoundaryInfo(mask=mask)

        return binfo

    @classmethod
    def infer_entity(
        cls,
        storage: MeshBlock,
        entity_name: str,
        binfo: dict[str, BoundaryInfo] | None = None
    ) -> BoundaryInfo:
        """Infer boundary info for one entity sector by ``entity_name``.

        ``entity_name`` is a sector id, not a Schema family name.  The method
        first identifies related codimension-one sectors from existing
        relation keys, infers their boundary masks, and then propagates the
        mask to the requested sector.
        """
        if binfo is None:
            binfo = {}

        if entity_name not in storage.sectors:
            raise ValueError(f"Entity block {entity_name!r} not found in storage")

        top_dim = storage.get_sector(entity_name).schema.top_dim
        highest_top_dim = cls._top_dimension(storage)

        if top_dim == highest_top_dim - 1:
            cls.infer_codim1(storage, [entity_name], binfo)
            return binfo[entity_name]

        # STEP 1: Find all face schemas that are related to the entity
        faces: list[str] = []

        if top_dim < highest_top_dim - 1:  # lower-dimensional entity
            for src_id, tgt_id in storage.relations:
                src_sector = storage.get_sector(src_id)
                if (
                    src_sector.schema.top_dim == highest_top_dim - 1
                    and tgt_id == entity_name
                ):
                    faces.append(src_id)
        else:  # top-dimensional entity
            for src_id, tgt_id in storage.relations:
                if src_id != entity_name:
                    continue
                tgt_sector = storage.get_sector(tgt_id)
                if tgt_sector.schema.top_dim == highest_top_dim - 1:
                    faces.append(tgt_id)

        # NOTE: Relations from cells to faces are already established,
        # so we can directly infer boundary info for these faces.
        cls.infer_codim1(storage, faces, binfo)

        # STEP 2: Ensure relations from faces to the entity are present
        for face_name in faces:
            from .relation import resolve_relation

            resolve_relation(storage, face_name, entity_name)

        # STEP 3: Infer boundary info for the entity
        cls.infer_all(storage, [entity_name], binfo)

        return binfo[entity_name]
