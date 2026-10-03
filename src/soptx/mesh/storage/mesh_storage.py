# 移植自 brighthe/fealpy ``fealpy/mesh/storage/mesh_storage.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ...backend import bm, Tensor
from .relation import Relation

if TYPE_CHECKING:
    from ..schema.entity_schema import EntitySchema

__all__ = ["EntityContext", "EntitySector", "EntitySet", "MeshBlock"]


@dataclass(slots=True)
class EntitySector:
    """A homogeneous entity batch with independent semantic identity.

    ``id`` is unique within its ``MeshBlock``, while ``schema`` is the concrete
    immutable :class:`EntitySchema` value shared by every entity in the sector.
    Several sectors may therefore use the same Schema while remaining distinct
    semantic domains, such as a shell and a derived solid face.

    For fixed-cardinality sectors (``indptr is None``), ``indices`` is a rank-2
    tensor whose last dimension equals ``schema.number_of_nodes()`` and whose
    values are block-global node ids.  ``source_cell_sector_id`` records the
    root-cell provenance of derived sectors and is ``None`` for direct sectors.

    Raises:
        TypeError: If ``id`` or ``schema`` has an invalid type.
        ValueError: If fixed-cardinality connectivity width does not match the
            Schema's complete local-node count.
    """

    id: str
    schema: "EntitySchema"
    indices: Tensor
    indptr: Tensor | None = None
    source_cell_sector_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ..schema.entity_schema import EntitySchema

        if type(self.id) is not str or not self.id:
            raise TypeError("EntitySector.id must be a non-empty string")
        if not isinstance(self.schema, EntitySchema):
            raise TypeError(
                "EntitySector.schema must be an EntitySchema instance, "
                f"got {type(self.schema).__name__}"
            )
        if self.indptr is None:
            if len(self.indices.shape) != 2:
                raise ValueError(
                    "EntitySector.indices must be rank-2 when indptr is None"
                )
            expected_width = self.schema.number_of_nodes()
            if int(self.indices.shape[-1]) != expected_width:
                raise ValueError(
                    f"EntitySector {self.id!r} connectivity width must equal "
                    f"schema.number_of_nodes() == {expected_width}, "
                    f"got {self.indices.shape[-1]}"
                )
        else:
            if len(self.indices.shape) != 1:
                raise ValueError(
                    "EntitySector.indices must be rank-1 when indptr is supplied"
                )
            if len(self.indptr.shape) != 1:
                raise ValueError("EntitySector.indptr must be rank-1")
            if self.indices.dtype not in (bm.int32, bm.int64):
                raise TypeError("EntitySector.indices must use an integer dtype")
            if self.indptr.dtype not in (bm.int32, bm.int64):
                raise TypeError("EntitySector.indptr must use an integer dtype")
            if int(self.indptr.shape[0]) == 0:
                raise ValueError("EntitySector.indptr must contain at least one entry")
            if bool(self.indptr[0] != 0):
                raise ValueError("EntitySector.indptr must start at zero")
            if bool(self.indptr[-1] != int(self.indices.shape[0])):
                raise ValueError(
                    "EntitySector.indptr must end at the connectivity length"
                )
            counts = self.indptr[1:] - self.indptr[:-1]
            if bool(bm.any(counts < 0)):
                raise ValueError("EntitySector.indptr must be nondecreasing")
            self.schema.validate_connectivity_counts(counts)


@dataclass(slots=True)
class EntitySet:
    """A named subset of one existing ``EntitySector``.

    ``sector_id`` must reference a sector in the owning ``MeshBlock``, and
    ``indices`` contains one-dimensional entity indices within that sector's
    numbering.  An ``EntitySet`` never creates a new entity numbering space.

    Raises:
        TypeError: If ``id`` or ``sector_id`` is not a non-empty string.
    """

    id: str
    sector_id: str
    indices: Tensor

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id:
            raise TypeError("EntitySet.id must be a non-empty string")
        if type(self.sector_id) is not str or not self.sector_id:
            raise TypeError("EntitySet.sector_id must be a non-empty string")

@dataclass(slots=True)
class MeshBlock:
    """Shared node/position numbering domain with one canonical node sector.

    Every constructed ``MeshBlock`` owns a canonical sector ``"node"`` whose
    indices enumerate ``positions`` as ``(NN, 1)``.  Ordinary
    :meth:`add_sector` calls must not replace or rewrite that sector.

    ``id`` is stable within its owning aggregate and defaults to ``"block"`` for
    single-block callers.  ``root_cell_sector_ids`` lists sector ids that can be
    used as view anchors; the canonical node sector is not a root cell sector.
    """

    positions: Tensor
    id: str = "block"
    sectors: dict[str, EntitySector] = field(default_factory=dict)
    entity_sets: dict[str, EntitySet] = field(default_factory=dict)
    relations: dict[tuple[str, str], Relation] = field(default_factory=dict)
    root_cell_sector_ids: list[str] = field(default_factory=list)
    _cache_boundary_info: dict[str, Any] | None = None
    _topology_revision: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        from ..schema import NodeSchema

        if type(self.id) is not str or not self.id:
            raise TypeError("MeshBlock.id must be a non-empty string")

        node_count = int(self.positions.shape[0])
        device = bm.get_device(self.positions)
        canonical = bm.arange(node_count, dtype=bm.int64, device=device)
        canonical = bm.reshape(canonical, (-1, 1))

        node = self.sectors.get("node")
        if node is None:
            self.sectors["node"] = EntitySector(
                id="node",
                schema=NodeSchema(),
                indices=canonical,
            )
            return

        if node.id != "node" or node.schema != NodeSchema():
            raise ValueError(
                "the canonical node sector must have id 'node' and NodeSchema"
            )
        if node.indices.dtype != bm.int64:
            raise ValueError(
                "the canonical node sector must use the block-global int64 index dtype"
            )
        if node.indices.shape != canonical.shape or not bm.all(
            node.indices == canonical
        ):
            raise ValueError(
                "the supplied node sector does not match the canonical node domain"
            )

        for entity_set in self.entity_sets.values():
            self._validate_entity_set(entity_set)

    def add_sector(self, sec: EntitySector, *, root: bool = False) -> None:
        """Add a homogeneous sector to this block.

        The sector's node ids must lie in the block node domain.  ``root=True``
        additionally records ``sec.id`` as a root cell sector id.

        Raises:
            ValueError: If ``sec.id`` already exists or its node ids are out of
                range.
        """
        if sec.id in self.sectors:
            raise ValueError(
                f"duplicate EntitySector id {sec.id!r} in this MeshBlock"
            )
        node_count = int(self.positions.shape[0])
        if bm.any(sec.indices < 0) or bm.any(sec.indices >= node_count):
            raise ValueError(
                f"EntitySector {sec.id!r} references node ids outside the "
                f"block node domain [0, {node_count})"
            )
        self.sectors[sec.id] = sec
        if root and sec.id not in self.root_cell_sector_ids:
            self.root_cell_sector_ids.append(sec.id)
            self._topology_revision += 1

    @property
    def topology_revision(self) -> int:
        """Return the controlled root-connectivity revision.

        The revision changes when a root sector is added or its connectivity
        is replaced.  Coordinate-only replacement preserves the revision.
        Direct mutation of sector connectivity is outside the controlled
        mutation contract and cannot be detected by this counter.
        """
        return self._topology_revision

    def get_sector(self, sector_id: str, /) -> EntitySector:
        """Return the sector with ``sector_id``.

        Raises:
            KeyError: If no such sector exists.
        """
        return self.sectors[sector_id]

    def has_sector(self, sector_id: str, /) -> bool:
        """Return whether a sector with ``sector_id`` exists."""
        return sector_id in self.sectors

    def _validate_entity_set(self, entity_set: EntitySet) -> None:
        sector = self.sectors.get(entity_set.sector_id)
        if sector is None:
            raise KeyError(
                f"EntitySet {entity_set.id!r} references unknown sector "
                f"{entity_set.sector_id!r}"
            )
        if entity_set.indices.ndim != 1:
            raise ValueError(
                f"EntitySet {entity_set.id!r} indices must be one-dimensional"
            )
        if entity_set.indices.dtype not in (bm.int32, bm.int64):
            raise TypeError(
                f"EntitySet {entity_set.id!r} indices must use an integer dtype"
            )
        entity_count = int(sector.indices.shape[0])
        if not bm.all(entity_set.indices >= 0) or not bm.all(
            entity_set.indices < entity_count
        ):
            raise ValueError(
                f"EntitySet {entity_set.id!r} indices are out of range for "
                f"sector {entity_set.sector_id!r}"
            )

    def add_entity_set(self, entity_set: EntitySet) -> None:
        """Add one validated ``EntitySet`` to this block.

        Raises:
            ValueError: If ``entity_set.id`` already exists.
            KeyError: If ``entity_set.sector_id`` is unknown.
            ValueError: If ``entity_set.indices`` is not a valid one-dimensional
                entity-index tensor for its referenced sector.
        """
        if entity_set.id in self.entity_sets:
            raise ValueError(
                f"duplicate EntitySet id {entity_set.id!r} in this MeshBlock"
            )
        self._validate_entity_set(entity_set)
        self.entity_sets[entity_set.id] = entity_set

    def add_relation(self, relation: Relation) -> None:
        """Add one canonical directed relation between existing sectors.

        The ordered pair ``(src_sector_id, tgt_sector_id)`` is the relation's
        identity.  Materialization differences must not create a second entry
        for the same pair.

        Raises:
            KeyError: If either referenced sector does not exist.
            ValueError: If the ordered pair already has a canonical relation.
        """
        if relation.src_sector_id not in self.sectors:
            raise KeyError(
                f"relation source sector {relation.src_sector_id!r} does not exist"
            )
        if relation.tgt_sector_id not in self.sectors:
            raise KeyError(
                f"relation target sector {relation.tgt_sector_id!r} does not exist"
            )
        key = (relation.src_sector_id, relation.tgt_sector_id)
        if key in self.relations:
            raise ValueError(f"duplicate canonical relation for pair {key}")
        self.relations[key] = relation

    def replace_positions(self, positions: Tensor) -> None:
        """Atomically replace coordinates without changing the node count.

        This is the minimal controlled position-update path: it keeps the
        canonical node sector's numbering intact, preserves all sectors,
        entity sets, and relations, and invalidates derived caches.

        Raises:
            TypeError: If ``positions`` is not a tensor.
            ValueError: If ``positions`` is not rank-2.
            NotImplementedError: If the node count would change.  Size-changing
                node-domain mutation requires an approved protocol and is not
                silently supported.
        """
        if not isinstance(positions, Tensor):
            raise TypeError("positions must be a Tensor")
        if len(positions.shape) != 2:
            raise ValueError("positions must be rank-2")

        old_count = int(self.positions.shape[0])
        new_count = int(positions.shape[0])
        if new_count != old_count:
            raise NotImplementedError(
                "size-changing position updates are not supported; "
                "construct a new MeshBlock or use an approved node-domain "
                "mutation protocol"
            )

        self.positions = positions
        node = self.sectors["node"]
        node.indices = bm.arange(
            new_count,
            dtype=bm.int64,
            device=bm.get_device(positions),
        ).reshape((-1, 1))
        self._cache_boundary_info = None

    def replace_topology(
        self,
        positions: Tensor,
        root_sector_id: str,
        root_indices: Tensor,
    ) -> None:
        """Atomically replace single-root topology and coordinates.

        The new root sector is constructed on a temporary block before any
        state on this block is changed.  On success the temporary block's
        canonical node domain, sectors and relations replace this block's
        topology.  Entity sets whose sector no longer exists are dropped.
        Node and cell attributes are not automatically remapped to the new
        numbering; callers are responsible for an explicit attribute mapping.

        Raises:
            TypeError: If ``positions`` or ``root_indices`` is invalid.
            KeyError: If ``root_sector_id`` is unknown.
            ValueError: If ``root_sector_id`` is not a root, connectivity width
                is wrong, or node ids fall outside the new node domain.
        """
        if not isinstance(positions, Tensor) or len(positions.shape) != 2:
            raise TypeError("positions must be a rank-2 Tensor")
        if root_sector_id not in self.sectors:
            raise KeyError(f"root sector {root_sector_id!r} not found")
        if root_sector_id not in self.root_cell_sector_ids:
            raise ValueError(
                f"sector {root_sector_id!r} is not a root cell sector"
            )
        if not isinstance(root_indices, Tensor) or len(root_indices.shape) != 2:
            raise TypeError("root_indices must be a rank-2 Tensor")

        node_count = int(positions.shape[0])
        if bm.any(root_indices < 0) or bm.any(root_indices >= node_count):
            raise ValueError(
                f"root sector {root_sector_id!r} references nodes outside "
                f"[0, {node_count})"
            )

        root_schema = self.sectors[root_sector_id].schema
        if int(root_indices.shape[-1]) != root_schema.number_of_nodes():
            raise ValueError(
                f"root connectivity width {root_indices.shape[-1]} does not "
                f"match schema node count {root_schema.number_of_nodes()}"
            )

        from ..topology.builder import TopologyBuilder

        temporary = MeshBlock(id=self.id, positions=positions)
        temporary.add_sector(
            EntitySector(
                id=root_sector_id,
                schema=root_schema,
                indices=root_indices,
            ),
            root=True,
        )
        TopologyBuilder.construct(temporary)

        self.positions = positions
        self.sectors = temporary.sectors
        self.relations = temporary.relations
        self.root_cell_sector_ids = temporary.root_cell_sector_ids
        self._cache_boundary_info = None
        self._topology_revision += 1

        self.entity_sets = {
            set_id: entity_set
            for set_id, entity_set in self.entity_sets.items()
            if entity_set.sector_id in self.sectors
        }

    def replace_root_topology(
        self,
        positions: Tensor,
        root_sector_id: str,
        root_indices: Tensor,
    ) -> None:
        """Atomically replace one root sector while preserving other roots.

        This is the multi-root refinement path.  It validates and builds the
        new root on a temporary block first, then replaces only the target root
        and its derived sectors/relations.  Sectors belonging to other roots
        remain unchanged.  Node attributes and target-root EntitySets are not
        automatically remapped and are dropped; node EntitySets keep their old
        indices because old nodes remain the prefix of the new node domain.
        """
        if not isinstance(positions, Tensor) or len(positions.shape) != 2:
            raise TypeError("positions must be a rank-2 Tensor")
        if root_sector_id not in self.sectors:
            raise KeyError(f"root sector {root_sector_id!r} not found")
        if root_sector_id not in self.root_cell_sector_ids:
            raise ValueError(
                f"sector {root_sector_id!r} is not a root cell sector"
            )
        if not isinstance(root_indices, Tensor) or len(root_indices.shape) != 2:
            raise TypeError("root_indices must be a rank-2 Tensor")

        node_count = int(positions.shape[0])
        if bm.any(root_indices < 0) or bm.any(root_indices >= node_count):
            raise ValueError(
                f"root sector {root_sector_id!r} references nodes outside "
                f"[0, {node_count})"
            )

        root_schema = self.sectors[root_sector_id].schema
        if int(root_indices.shape[-1]) != root_schema.number_of_nodes():
            raise ValueError(
                f"root connectivity width {root_indices.shape[-1]} does not "
                f"match schema node count {root_schema.number_of_nodes()}"
            )

        from ..topology.builder import TopologyBuilder

        temporary = MeshBlock(id=self.id, positions=positions)
        temporary.add_sector(
            EntitySector(
                id=root_sector_id,
                schema=root_schema,
                indices=root_indices,
            ),
            root=True,
        )
        TopologyBuilder.construct(temporary)

        target_ids = {
            root_sector_id,
            *(
                sector.id
                for sector in self.sectors.values()
                if sector.source_cell_sector_id == root_sector_id
            ),
        }

        new_sectors = {
            sector_id: sector
            for sector_id, sector in self.sectors.items()
            if sector_id not in target_ids
        }
        new_relations = {
            key: relation
            for key, relation in self.relations.items()
            if key[0] not in target_ids and key[1] not in target_ids
        }

        new_sectors["node"] = temporary.sectors["node"]
        new_sectors.update(temporary.sectors)
        new_relations.update(temporary.relations)

        self.positions = positions
        self.sectors = new_sectors
        self.relations = new_relations
        self._cache_boundary_info = None
        self._topology_revision += 1

        self.entity_sets = {
            set_id: entity_set
            for set_id, entity_set in self.entity_sets.items()
            if entity_set.sector_id in self.sectors
        }

    def add_nodes(self, positions: Tensor) -> None:
        """Reject size-changing node insertion until a protocol is approved."""
        raise NotImplementedError(
            "node insertion is not supported until an approved canonical-node "
            "mutation protocol exists"
        )

    def remove_nodes(self, indices: Tensor) -> None:
        """Reject size-changing node removal until a protocol is approved."""
        raise NotImplementedError(
            "node removal is not supported until an approved canonical-node "
            "mutation protocol exists"
        )


@dataclass(slots=True, frozen=True)
class EntityContext:
    """Lightweight block/sector context passed to Schema computation methods."""

    block: MeshBlock
    sector: EntitySector
