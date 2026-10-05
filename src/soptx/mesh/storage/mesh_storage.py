# 移植自 brighthe/fealpy ``fealpy/mesh/storage/mesh_storage.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""网格存储: 实体分区、实体子集、网格块与 Schema 计算上下文."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ...backend import bm, Tensor
from .relation import Relation

if TYPE_CHECKING:
    from ..schema.entity_schema import EntitySchema

__all__ = ["EntityContext", "EntitySector", "EntitySet", "MeshBlock"]


@dataclass(slots=True)
class EntitySector:
    """一批同类实体, 具有独立的语义身份 (实体分区).

    ``id`` 在所属 ``MeshBlock`` 内唯一; ``schema`` 是分区内所有实体共享的具体、
    不可变的 :class:`EntitySchema` 值. 因此多个分区可以使用同一个 Schema 而仍是
    不同的语义域, 例如壳体与实体的派生面.

    定长分区 (``indptr is None``) 的 ``indices`` 为二维张量, 末轴长度等于
    ``schema.number_of_nodes()``, 元素为网格块全局的节点编号. 变长分区以
    ``(indices, indptr)`` 压缩存储. ``source_cell_sector_id`` 记录派生分区来自
    哪个根单元分区, 直接给出的分区为 None.

    Raises
    ------
    TypeError
        ``id`` 或 ``schema`` 类型不对, 或变长分区的索引不是整数类型.
    ValueError
        定长分区的连接宽度与 Schema 的局部节点数不符, 或变长分区的
        ``indptr`` 不合法.
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
    """某个已有 ``EntitySector`` 的具名子集.

    ``sector_id`` 须指向所属 ``MeshBlock`` 中的分区, ``indices`` 为该分区编号下的
    一维实体编号. ``EntitySet`` 从不建立新的实体编号空间.

    Raises
    ------
    TypeError
        ``id`` 或 ``sector_id`` 不是非空字符串.
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
    """共享节点 (坐标) 编号域的网格块, 含一个规范节点分区.

    每个构造好的 ``MeshBlock`` 都拥有规范分区 ``"node"``, 其索引以 ``(NN, 1)``
    依次枚举 ``positions``. 普通的 :meth:`add_sector` 调用不得替换或改写该分区.

    ``id`` 在所属聚合中保持不变, 单块调用方默认为 ``"block"``.
    ``root_cell_sector_ids`` 列出可作为视图锚点的分区 id; 规范节点分区不是根单元
    分区.

    Raises
    ------
    TypeError
        ``id`` 不是非空字符串.
    ValueError
        给出的节点分区与规范节点域不符.
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
        """向网格块加入一个同类实体分区.

        分区引用的节点编号须落在本块的节点域内. ``root=True`` 时还把 ``sec.id``
        记为根单元分区.

        Raises
        ------
        ValueError
            ``sec.id`` 已存在, 或节点编号越界.
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
        """受控的根连接版本号.

        加入根分区或替换其连接关系时版本号改变, 只替换坐标时不变. 直接修改分区的
        连接数组不在受控修改的约定之内, 本计数器无法察觉.
        """
        return self._topology_revision

    def get_sector(self, sector_id: str, /) -> EntitySector:
        """返回 id 为 ``sector_id`` 的分区.

        Raises
        ------
        KeyError
            不存在该分区.
        """
        return self.sectors[sector_id]

    def has_sector(self, sector_id: str, /) -> bool:
        """是否存在 id 为 ``sector_id`` 的分区."""
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
        """校验后向网格块加入一个 ``EntitySet``.

        Raises
        ------
        ValueError
            ``entity_set.id`` 已存在, 或其索引不是所引用分区的合法一维实体编号.
        KeyError
            ``entity_set.sector_id`` 未知.
        TypeError
            索引不是整数类型.
        """
        if entity_set.id in self.entity_sets:
            raise ValueError(
                f"duplicate EntitySet id {entity_set.id!r} in this MeshBlock"
            )
        self._validate_entity_set(entity_set)
        self.entity_sets[entity_set.id] = entity_set

    def add_relation(self, relation: Relation) -> None:
        """在已有分区之间加入一条规范的有向关系.

        有序对 ``(src_sector_id, tgt_sector_id)`` 即关系的身份; 实体化方式的不同
        不得为同一对产生第二条记录.

        Raises
        ------
        KeyError
            所引用的分区不存在.
        ValueError
            该有序对已有规范关系.
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
        """在节点数不变的前提下原子地替换坐标.

        这是最小的受控坐标更新路径: 保持规范节点分区的编号, 保留所有分区、实体子集
        与关系, 并使派生缓存失效.

        Raises
        ------
        TypeError
            ``positions`` 不是张量.
        ValueError
            ``positions`` 不是二维.
        NotImplementedError
            节点数会改变. 改变节点域大小需要经批准的协议, 不会被静默支持.
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
        """原子地替换单根拓扑与坐标.

        先在临时网格块上构造新的根分区, 期间不改动本块的任何状态; 成功后用临时块的
        规范节点域、分区与关系替换本块的拓扑. 所引用分区已不存在的实体子集被丢弃.
        节点与单元属性不会自动映射到新编号, 由调用方显式处理.

        Raises
        ------
        TypeError
            ``positions`` 或 ``root_indices`` 不合法.
        KeyError
            ``root_sector_id`` 未知.
        ValueError
            ``root_sector_id`` 不是根分区, 连接宽度不对, 或节点编号越出新的节点域.
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
        """原子地替换一个根分区, 其他根分区保持不变.

        这是多根网格的加密路径: 先在临时网格块上校验并构造新的根, 再只替换目标根及
        其派生分区与关系, 属于其他根的分区不变. 节点属性与目标根的 EntitySet 不会
        自动映射, 而是被丢弃; 节点的 EntitySet 保留原编号, 因为旧节点仍是新节点域的
        前缀.

        Raises
        ------
        TypeError
            ``positions`` 或 ``root_indices`` 不合法.
        KeyError
            ``root_sector_id`` 未知.
        ValueError
            ``root_sector_id`` 不是根分区, 连接宽度不对, 或节点编号越界.
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
        """拒绝改变节点数的插入, 在批准相应协议之前一律抛 ``NotImplementedError``."""
        raise NotImplementedError(
            "node insertion is not supported until an approved canonical-node "
            "mutation protocol exists"
        )

    def remove_nodes(self, indices: Tensor) -> None:
        """拒绝改变节点数的删除, 在批准相应协议之前一律抛 ``NotImplementedError``."""
        raise NotImplementedError(
            "node removal is not supported until an approved canonical-node "
            "mutation protocol exists"
        )


@dataclass(slots=True, frozen=True)
class EntityContext:
    """传给 Schema 计算方法的轻量上下文: 网格块与实体分区."""

    block: MeshBlock
    sector: EntitySector
