# 移植自 brighthe/fealpy ``fealpy/mesh/topology/builder.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""按根分区构造派生子分区与规范关联关系.

本模块实现网格四层模型 (源自 FEALPy) 中的拓扑部分. 拓扑构造以一个
:class:`EntitySector` 为根; 从不合并不同的根语义分区, 也从不创建、裁剪、合并或
重编号 :class:`MeshBlock` 所拥有的规范 ``"node"`` 分区.

对协调的 Lagrange 网格, 每个局部子实体用两级身份:

``(子 Schema 的 Python 类型, 规范顶点元组)``
    标识一个拓扑实体;

``(具体 Schema 的 id, 规范完整节点元组)``
    校验该拓扑身份下的协调一致性.

实现上用 :func:`extract_local_entity_occurrences` 提取出现, 用
:func:`canonicalize_local_entity_occurrences` 规范化定向, 并同时存储正反两个方向的
规范 :class:`EntityRelation`.
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
    """对多个二维张量的行逐行规范化后统一去重.

    先对每行排序, 即把行视为无序集合.
    """
    if not arrays:
        raise ValueError("at least one array is required")

    for arr in arrays:
        if len(arr.shape) != 2:
            raise ValueError("only 2D tensors are supported")

    total = bm.concat(arrays, axis=0)  # 形状 (total_rows, ncols)
    canonical_total = bm.sort(total, axis=1)

    indices = bm.lexsort(tuple(reversed(canonical_total.T)), axis=0)  # 排序后 <-> 原始顺序
    sorted_canonical = canonical_total[indices]

    diff_flag = bm.any(sorted_canonical[1:] != sorted_canonical[:-1], axis=1)
    true = bm.ones((1,), dtype=bm.bool, device=diff_flag.device)
    diff_flag = bm.concat([true, diff_flag])

    # 取各组代表行, 保持其原始顺序
    unique = total[indices[diff_flag]]
    sorted_to_unique = bm.cumulative_sum(diff_flag, axis=0) - 1  # 排序后 -> 去重后

    original_to_sorted = bm.empty_like(indices)
    original_to_sorted[indices] = bm.arange(
        len(indices),
        dtype=original_to_sorted.dtype,
        device=original_to_sorted.device,
    )
    total_to_unique = sorted_to_unique[original_to_sorted]  # 原始顺序 -> 去重后

    array_indptr = [0]
    for arr in arrays:
        array_indptr.append(array_indptr[-1] + len(arr))

    arr_to_unique = tuple(
        total_to_unique[array_indptr[i]:array_indptr[i + 1]]
        for i in range(len(arrays))
    )

    return unique, arr_to_unique


def _unique_ordered_rows(rows: Tensor) -> tuple[Tensor, Tensor]:
    """按字典序对二维张量的行精确去重, 并给出输入行到去重结果的映射."""
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
    """按局部面编号收集所有单元的局部面, 形状 ``(NC * NFC_local, NVF)``, 未去重."""
    total_face = cell[:, local_face]
    NFC = len(local_face[0])
    return bm.reshape(total_face, (-1, NFC))


def _lower_entities(
    schema: type["EntitySchema"],
    excluded: set[str] | None = None,
) -> dict[str, list[list[int]]]:
    """返回指向低维 Schema 的局部面 (OFace) 条目."""
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
    """返回限定于根的派生分区 id.

    分区 id 等于其 Schema 名的单一经典根沿用子 Schema 的短名, 以保持历史上的单类型
    布局; 其他根使用 ``f"{root.id}_{child_schema.name}"``, 使不同的语义分区不会共用
    派生 id.
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
    """返回某个根的子 Schema 所对应的派生分区 id."""
    if use_legacy_name:
        return child_schema.name
    return f"{root_id}_{child_schema.name}"


def _ordered_rows_to_existing(
    existing: Tensor,
    new: Tensor,
    target_id: str,
) -> Tensor:
    """把 ``new`` 中的规范行精确映射到已存在的规范分区.

    ``existing`` 须已是目标分区规范、去重、按字典序排列的行集. 本函数建立合并后的
    有序去重索引, 校验前 ``len(existing)`` 行仍映射到自身, 并返回 ``new`` 各行的
    目标编号.
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
    """在一个根分区内校验双键协调规则.

    拓扑键为 ``(type(子 Schema), 规范顶点元组)``. 首次出现时记录具体 Schema 的 id
    与规范完整节点元组; 之后的出现须有相同的具体 Schema id 与相同的规范完整节点,
    否则网格不协调.
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
    """一层低维实体的构造结果: 实体类型名、去重后的实体及各输入单元到它的映射."""
    face_type: str
    face: Tensor
    cell_to_face: tuple[Tensor, ...]


class TopologyBuilder:
    """按根分区构造派生子分区与规范关联关系.

    ``construct`` 是稳定的入口, 对一个或多个根单元分区分别处理:

    - 提取完整的子实体出现;
    - 规范化允许的定向;
    - 校验双键协调规则;
    - 按规范完整节点行精确去重;
    - 绑定带 ``source_cell_sector_id`` 的派生分区;
    - 建立根/派生/节点之间的关系, 并实体化反向关系.

    本类不从分区 id 解析出单元/面/边的角色; 角色的解释属于 :class:`MeshView`,
    依据其根锚点以及存储的来源与关系.
    """

    @classmethod
    def construct_lower_dims(
        cls,
        cells: Iterable[Tensor],
        local_face_dicts: Iterable[dict[str, list[list[int]]]],
    ) -> Iterator[ConstructResult]:
        """构造低一维的实体.

        Parameters
        ----------
        cells : iterable of Tensor
            单元序列, 每个形状为 ``(NC, NVF)``.
        local_face_dicts : iterable of dict
            局部面字典序列: 键用于标记面的类型, 值为局部面的顶点编号.

        Yields
        ------
        ConstructResult
            面的类型名、去重后的面数组, 以及各输入单元到面的映射.
        """
        # NOTE: 结构为 {face_kind: ([total_face,], [NFC,])}
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
        """由 ``blocks`` 构造一层局部面 (OFace) 实体, 返回涉及的分区."""
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
        """按根单元分区构造低维分区, 原地修改 ``storage``.

        各根分区分别处理: 从其参数化的局部实体组提取出现, 规范化定向, 去重, 并绑定到
        ``source_cell_sector_id`` 为该根 id 的派生分区; 为派生分区建立嵌套的规范关系,
        并实体化反向关系. 不同的根语义分区从不合并.

        本方法不是幂等的: 它假定所需的派生分区与关系尚不存在, 重建路径须先清除过时的
        分区与关系.

        Parameters
        ----------
        storage : MeshBlock
            要修改的网格存储对象.
        src_name : str, optional
            起始的源分区名; 为 None (默认) 时处理所有根分区.
        exclude : list of str, optional
            不构造的低维形状名.

        Returns
        -------
        list of EntitySector
            新构造的低维分区.
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
        """为一个多边形根分区构造循环的边与不等长的关联关系."""
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
        """为一个根分区构造全部低维分区.

        出现按具体子 Schema 分组, 规范化定向, 校验协调一致性, 并按规范完整节点行精确
        去重. 得到的每个分区记录其 ``source_cell_sector_id``, 并直接获得根→派生与
        根→节点的关系.
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
        """在同一根派生出的分区之间建立嵌套关系.

        派生分区按维数从高到低处理: 对每个源分区提取其子实体出现, 匹配到已建好的、
        限定于该根的目标分区. 由此得到面→边、边→节点等低维关系链, 不跨根合并分区.
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
        """为每条已存储的关系实体化其规范反向关系.

        只在反向有序对不存在时加入逆关系. 由于 :class:`MeshBlock` 规定每个有序对只有
        一条关系, 结果是确定的, 并保持转置与关联的一致性.
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
        """按需在已建好的低维实体之间建立关系."""
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
    """把源分区连接到已存在的目标分区, 建立 ``src -> tgt`` 关系."""
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
        """为已存在的目标分区建立 ``src -> tgt`` 映射, 原地修改 ``storage``.

        只由源分区构造一层临时目标实体, 把临时目标实体匹配回已存在的目标分区, 再把
        临时关系重映射到存储中的编号.

        Parameters
        ----------
        storage : MeshBlock
            要修改的网格存储对象.
        src_name : str
            源分区名.
        tgt_name : str
            目标分区名.

        Returns
        -------
        Relation
            构造出的 ``src -> tgt`` 关系.

        Raises
        ------
        ValueError
            所需的目标实体不在已存在的目标分区中.
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
    """经中间维数的关系复合, 推断高维分区到低维分区的关系."""
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

        for mid_dim in range(src_dim - 1, dst_dim, -1): # 取值于 [src_dim-1, dst_dim+1]
            # 1) 取得所有 `src -> parent` 关系
            src_to_mids: list[tuple[str, Relation]] = []

            if mid_dim == src_dim - 1: # 最高一层按 schema 取直接子实体
                src_to_mids = list(cls._iter_adjacent_children(storage, src_name))
            else: # 其余按维数选取
                for (rel_src, rel_tgt), relation in storage.relations.items():
                    if rel_src != src_name:
                        continue
                    if cls._dim(storage, rel_tgt) != mid_dim:
                        continue
                    src_to_mids.append((rel_tgt, relation))

            if not src_to_mids:
                raise ValueError(f"cannot infer relation from {src_name!r} to {dst_name!r}")

            # 2) 取得所有 `mid -> dst` 关系, 与 `src -> mid` 复合,
            #   得到 `src -> dst` 的候选.
            candidates_of_dst: dict[str, list[Tensor]] = {}
            for mid_name, src_to_mid in src_to_mids:
                for child_name, mid_to_dst in cls._iter_adjacent_children(storage, mid_name):
                    if (src_name, child_name) in storage.relations:
                        composed = storage.relations[(src_name, child_name)].tgt_indices
                    else:
                        composed = cls._compose_homogeneous(src_to_mid, mid_to_dst)
                    candidates_of_dst.setdefault(child_name, []).append(composed)

            for child_name, candidates in candidates_of_dst.items():
                merged = cls._merge_candidates(candidates, src_name, child_name, mid_dim)
                storage.add_relation(Relation(
                    src_sector_id=src_name,
                    tgt_sector_id=child_name,
                    tgt_indices=merged,
                ))

        if (src_name, dst_name) not in storage.relations:
            raise ValueError(f"cannot infer relation from {src_name!r} to {dst_name!r}")

    @classmethod
    def infer(cls, storage: MeshBlock, src_name: str, dst_name: str) -> None:
        """推断并登记 ``src_name -> dst_name`` 的关系.

        Raises
        ------
        ValueError
            源分区维数不高于目标分区, 或无法推断.
        """
        src_dim = cls._dim(storage, src_name)
        dst_dim = cls._dim(storage, dst_name)

        if src_dim <= dst_dim:
            raise ValueError(f"expect src dimension > dst dimension, got {src_name!r} -> {dst_name!r}")

        cls._infer_from(storage, src_name, dst_name)
