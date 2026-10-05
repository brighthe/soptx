# 移植自 brighthe/fealpy ``fealpy/mesh/topology/boundary.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""由规范的分区对关系推断边界掩码.

边界查询使用 ``EntitySector.id``, 而不是 Schema 名或解析出的本体字符串. 单根
网格的边界由与余维 1 派生分区的关联计数确定; 多个拓扑维数不同的根并存时, 在引入
带锚点的 :class:`MeshView` 上下文 (P4 阶段) 之前, 全块边界仍以最高根维数为准.
"""

from dataclasses import dataclass
from functools import cached_property

from ...backend import bm
from ...backend import Tensor
from ..storage import MeshBlock

__all__ = ["BoundaryInfo", "BoundaryInferencer"]


@dataclass(frozen=True)
class BoundaryInfo:
    """某个实体分区的边界掩码."""
    mask: Tensor

    @cached_property
    def index(self) -> Tensor:
        """边界实体的编号."""
        return bm.nonzero(self.mask)[0]


class BoundaryInferencer:
    """由规范关系推断全块的边界掩码.

    余维 1 实体恰与一个最高维实体相邻时为边界实体; 更高或更低维实体的边界掩码再经
    已有的正向关系传递得到.
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
        """推断所有余维 1 实体分区的边界信息.

        判定规则: 统计相邻的最高维实体个数, 余维 1 实体当且仅当该数为 1 时位于边界.

        Raises
        ------
        ValueError
            给出的分区不是余维 1.
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
        """推断最高维实体分区的边界信息.

        判定规则: 最高维实体当且仅当至少引用一个边界上的余维 1 实体时位于边界.
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
        """推断低维实体分区的边界信息: 被边界上的余维 1 实体引用者位于边界.

        Raises
        ------
        ValueError
            给出的分区是余维 1, 应改用 ``infer_codim1``.
        """
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
        """推断 id 为 ``entity_name`` 的实体分区的边界信息.

        ``entity_name`` 是分区 id, 不是 Schema 族名. 先从已有关系的键中找出相关的
        余维 1 分区并推断其边界掩码, 再把掩码传递到所求分区.

        Raises
        ------
        ValueError
            分区不存在.
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

        # 第 1 步: 找出与该实体相关的所有面分区
        faces: list[str] = []

        if top_dim < highest_top_dim - 1:  # 低维实体
            for src_id, tgt_id in storage.relations:
                src_sector = storage.get_sector(src_id)
                if (
                    src_sector.schema.top_dim == highest_top_dim - 1
                    and tgt_id == entity_name
                ):
                    faces.append(src_id)
        else:  # 最高维实体
            for src_id, tgt_id in storage.relations:
                if src_id != entity_name:
                    continue
                tgt_sector = storage.get_sector(tgt_id)
                if tgt_sector.schema.top_dim == highest_top_dim - 1:
                    faces.append(tgt_id)

        # NOTE: 单元到面的关系已经建立, 可以直接推断这些面的边界信息.
        cls.infer_codim1(storage, faces, binfo)

        # 第 2 步: 确保面到该实体的关系存在
        for face_name in faces:
            from .relation import resolve_relation

            resolve_relation(storage, face_name, entity_name)

        # 第 3 步: 推断该实体的边界信息
        cls.infer_all(storage, [entity_name], binfo)

        return binfo[entity_name]
