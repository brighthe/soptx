# 移植自 brighthe/fealpy ``fealpy/mesh/topology/relation.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""按显式的分区身份解析规范有向关系.

解析关系从不查看 Schema 名, 也不解析本体字符串, 只用所属 :class:`MeshBlock` 中
:class:`EntitySector` 的 id. 规范关系的身份是有序对 ``(src_sector_id,
tgt_sector_id)``.
"""

from __future__ import annotations

from ..storage import MeshBlock, Relation

__all__ = ["resolve_relation"]


def resolve_relation(
    block: MeshBlock,
    src_sector_id: str,
    tgt_sector_id: str,
) -> Relation:
    """解析从 ``src_sector_id`` 到 ``tgt_sector_id`` 的规范关系.

    正向关系不存在时, 若已实体化反向关系, 则由其构造规范逆关系并登记; 否则在源
    分区拓扑维数高于目标时, 退而调用拓扑推断补建.

    Raises
    ------
    KeyError
        任一分区 id 未知.
    ValueError
        无法解析或推断出该关系.
    """
    src_sector = block.get_sector(src_sector_id)
    tgt_sector = block.get_sector(tgt_sector_id)

    relation = block.relations.get((src_sector_id, tgt_sector_id))
    if relation is not None:
        return relation

    reverse_key = (tgt_sector_id, src_sector_id)
    if reverse_key in block.relations:
        inverse = block.relations[reverse_key].inverse()
        block.add_relation(inverse)
        return inverse

    if src_sector.schema.top_dim > tgt_sector.schema.top_dim:
        from .builder import TopRelationInferer

        try:
            TopRelationInferer.infer(block, src_sector_id, tgt_sector_id)
        except ValueError:
            pass
        else:
            relation = block.relations.get((src_sector_id, tgt_sector_id))
            if relation is not None:
                return relation

    raise ValueError(
        f"relation from {src_sector_id!r} to {tgt_sector_id!r} not found"
    )
