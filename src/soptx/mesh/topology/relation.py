# 移植自 brighthe/fealpy ``fealpy/mesh/topology/relation.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Resolve canonical directed relations by explicit sector identity.

Relation resolution never consults a Schema name or parses an ontology string;
it uses :class:`EntitySector` ids from the owning :class:`MeshBlock`.  The
canonical relation identity is the ordered pair ``(src_sector_id,
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
    """Resolve the canonical relation from ``src_sector_id`` to ``tgt_sector_id``.

    If the forward pair is absent, a materialized reverse relation is used to
    create its canonical inverse.  Otherwise, when the source sector has higher
    topological dimension than the target, the topology inference service is
    used as a fallback.

    Raises:
        KeyError: If either sector id is unknown.
        ValueError: If no relation can be resolved or inferred.
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
