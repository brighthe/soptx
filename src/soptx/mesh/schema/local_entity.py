# 移植自 brighthe/fealpy ``fealpy/mesh/schema/local_entity.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Immutable local-subentity descriptions for parameterized Schemas."""

from __future__ import annotations

from dataclasses import dataclass

from .entity_schema import EntitySchema

__all__ = ["LocalEntityGroup"]


@dataclass(frozen=True, slots=True)
class LocalEntityGroup:
    """Describe homogeneous local subentities in a parent Schema layout.

    ``schema`` is the concrete immutable child Schema shared by every row.
    Each row of ``local_node_indices`` contains parent local-node column
    positions in the child's complete canonical node order.  The rows do not
    contain block-global node IDs and do not omit higher-order child nodes.

    A parent may return several groups at one topological dimension when its
    children have different Schema values or node layouts.  At dimension zero
    a parent Schema uses point groups to cover all interpolation nodes; this is
    distinct from its topological vertex skeleton.

    Parameters:
        schema: Concrete child Schema shared by the group.
        local_node_indices: Immutable occurrence-by-child-node rows indexing
            the parent Schema's complete local-node layout.

    Raises:
        TypeError: If the Schema or tuple structure has an invalid type.
        ValueError: If a row has the wrong width, contains negative indices,
            or repeats a parent local-node column.
    """

    schema: EntitySchema
    local_node_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, EntitySchema):
            raise TypeError(
                "schema must be an EntitySchema instance, "
                f"got {type(self.schema).__name__}"
            )
        if type(self.local_node_indices) is not tuple:
            raise TypeError("local_node_indices must be a tuple of tuples")

        expected_width = self.schema.number_of_nodes()
        for row in self.local_node_indices:
            if type(row) is not tuple:
                raise TypeError("each local-node row must be a tuple")
            if len(row) != expected_width:
                raise ValueError(
                    f"local-node row width must be {expected_width}, "
                    f"got {len(row)}"
                )
            if any(type(index) is not int for index in row):
                raise TypeError("local-node indices must be plain integers")
            if any(index < 0 for index in row):
                raise ValueError("local-node indices must be non-negative")
            if len(set(row)) != len(row):
                raise ValueError("local-node rows must not repeat an index")
