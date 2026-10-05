# 移植自 brighthe/fealpy ``fealpy/mesh/schema/local_entity.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参数化 Schema 的不可变局部子实体描述."""

from __future__ import annotations

from dataclasses import dataclass

from .entity_schema import EntitySchema

__all__ = ["LocalEntityGroup"]


@dataclass(frozen=True, slots=True)
class LocalEntityGroup:
    """描述父 Schema 布局中的一组同类局部子实体.

    ``schema`` 是各行共享的具体、不可变的子 Schema. ``local_node_indices`` 的每一行按子实体
    完整的规范节点顺序, 给出父实体中的局部节点列位置; 各行不含网格块全局节点编号, 也不
    省略子实体的高阶节点.

    子实体的 Schema 值或节点布局不同时, 父实体在同一拓扑维数上可返回多个组. 零维时父
    Schema 用点组覆盖全部插值节点, 这与其拓扑顶点骨架不同.

    Parameters
    ----------
    schema : EntitySchema
        组内共享的具体子 Schema.
    local_node_indices : tuple of tuple of int
        不可变的 "出现 x 子节点" 行, 索引父 Schema 的完整局部节点布局.

    Raises
    ------
    TypeError
        Schema 或元组结构的类型不对.
    ValueError
        某行宽度不对、含负索引或重复的父局部节点列.
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
