# 移植自 brighthe/fealpy ``fealpy/mesh/aggregate.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""顶层的多 :class:`MeshBlock` 聚合对象.

聚合 ``Mesh`` 拥有若干 ``MeshBlock``, 提供带锚点的 ``MeshView`` 访问, 但不定义
跨块的 ``cell``/``face``/``edge``/``node`` 角色.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .storage import MeshBlock
from .view.mesh_view import MeshView

__all__ = ["Mesh"]


@dataclass(slots=True)
class Mesh:
    """聚合一个或多个 :class:`MeshBlock`.

    以 :attr:`MeshBlock.id` 为键. 用 :meth:`view` 取得锚定在根单元上的
    :class:`MeshView`.
    """

    blocks: dict[str, MeshBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验网格块集合及其键."""
        if not isinstance(self.blocks, dict):
            raise TypeError("Mesh.blocks must be a dict")
        for key, block in self.blocks.items():
            if not isinstance(block, MeshBlock):
                raise TypeError(
                    f"MeshBlock {key!r} value must be a MeshBlock"
                )
            if block.id != key:
                raise ValueError(
                    f"Mesh block key {key!r} does not match "
                    f"MeshBlock.id {block.id!r}"
                )

    @classmethod
    def from_blocks(cls, blocks: Iterable[MeshBlock]) -> "Mesh":
        """由网格块的可迭代对象构造聚合.

        Raises
        ------
        ValueError
            两个网格块的 ``id`` 相同.
        """
        mapping: dict[str, MeshBlock] = {}
        for block in blocks:
            if block.id in mapping:
                raise ValueError(f"duplicate MeshBlock id {block.id!r}")
            mapping[block.id] = block
        return cls(blocks=mapping)

    def __len__(self) -> int:
        """网格块的个数."""
        return len(self.blocks)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """按插入顺序返回各网格块的 id."""
        return tuple(self.blocks)

    def block(self, block_id: str, /) -> MeshBlock:
        """返回 id 为 ``block_id`` 的网格块.

        Raises
        ------
        KeyError
            不存在该网格块.
        """
        if block_id not in self.blocks:
            raise KeyError(f"MeshBlock {block_id!r} not found")
        return self.blocks[block_id]

    def view(self, block_id: str, cell_sector_id: str, /) -> MeshView:
        """返回锚定在某个网格块根单元分区上的视图."""
        return MeshView(self.block(block_id), cell_sector_id)
