# 移植自 brighthe/fealpy ``fealpy/mesh/aggregate.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Top-level multi-:class:`MeshBlock` aggregate object.

The aggregate ``Mesh`` owns several ``MeshBlock`` values and exposes anchored
``MeshView`` access, but it does not define block-global ``cell``/``face``/
``edge``/``node`` roles.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .storage import MeshBlock
from .view.mesh_view import MeshView

__all__ = ["Mesh"]


@dataclass(slots=True)
class Mesh:
    """Aggregate one or more :class:`MeshBlock` values.

    Block keys are :attr:`MeshBlock.id`.  Use :meth:`view` to obtain a
    root-cell-anchored :class:`MeshView`.
    """

    blocks: dict[str, MeshBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the block collection and its keys."""
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
        """Build an aggregate from an iterable of mesh blocks.

        Raises:
            ValueError: If two blocks share the same ``id``.
        """
        mapping: dict[str, MeshBlock] = {}
        for block in blocks:
            if block.id in mapping:
                raise ValueError(f"duplicate MeshBlock id {block.id!r}")
            mapping[block.id] = block
        return cls(blocks=mapping)

    def __len__(self) -> int:
        """Return the number of mesh blocks."""
        return len(self.blocks)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Return the stable block ids in insertion order."""
        return tuple(self.blocks)

    def block(self, block_id: str, /) -> MeshBlock:
        """Return the mesh block with ``block_id``.

        Raises:
            KeyError: If no such block exists.
        """
        if block_id not in self.blocks:
            raise KeyError(f"MeshBlock {block_id!r} not found")
        return self.blocks[block_id]

    def view(self, block_id: str, cell_sector_id: str, /) -> MeshView:
        """Return an anchored view on one block's root cell sector."""
        return MeshView(self.block(block_id), cell_sector_id)
