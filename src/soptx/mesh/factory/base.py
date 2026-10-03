# 移植自 brighthe/fealpy ``fealpy/mesh/factory/base.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Real classic ``MeshView`` specializations.

The classic mesh family is implemented as thin ``MeshView`` subclasses keyed by
their concrete Schema type.  ``ClassicMeshView(node, cell)`` constructs a
single-root ``MeshBlock`` and returns the concrete classic type.  Historical
classic names are retained as source and behavior compatibility; the public
multi-block ownership object remains ``fealpy.mesh.Mesh``.
"""

from typing import ClassVar

from ...backend import Tensor
from ..schema.entity_schema import EntitySchema
from ..storage import EntitySector, MeshBlock
from ..topology.builder import TopologyBuilder
from ..view.mesh_view import MeshView


VIEW_REGISTRY: dict[type, type["ClassicMeshView"]] = {}
"""Map a concrete Schema type to its classic MeshView specialization."""


def register_classic_view(
    schema_type: type,
    view_type: type["ClassicMeshView"],
) -> None:
    """Register a concrete classic MeshView specialization for a Schema type."""
    VIEW_REGISTRY[schema_type] = view_type


class ClassicMeshView(MeshView):
    """Base for real classic MeshView subclasses.

    Classic mesh types are thin ``MeshView`` subclasses bound to one root cell
    Schema.  ``ClassicMeshView(node, cell)`` constructs a single-root
    ``MeshBlock``, builds canonical topology, and returns the concrete classic
    type.  ``from_block`` re-wraps an already constructed block without
    rebuilding it.  Classic basis methods preserve the historical column order
    through private ``EntityView`` adapters; downstream FunctionSpace code
    should consume this object as a :class:`MeshView` and use its historical
    accessors rather than relying on private adapter names.
    """

    schema_name: ClassVar[str]
    schema_type: ClassVar[type[EntitySchema]]

    def __init__(self, node: Tensor, cell: Tensor) -> None:
        """Construct a single-root mesh block and bind this classic view."""
        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id=self.schema_name,
                schema=self.schema_type(),
                indices=cell,
            ),
            root=True,
        )
        TopologyBuilder.construct(block)
        super().__init__(block, cell_sector_id=self.schema_name)

    @classmethod
    def from_block(cls, block: MeshBlock) -> "ClassicMeshView":
        """Wrap an already constructed block in this classic view type."""
        instance = object.__new__(cls)
        MeshView.__init__(instance, block, cell_sector_id=cls.schema_name)
        return instance
