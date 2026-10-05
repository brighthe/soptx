# 移植自 brighthe/fealpy ``fealpy/mesh/factory/base.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典网格类型: ``MeshView`` 的具体特化.

经典网格族实现为按具体 Schema 类型登记的轻量 ``MeshView`` 子类.
``ClassicMeshView(node, cell)`` 构造只含一个根分区的 ``MeshBlock`` 并返回具体的
经典网格类型. 保留历史上的经典网格名以兼容源码与行为; 公开的多块所有权对象仍是
``soptx.mesh.Mesh``.
"""

from typing import ClassVar

from ...backend import Tensor
from ..schema.entity_schema import EntitySchema
from ..storage import EntitySector, MeshBlock
from ..topology.builder import TopologyBuilder
from ..view.mesh_view import MeshView


VIEW_REGISTRY: dict[type, type["ClassicMeshView"]] = {}
"""具体 Schema 类型到其经典 MeshView 特化的映射."""


def register_classic_view(
    schema_type: type,
    view_type: type["ClassicMeshView"],
) -> None:
    """为 Schema 类型登记其经典 MeshView 特化."""
    VIEW_REGISTRY[schema_type] = view_type


class ClassicMeshView(MeshView):
    """经典网格视图的基类.

    经典网格类型是绑定单一根单元 Schema 的轻量 ``MeshView`` 子类.
    ``ClassicMeshView(node, cell)`` 构造只含一个根分区的 ``MeshBlock``, 建立规范
    拓扑并返回具体的经典类型; ``from_block`` 把已构造好的网格块重新包装, 不重建.
    经典的基函数方法经私有的 ``EntityView`` 适配器保持历史列序; 下游的函数空间代码
    应把本对象当作 :class:`MeshView` 使用其历史接口, 不要依赖私有适配器的名字.

    Parameters
    ----------
    node : Tensor
        节点坐标, 形状 ``(NN, GD)``.
    cell : Tensor
        单元的顶点编号, 形状 ``(NC, NVC)``, 顶点顺序遵循对应 Schema 的约定.
    """

    schema_name: ClassVar[str]
    schema_type: ClassVar[type[EntitySchema]]

    def __init__(self, node: Tensor, cell: Tensor) -> None:
        """构造单根网格块并绑定到本视图."""
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
        """把已构造好的网格块包装为本经典视图类型, 不重建拓扑."""
        instance = object.__new__(cls)
        MeshView.__init__(instance, block, cell_sector_id=cls.schema_name)
        return instance
