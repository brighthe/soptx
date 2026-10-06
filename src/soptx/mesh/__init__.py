"""网格包.

v0.4 网格树移植自 FEALPy ``fealpy/mesh/`` @ f474a5775 (来源见各文件头), 只保留
SOPTX 用到的四类经典网格, 未移植绘图、半边结构与局部加密/粗化; ``structured_box``
与 ``structured_triangle`` 为 SOPTX 自有的结构网格生成器.

导入本包只依赖 numpy 与 scipy; vtk 仅在调用 ``write_mesh_to_vtu`` 时加载.
"""

# 以下导入顺序不可调换: 顶层 ``Mesh`` 须最后由 ``aggregate`` 覆盖
# ``mesh_base`` 中的同名别名.
from .schema import *
from .storage import *
from .view import *
from .vtk_writter import write_mesh_to_vtu
from .factory import *
from .mesh_base import *
from .aggregate import Mesh

from .structured_box import (
    MESH_TYPES,
    BoxMesh,
    BoxTranslationClasses,
    create_box_mesh,
)
from .structured_triangle import (
    create_huzhang_checkerboard_mesh,
    create_huzhang_symmetric_single_diagonal_mesh,
)

__all__ = [
    "Mesh",
    "MeshView",
    "EntityView",
    "MeshBlock",
    "EntitySector",
    "EntitySet",
    "EntityRelation",
    "EntityContext",
    "EntitySchema",
    "SchemaDescriptor",
    "SchemaResolver",
    "SchemaTypeRegistry",
    "LocalEntityGroup",
    "encode_schema_descriptor",
    "decode_schema_descriptor",
    "NodeSchema",
    "EdgeSchema",
    "LagrangeEdgeSchema",
    "TriangleSchema",
    "LagrangeTriangleSchema",
    "QuadrilateralSchema",
    "LagrangeQuadrilateralSchema",
    "TetrahedronSchema",
    "LagrangeTetrahedronSchema",
    "PrismSchema",
    "LagrangePrismSchema",
    "PyramidSchema",
    "LagrangePyramidSchema",
    "HexahedronSchema",
    "PolygonSchema",
    "LagrangeHexahedronSchema",
    "ClassicMeshView",
    "TriangleMesh",
    "QuadrangleMesh",
    "TetrahedronMesh",
    "HexahedronMesh",
    "write_mesh_to_vtu",
    "MESH_TYPES",
    "BoxMesh",
    "BoxTranslationClasses",
    "create_box_mesh",
    "create_huzhang_checkerboard_mesh",
    "create_huzhang_symmetric_single_diagonal_mesh",
]
