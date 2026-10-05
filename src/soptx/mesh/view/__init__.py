# 移植自 brighthe/fealpy ``fealpy/mesh/view/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""公开的视图对象.

本包提供锚定在根单元上的 :class:`MeshView` 与绑定在实体分区上的 :class:`EntityView`.
这里的 ``Mesh`` 是 ``MeshView`` 的私有过渡别名, 只为仓库内的旧导入保留; 公开的顶层
``soptx.mesh.Mesh`` 是多块聚合对象.
"""

from .entity_view import EntityView
from .mesh_view import MeshView

# 私有兼容别名. 新代码须直接导入 ``MeshView``, 多块所有权则使用顶层的
# ``soptx.mesh.Mesh`` 聚合对象.
Mesh = MeshView

__all__ = ["EntityView", "MeshView"]
