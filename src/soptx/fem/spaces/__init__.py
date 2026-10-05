"""Hu--Zhang 应力空间的向后兼容导入路径.

应力空间已迁至 ``soptx.functionspace``, 结构网格生成器已迁至 ``soptx.mesh.structured_triangle``;
此处只做转发, 新代码请直接从这两处导入.
"""

from ...functionspace import (
    HuZhangFESpace,
    HuZhangFESpace2d,
    HuZhangFESpace3d,
    boundary_outward_sign,
)
from ...mesh import (
    create_huzhang_checkerboard_mesh,
    create_huzhang_symmetric_single_diagonal_mesh,
)

__all__ = [
    "HuZhangFESpace",
    "HuZhangFESpace2d",
    "boundary_outward_sign",
    "HuZhangFESpace3d",
    "create_huzhang_checkerboard_mesh",
    "create_huzhang_symmetric_single_diagonal_mesh",
]
