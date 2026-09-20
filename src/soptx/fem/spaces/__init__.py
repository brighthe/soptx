"""Finite-element function spaces."""

from .huzhang_fe_space import HuZhangFESpace
from .huzhang_fe_space_2d import HuZhangFESpace2d, boundary_outward_sign
from .huzhang_fe_space_3d import HuZhangFESpace3d

# 向后兼容别名: 结构网格生成器已迁至 soptx.mesh.structured_triangle,
# 此处保留旧导入路径, 新代码请直接从 soptx.mesh 导入。
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
