"""网格构造工具."""

from .structured_triangle import (
    create_huzhang_checkerboard_mesh,
    create_huzhang_symmetric_single_diagonal_mesh,
)

__all__ = [
    "create_huzhang_checkerboard_mesh",
    "create_huzhang_symmetric_single_diagonal_mesh",
]
