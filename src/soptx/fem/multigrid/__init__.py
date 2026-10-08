"""几何多重网格的层次构造: 网格序列, 延拓与 Galerkin 粗层算子.

求解层的 :class:`~soptx.solvers.Multigrid` 只认算子与延拓, 层次由本子包按离散侧的
网格与空间构造后交给它.
"""

from .structured_hex import StructuredHexGrid, StructuredHexHierarchy

__all__ = [
    "StructuredHexGrid",
    "StructuredHexHierarchy",
]
