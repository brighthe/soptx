"""子结构接口迹空间及其线性变换."""

from .base import TraceBasis
from .full import FullTraceBasis
from .linear_corner import LinearCornerTraceBasis

__all__ = [
    "TraceBasis", "FullTraceBasis", "LinearCornerTraceBasis",
]
