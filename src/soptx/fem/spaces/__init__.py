"""Finite-element function spaces."""

from .huzhang_fe_space import HuZhangFESpace
from .huzhang_fe_space_2d import HuZhangFESpace2d
from .huzhang_fe_space_3d import HuZhangFESpace3d

__all__ = [
    "HuZhangFESpace",
    "HuZhangFESpace2d",
    "HuZhangFESpace3d",
]
