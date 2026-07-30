"""Compatibility imports for the three-dimensional Hu-Zhang space."""

from soptx.fem.spaces.huzhang_fe_space_3d import (
    HuZhangFECellDof3d,
    HuZhangFEDof3d,
    HuZhangFESpace3d,
    TensorDofsOnSubsimplex,
    multiindex_to_number,
    number_of_multiindex,
)

__all__ = [
    "HuZhangFECellDof3d",
    "HuZhangFEDof3d",
    "HuZhangFESpace3d",
    "TensorDofsOnSubsimplex",
    "multiindex_to_number",
    "number_of_multiindex",
]
