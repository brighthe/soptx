"""Compatibility imports for the two-dimensional Hu-Zhang space."""

from soptx.fem.spaces.huzhang_fe_space_2d import (
    HuZhangFECellDof2d,
    HuZhangFEDof2d,
    HuZhangFESpace2d,
    TensorDofsOnSubsimplex,
    multiindex_to_number,
    number_of_multiindex,
)

__all__ = [
    "HuZhangFECellDof2d",
    "HuZhangFEDof2d",
    "HuZhangFESpace2d",
    "TensorDofsOnSubsimplex",
    "multiindex_to_number",
    "number_of_multiindex",
]
