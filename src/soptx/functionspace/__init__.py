"""Deprecated function-space namespace retained for SOPTX 1.1.x."""

from warnings import warn

from soptx.fem.spaces import (
    HuZhangFESpace,
    HuZhangFESpace2d,
    HuZhangFESpace3d,
)

warn(
    "soptx.functionspace is deprecated; import spaces from soptx.fem.spaces",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HuZhangFESpace",
    "HuZhangFESpace2d",
    "HuZhangFESpace3d",
]
