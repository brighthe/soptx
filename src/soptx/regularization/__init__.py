"""Deprecated filter namespace retained for SOPTX 1.1.x."""

from warnings import warn

warn(
    "soptx.regularization is deprecated; import filters from "
    "soptx.topology.filters",
    DeprecationWarning,
    stacklevel=2,
)
