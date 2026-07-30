"""Deprecated optimization namespace retained for SOPTX 1.1.x."""

from warnings import warn

warn(
    "soptx.optimization is deprecated; import from soptx.topology",
    DeprecationWarning,
    stacklevel=2,
)
