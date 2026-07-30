"""Compatibility exports for utilities moved to :mod:`soptx.core`."""

from warnings import warn

from soptx.core import timer

warn(
    "soptx.utils is deprecated; import shared infrastructure from "
    "soptx.core",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'timer',
]
