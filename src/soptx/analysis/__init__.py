"""Deprecated analysis namespace retained for SOPTX 1.1.x."""

from warnings import warn

from soptx.fem.solvers import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer

warn(
    "soptx.analysis is deprecated; import FEM components from soptx.fem",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HuZhangMFEMAnalyzer", "LagrangeFEMAnalyzer"]
