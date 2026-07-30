"""Small stable result types used across solver implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SolverResult:
    """Dimension-independent summary of a numerical solve."""

    converged: bool
    iterations: int
    residual_norm: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
