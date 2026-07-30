"""Shared validation for elasticity problem data."""

from __future__ import annotations

from math import isfinite
from typing import Sequence


def validated_domain(
    domain: Sequence[float],
    dimension: int,
) -> tuple[float, ...]:
    """Return validated axis-aligned box bounds."""

    values = tuple(float(value) for value in domain)
    if len(values) != 2 * dimension:
        raise ValueError(
            f"{dimension}D problems require {2 * dimension} domain bounds, "
            f"received {len(values)}"
        )
    for axis in range(dimension):
        lower, upper = values[2 * axis : 2 * axis + 2]
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError(
                f"domain axis {axis} must contain finite lower < upper "
                f"bounds, received ({lower}, {upper})"
            )
    return values
