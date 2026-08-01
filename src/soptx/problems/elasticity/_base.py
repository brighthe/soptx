"""Shared validation and boundary defaults for elasticity problem data."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


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


class AllDisplacementBoundaryMixin:
    """Mixed-formulation boundary defaults for an all-Dirichlet box problem.

    ``HuZhangMFEMAnalyzer`` partitions the boundary into a displacement part
    imposed weakly and a traction part imposed strongly.  For a problem whose
    whole boundary carries prescribed displacement the partition is trivial,
    and these defaults spell it out so such problems satisfy
    ``MixedBoundaryElasticityProblem`` without a per-caller adapter.

    Requires the host class to expose ``domain``, ``dimension`` and
    ``dirichlet_bc``.
    """

    _eps = 1.0e-12

    def mark_corners(self, node: TensorLike) -> TensorLike:
        """Return the corner coordinates of the axis-aligned box domain.

        A node is a corner when it sits on a bound of *every* axis, which
        generalises the 2D square case to any dimension.
        """
        domain = self.domain
        on_every_axis = None
        for axis in range(self.dimension):
            coordinate = node[:, axis]
            lower, upper = domain[2 * axis], domain[2 * axis + 1]
            on_axis = (
                bm.abs(coordinate - lower) < self._eps
            ) | (bm.abs(coordinate - upper) < self._eps)
            on_every_axis = (
                on_axis if on_every_axis is None else on_every_axis & on_axis
            )
        return node[on_every_axis]

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """The whole boundary prescribes displacement."""
        return bm.ones(points.shape[:-1], dtype=bm.bool)

    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """No part of the boundary prescribes traction."""
        return bm.zeros(points.shape[:-1], dtype=bm.bool)

    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """Weak displacement data for the mixed formulation.

        ``HuZhangMFEMAnalyzer`` looks this member up with a ``getattr``
        default and falls back to homogeneous data when it is missing, which
        is silently wrong for any problem whose exact displacement does not
        vanish on the boundary.  Reusing ``dirichlet_bc`` keeps the weak data
        exact and identical to the primal formulation's strong data.
        """
        return self.dirichlet_bc(points)

    def traction_bc(self, points: TensorLike) -> TensorLike:
        """Reject traction queries instead of inventing zero traction data.

        ``is_traction_boundary`` is empty, so no analyzer can reach this on a
        well-formed path.  Raising keeps the member present for the protocol
        while refusing to answer a question this problem cannot answer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} prescribes displacement on the whole "
            "boundary and carries no traction data"
        )
