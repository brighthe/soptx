"""Mesh factories required by maintained finite-element workflows."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh


def create_huzhang_checkerboard_mesh(
    box: Sequence[float],
    nx: int,
    ny: int,
    *,
    device=None,
) -> TriangleMesh:
    """Create a rectangular triangle mesh compatible with 2D corner relaxation.

    Each quadrilateral is split along alternating checkerboard diagonals.  For
    positive even ``nx`` and ``ny``, every rectangular-domain corner is
    incident to exactly two triangles with one shared interior edge, which is
    the topology supported by the current Hu--Zhang corner-relaxation code.

    This is a software constraint of the current relaxation implementation,
    not a general restriction of the Hu--Zhang method.
    """
    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, received {value!r}")
        if value <= 0 or value % 2 != 0:
            raise ValueError(
                f"{name} must be a positive even integer for Hu-Zhang "
                f"corner relaxation, received {value}"
            )

    qmesh = QuadrangleMesh.from_box(
        box=list(box),
        nx=int(nx),
        ny=int(ny),
        device=device,
    )
    node = qmesh.entity("node")
    quad = qmesh.entity("cell")

    ix = bm.arange(int(nx), dtype=bm.int32, device=qmesh.device)[:, None]
    iy = bm.arange(int(ny), dtype=bm.int32, device=qmesh.device)[None, :]
    use_left_diagonal = ((ix + iy) % 2 == 0).reshape(-1)
    left = quad[use_left_diagonal]
    right = quad[~use_left_diagonal]

    cell = bm.concatenate(
        [
            left[:, [1, 2, 0]],
            left[:, [3, 0, 2]],
            right[:, [0, 1, 3]],
            right[:, [2, 3, 1]],
        ],
        axis=0,
    )
    return TriangleMesh(node, cell)


__all__ = ["create_huzhang_checkerboard_mesh"]
