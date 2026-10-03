# 移植自 brighthe/fealpy ``fealpy/mesh/factory/triangle_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from ..schema import LagrangeTriangleSchema
from .base import ClassicMeshView, register_classic_view


class TriangleMesh(ClassicMeshView):
    """Classic triangular mesh view.

    ``TriangleMesh(node, cell)`` constructs a single-root ``tri`` sector with
    derived edges.  ``from_box`` builds a structured box triangulation.
    """

    schema_name = "tri"
    schema_type = LagrangeTriangleSchema

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1],
        nx=10,
        ny=10,
        *,
        threshold=None,
        device=None,
    ):
        """Create a triangle mesh of the box."""
        from ..generation import Box2d

        box = Box2d(box, nx, ny, device=device)
        return cls.from_block(box.triangulate().block)


register_classic_view(LagrangeTriangleSchema, TriangleMesh)
