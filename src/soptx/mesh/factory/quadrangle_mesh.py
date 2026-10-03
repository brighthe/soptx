# 移植自 brighthe/fealpy ``fealpy/mesh/factory/quadrangle_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from ..schema import LagrangeQuadrilateralSchema
from .base import ClassicMeshView, register_classic_view


class QuadrangleMesh(ClassicMeshView):
    schema_type = LagrangeQuadrilateralSchema
    """Classic quadrilateral mesh view.

    ``QuadrangleMesh(node, cell)`` constructs a single-root ``quad`` sector
    with derived edges.  ``from_box`` builds a structured box
    quadrangulation.
    """

    schema_name = "quad"

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
        """Create a quadrangle mesh of the box."""
        from ..generation import Box2d

        box = Box2d(box, nx, ny, device=device)
        return cls.from_block(box.quadrangulate().block)


register_classic_view(LagrangeQuadrilateralSchema, QuadrangleMesh)
