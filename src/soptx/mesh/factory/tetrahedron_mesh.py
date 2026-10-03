# 移植自 brighthe/fealpy ``fealpy/mesh/factory/tetrahedron_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from ..schema import LagrangeTetrahedronSchema
from .base import ClassicMeshView, register_classic_view


class TetrahedronMesh(ClassicMeshView):
    schema_type = LagrangeTetrahedronSchema
    """Classic tetrahedral mesh view.

    ``TetrahedronMesh(node, cell)`` constructs a single-root ``tet`` sector
    with derived triangular faces and edges.
    """

    schema_name = "tet"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1, 0, 1],
        nx=10,
        ny=10,
        nz=10,
        *,
        threshold=None,
        device=None,
    ):
        """Create a tetrahedron mesh of the box."""
        from ..generation import Box3d

        box = Box3d(box, nx, ny, nz, device=device)
        return cls.from_block(box.tetrahedralize().block)


register_classic_view(LagrangeTetrahedronSchema, TetrahedronMesh)
