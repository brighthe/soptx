# 移植自 brighthe/fealpy ``fealpy/mesh/factory/hexahedron_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典六面体网格视图."""

from ..schema import LagrangeHexahedronSchema
from .base import ClassicMeshView, register_classic_view


class HexahedronMesh(ClassicMeshView):
    """经典六面体网格视图, 派生出四边形面与边."""
    schema_type = LagrangeHexahedronSchema

    schema_name = "hex"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1, 0, 1],
        nx=10,
        ny=10,
        nz=10,
        *,
        device=None,
    ):
        """在 长方体区域上生成六面体网格.

        Parameters
        ----------
        box : list of float, optional
            区域范围 ``[x0, x1, y0, y1, z0, z1]``.
        nx, ny, nz : int, optional
            各方向的剖分数, 默认 10.
        device : optional
            设备.
        """
        from ..generation import Box3d

        box = Box3d(box, nx, ny, nz, device=device)
        return cls.from_block(box.hexahedralize().block)


register_classic_view(LagrangeHexahedronSchema, HexahedronMesh)
