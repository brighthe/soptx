# 移植自 brighthe/fealpy ``fealpy/mesh/factory/triangle_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典三角形网格视图."""

from ..schema import LagrangeTriangleSchema
from .base import ClassicMeshView, register_classic_view


class TriangleMesh(ClassicMeshView):
    """经典三角形网格视图.

    ``TriangleMesh(node, cell)`` 构造单根 ``tri`` 分区并派生出边; ``from_box``
    生成矩形区域的结构化三角剖分 (每个矩形沿对角线分成两个三角形).
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
        device=None,
    ):
        """在 矩形区域上生成三角形网格.

        Parameters
        ----------
        box : list of float, optional
            区域范围 ``[x0, x1, y0, y1]``.
        nx, ny : int, optional
            各方向的剖分数, 默认 10.
        device : optional
            设备.
        """
        from ..generation import Box2d

        box = Box2d(box, nx, ny, device=device)
        return cls.from_block(box.triangulate().block)


register_classic_view(LagrangeTriangleSchema, TriangleMesh)
