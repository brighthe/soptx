# 移植自 brighthe/fealpy ``fealpy/mesh/factory/quadrangle_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典四边形网格视图."""

from ..schema import LagrangeQuadrilateralSchema
from .base import ClassicMeshView, register_classic_view


class QuadrangleMesh(ClassicMeshView):
    """经典四边形网格视图.

    ``QuadrangleMesh(node, cell)`` 构造单根 ``quad`` 分区并派生出边; ``from_box``
    生成矩形区域的结构化四边形剖分. 单元顶点按逆时针循环顺序给出.
    """
    schema_type = LagrangeQuadrilateralSchema

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
        """在 矩形区域上生成四边形网格.

        Parameters
        ----------
        box : list of float, optional
            区域范围 ``[x0, x1, y0, y1]``.
        nx, ny : int, optional
            各方向的剖分数, 默认 10.
        threshold : optional
            未使用.
        device : optional
            设备.
        """
        from ..generation import Box2d

        box = Box2d(box, nx, ny, device=device)
        return cls.from_block(box.quadrangulate().block)


register_classic_view(LagrangeQuadrilateralSchema, QuadrangleMesh)
