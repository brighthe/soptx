# 移植自 brighthe/fealpy ``fealpy/mesh/factory/tetrahedron_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典四面体网格视图."""

from ..schema import LagrangeTetrahedronSchema
from .base import ClassicMeshView, register_classic_view


class TetrahedronMesh(ClassicMeshView):
    """经典四面体网格视图.

    ``TetrahedronMesh(node, cell)`` 构造单根 ``tet`` 分区并派生出三角形面与边.
    """
    schema_type = LagrangeTetrahedronSchema

    schema_name = "tet"

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
        """在长方体区域上生成四面体网格.

        Parameters
        ----------
        box : list of float, optional
            区域范围 ``[x0, x1, y0, y1, z0, z1]``.
        nx, ny, nz : int, optional
            各方向的剖分数, 默认 10.
        device : optional
            设备.

        Notes
        -----
        单元编号约定 (公开契约, 平移类依赖它): 节点与格子均按 x 最慢、z 最快的字典序
        排列, 每个格子切出的 N_k = 6 个单元连续编号, 单元 e 位于第 ``e // 6`` 个格子,
        属于第 ``e % 6`` 个平移类. 局部顶点相对格子最小角点的偏移依次为

        - ``(0,0,0), (1,0,0), (0,1,0), (0,1,1)``
        - ``(0,0,0), (1,0,1), (1,0,0), (0,1,1)``
        - ``(0,0,0), (0,0,1), (1,0,1), (0,1,1)``
        - ``(0,1,0), (1,0,0), (1,1,0), (1,1,1)``
        - ``(1,0,0), (1,0,1), (1,1,1), (0,1,1)``
        - ``(0,1,0), (1,1,1), (0,1,1), (1,0,0)``
        """
        from ..generation import Box3d

        box = Box3d(box, nx, ny, nz, device=device)
        return cls.from_block(box.tetrahedralize().block)


register_classic_view(LagrangeTetrahedronSchema, TetrahedronMesh)
