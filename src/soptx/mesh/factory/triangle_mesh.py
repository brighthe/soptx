# 移植自 brighthe/fealpy ``fealpy/mesh/factory/triangle_mesh.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典三角形网格视图."""

from numbers import Integral

from soptx.backend import backend_manager as bm

from ..schema import LagrangeTriangleSchema
from .base import ClassicMeshView, register_classic_view
from .quadrangle_mesh import QuadrangleMesh


class TriangleMesh(ClassicMeshView):
    """经典三角形网格视图.

    ``TriangleMesh(node, cell)`` 构造单根 ``tri`` 分区并派生出边. 矩形区域上的结构化
    三角剖分有三种, 区别在于每个格子沿哪条对角线切开: ``from_box`` 全部取 ``/``,
    ``from_checkerboard_box`` 交替取 ``/`` 与 ``\\``, ``from_mirror_box`` 左右镜像.
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
        """在矩形区域上生成三角形网格, 每个格子沿 ``/`` 对角线切成两个三角形.

        Parameters
        ----------
        box : list of float, optional
            区域范围 ``[x0, x1, y0, y1]``.
        nx, ny : int, optional
            各方向的剖分数, 默认 10.
        device : optional
            设备.

        Notes
        -----
        单元编号约定 (公开契约, 平移类与几何多重网格依赖它): 格子按 x 最慢、y 最快的
        字典序排列, 每个格子切出的 N_k = 2 个单元连续编号, 局部顶点相对格子最小角点
        的偏移依次为 ``(0,0), (1,0), (1,1)`` 与 ``(0,0), (1,1), (0,1)``. 于是单元 e
        位于第 ``e // 2`` 个格子, 属于第 ``e % 2`` 个平移类.
        """
        from ..generation import Box2d

        box = Box2d(box, nx, ny, device=device)
        return cls.from_block(box.triangulate().block)

    @classmethod
    def from_checkerboard_box(cls, box, nx, ny, *, device=None):
        """在矩形区域上生成棋盘格三角形网格, 相邻格子交替取 ``/`` 与 ``\\`` 对角线.

        格子 ``(i, j)`` 在 ``i + j`` 为偶数时取 ``/``, 否则取 ``\\``. ``nx``、``ny``
        均为正偶数时, 矩形的四个角点各只与两个共享一条内部边的三角形相邻, 满足
        Hu--Zhang 角点松弛的拓扑要求. 该要求来自当前松弛代码的实现, 网格本身同样
        可用于位移元.

        Parameters
        ----------
        box : sequence of float
            区域范围 ``[x0, x1, y0, y1]``.
        nx, ny : int
            各方向的剖分数, 须为正偶数.
        device : optional
            设备.

        Returns
        -------
        TriangleMesh
            单元先排所有 ``/`` 格子切出的三角形, 再排所有 ``\\`` 格子切出的三角形,
            不遵循 ``from_box`` 的编号约定.

        Raises
        ------
        TypeError
            ``nx`` 或 ``ny`` 不是整数.
        ValueError
            ``nx`` 或 ``ny`` 不是正偶数.
        """
        for name, value in (("nx", nx), ("ny", ny)):
            _check_integer(name, value)
            if value <= 0 or value % 2 != 0:
                raise ValueError(f"{name} 须为正偶数, 实为 {value}")

        qmesh = QuadrangleMesh.from_box(box=list(box), nx=int(nx), ny=int(ny), device=device)
        ix = bm.arange(int(nx), dtype=bm.int32, device=qmesh.device)[:, None]
        iy = bm.arange(int(ny), dtype=bm.int32, device=qmesh.device)[None, :]
        use_left_diagonal = (bm.remainder(ix + iy, 2) == 0).reshape(-1)
        return cls(*_split_quads_along_diagonals(qmesh, use_left_diagonal))

    @classmethod
    def from_mirror_box(cls, box, nx, ny, *, device=None):
        """在矩形区域上生成左右镜像对称的三角形网格.

        左半 (``i < nx // 2``) 的格子取 ``/``, 右半取 ``\\``, 剖分关于竖直中线
        ``x = (x0 + x1) / 2`` 镜像对称. 两个顶角格子 ``(0, ny-1)`` 与 ``(nx-1, ny-1)``
        分别翻转为 ``\\`` 与 ``/``, 使矩形的四个角点各只与两个共享一条内部边的三角形
        相邻 (Hu--Zhang 角点松弛的拓扑要求); 两处翻转互为镜像. 除这两个格子外, 每个
        内部顶点周围都是 6 个三角形, 即低阶位移元体积闭锁的经典构型.

        Parameters
        ----------
        box : sequence of float
            区域范围 ``[x0, x1, y0, y1]``.
        nx : int
            x 方向剖分数, 须为不小于 2 的偶数, 使镜像中线落在网格线上.
        ny : int
            y 方向剖分数, 须不小于 2.
        device : optional
            设备.

        Returns
        -------
        TriangleMesh
            单元先排所有 ``/`` 格子切出的三角形, 再排所有 ``\\`` 格子切出的三角形,
            不遵循 ``from_box`` 的编号约定.

        Raises
        ------
        TypeError
            ``nx`` 或 ``ny`` 不是整数.
        ValueError
            ``nx`` 或 ``ny`` 小于 2, 或 ``nx`` 为奇数.
        """
        for name, value in (("nx", nx), ("ny", ny)):
            _check_integer(name, value)
            if value < 2:
                raise ValueError(f"{name} 须不小于 2, 实为 {value}")
        if nx % 2 != 0:
            raise ValueError(f"nx 须为偶数, 使镜像中线落在网格线上, 实为 {nx}")

        qmesh = QuadrangleMesh.from_box(box=list(box), nx=int(nx), ny=int(ny), device=device)
        # 四边形按 i * ny + j 排列 (i 沿 x, j 沿 y); 左半 '/', 右半 '\'
        ix = bm.arange(int(nx), dtype=bm.int32, device=qmesh.device)[:, None]
        iy = bm.arange(int(ny), dtype=bm.int32, device=qmesh.device)[None, :]
        use_left_diagonal = ((ix + 0 * iy) < (int(nx) // 2)).reshape(-1)
        top_left = bm.array([int(ny) - 1], dtype=bm.int32, device=qmesh.device)
        top_right = bm.array([(int(nx) - 1) * int(ny) + (int(ny) - 1)], dtype=bm.int32, device=qmesh.device)
        use_left_diagonal = bm.set_at(use_left_diagonal, top_left, False)
        use_left_diagonal = bm.set_at(use_left_diagonal, top_right, True)
        return cls(*_split_quads_along_diagonals(qmesh, use_left_diagonal))


def _check_integer(name, value):
    """检查剖分数是整数 (排除 bool)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} 须为整数, 实为 {value!r}")


def _split_quads_along_diagonals(qmesh, use_left_diagonal):
    """把每个四边形沿 ``/`` (node0--node2) 或 ``\\`` (node1--node3) 切成两个三角形.

    Parameters
    ----------
    qmesh : QuadrangleMesh
        ``QuadrangleMesh.from_box`` 生成的网格, 局部顶点依次为左下、右下、右上、左上.
    use_left_diagonal : TensorLike
        (NC, ) 的布尔数组, ``True`` 取 ``/``, ``False`` 取 ``\\``.

    Returns
    -------
    node : TensorLike
        节点坐标, 与 ``qmesh`` 相同.
    cell : TensorLike
        三角形单元, 先排 ``/`` 格子切出的, 再排 ``\\`` 格子切出的.
    """
    node = qmesh.entity("node")
    quad = qmesh.entity("cell")
    left = quad[use_left_diagonal]
    right = quad[~use_left_diagonal]

    cell = bm.concat(
        [
            left[:, [1, 2, 0]],
            left[:, [3, 0, 2]],
            right[:, [0, 1, 3]],
            right[:, [2, 3, 1]],
        ],
        axis=0,
    )
    return node, cell


register_classic_view(LagrangeTriangleSchema, TriangleMesh)
