"""结构化三角形网格生成器.

本模块提供矩形区域上的结构化三角剖分构造函数。两者的共同约束是: 矩形区域的
每个几何角点恰好与两个共享一条内部边的三角形相邻——这是 2D 胡张元角点松弛
实现所要求的拓扑。该约束来自当前松弛代码的实现方式, 并非胡张元方法本身的
限制, 因此这些网格同样可以作为位移元的对比算例使用。

Notes
-----
函数名中的 ``huzhang`` 前缀是历史遗留, 反映最初的使用场景; 网格构造本身不
依赖任何胡张元的函数空间对象。
"""

from numbers import Integral
from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.typing import TensorLike


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
    ix = bm.arange(int(nx), dtype=bm.int32, device=qmesh.device)[:, None]
    iy = bm.arange(int(ny), dtype=bm.int32, device=qmesh.device)[None, :]
    use_left_diagonal = ((ix + iy) % 2 == 0).reshape(-1)
    return _split_quads_along_diagonals(qmesh, use_left_diagonal)


def create_huzhang_symmetric_single_diagonal_mesh(
    box: Sequence[float],
    nx: int,
    ny: int,
    *,
    device=None,
) -> TriangleMesh:
    """Create a mirror-symmetric single-diagonal mesh compatible with corner relaxation.

    Quadrilaterals in the left half (``i < nx // 2``) are split along ``/`` and
    those in the right half along ``\\``, so the triangulation is symmetric
    about the vertical mid-line ``x = (xmin + xmax) / 2``.  The two top corner
    quadrilaterals ``(0, ny-1)`` and ``(nx-1, ny-1)`` are flipped (to ``\\``
    and ``/`` respectively) so that every rectangular-domain corner is
    incident to exactly two triangles sharing one interior edge; the two
    flips are mirror images of each other.  ``nx`` must be an even integer
    >= 2 (the seam has to fall on a mesh line) and ``ny`` an integer >= 2.

    Away from the two flipped cells every interior vertex fan has six
    triangles -- the classical volumetric-locking configuration for low-order
    displacement elements -- while a mirror-symmetric problem keeps a
    mirror-symmetric discretisation.
    """
    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, received {value!r}")
        if value < 2:
            raise ValueError(
                f"{name} must be an integer >= 2 for the symmetric single-diagonal "
                f"Hu-Zhang mesh, received {value}"
            )
    if nx % 2 != 0:
        raise ValueError(
            f"nx must be an even integer so that the mirror seam falls on a mesh "
            f"line, received {nx}"
        )

    qmesh = QuadrangleMesh.from_box(
        box=list(box),
        nx=int(nx),
        ny=int(ny),
        device=device,
    )
    # from_box 的四边形按 i*ny + j 排列 (i 沿 x, j 沿 y); 左半 '/', 右半 '\'
    ix = bm.arange(int(nx), dtype=bm.int32, device=qmesh.device)[:, None]
    iy = bm.arange(int(ny), dtype=bm.int32, device=qmesh.device)[None, :]
    use_left_diagonal = ((ix + 0 * iy) < (int(nx) // 2)).reshape(-1)
    top_left = bm.array([0 * int(ny) + (int(ny) - 1)], dtype=bm.int32, device=qmesh.device)
    top_right = bm.array([(int(nx) - 1) * int(ny) + (int(ny) - 1)], dtype=bm.int32, device=qmesh.device)
    use_left_diagonal = bm.set_at(use_left_diagonal, top_left, False)
    use_left_diagonal = bm.set_at(use_left_diagonal, top_right, True)
    return _split_quads_along_diagonals(qmesh, use_left_diagonal)


def _split_quads_along_diagonals(
    qmesh: QuadrangleMesh,
    use_left_diagonal: TensorLike,
) -> TriangleMesh:
    """Split each quadrilateral along node0--node2 (``/``) or node1--node3 (``\\``).

    ``from_box`` orders the local nodes bottom-left, bottom-right, top-right,
    top-left, so ``use_left_diagonal[c] = True`` selects the ``/`` diagonal
    of quadrilateral ``c`` and ``False`` selects ``\\``.
    """
    node = qmesh.entity("node")
    quad = qmesh.entity("cell")
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
    
