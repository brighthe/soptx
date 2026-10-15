"""全局插值点个数与插值点、单元到插值点映射两条入口一致.

``MeshView.number_of_global_ipoints`` 只计被单元引用的顶点; 曾用 ``unique`` 排序计数,
随每次 ``space.function()`` 调用, 在百万单元网格上每次耗时秒级, 现改为布尔掩码计数.
本测试锁定计数语义: 含未被引用的节点时也与另外两条入口一致.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh

MESHES = [
    (TriangleMesh, [0, 1, 0, 1], (3, 2)),
    (QuadrangleMesh, [0, 1, 0, 1], (3, 2)),
    (TetrahedronMesh, [0, 1, 0, 1, 0, 1], (2, 2, 1)),
    (HexahedronMesh, [0, 1, 0, 1, 0, 1], (3, 2, 2)),
]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _assert_consistent(mesh, p: int) -> int:
    """断言 p 次全局插值点个数与插值点行数、cell_to_ipoint 的最大编号加一相同, 返回该个数."""
    count = mesh.number_of_global_ipoints(p)
    assert isinstance(count, int)
    assert count == int(mesh.interpolation_points(p).shape[0])
    assert count == int(bm.max(mesh.cell_to_ipoint(p))) + 1
    return count


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("mesh_type,box,n", MESHES, ids=[m[0].__name__ for m in MESHES])
def test_global_ipoints_match_other_entries(backend, mesh_type, box, n, p):
    """结构网格上三条入口给出相同的全局插值点个数, p = 1 时即节点数."""
    if backend == "pytorch":
        pytest.importorskip("torch")
    bm.set_backend(backend)
    mesh = mesh_type.from_box(box, *n)
    count = _assert_consistent(mesh, p)
    if p == 1:
        assert count == mesh.number_of_nodes()


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
@pytest.mark.parametrize("unused_first", [False, True], ids=["unused_last", "unused_first"])
def test_unreferenced_node_not_counted(backend, unused_first):
    """未被单元引用的节点不计入: 两个三角形共 4 个顶点、5 条边, 另有 1 个孤立节点."""
    if backend == "pytorch":
        pytest.importorskip("torch")
    bm.set_backend(backend)
    node = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cell = np.array([[0, 1, 2], [3, 2, 1]])
    isolated = np.array([[5.0, 5.0]])
    if unused_first:
        node, cell = np.vstack([isolated, node]), cell + 1
    else:
        node = np.vstack([node, isolated])
    mesh = TriangleMesh(bm.from_numpy(node), bm.from_numpy(cell))

    assert mesh.number_of_nodes() == 5
    # p 次: 顶点 4 个, 每边 p - 1 个内点, 每个单元 (p - 1)(p - 2) / 2 个内点
    for p, expected in [(1, 4), (2, 9), (3, 16)]:
        assert _assert_consistent(mesh, p) == expected
