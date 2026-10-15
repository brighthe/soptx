"""二维网格几何角点检出 ``boundary_corner_index`` 的单元测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.mesh import TriangleMesh


def _corner_coordinates(mesh) -> np.ndarray:
    node = bm.to_numpy(mesh.entity("node"))
    index = bm.to_numpy(mesh.boundary_corner_index())
    coords = node[index]
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TriangleMesh.from_checkerboard_box(box=[0.0, 4.0, 0.0, 2.0], nx=4, ny=2),
        lambda: TriangleMesh.from_box(box=[0.0, 4.0, 0.0, 2.0], nx=4, ny=2),
    ],
    ids=["checkerboard", "single-diagonal"],
)
def test_box_mesh_has_exactly_four_corners(factory) -> None:
    """盒形域无论剖分方式, 几何角点都是四个顶点, 直边上的边界顶点不算."""
    bm.set_backend("numpy")
    mesh = factory()
    expected = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0]])
    np.testing.assert_allclose(_corner_coordinates(mesh), expected)
    assert int(bm.sum(mesh.boundary_node_flag())) > 4


def test_notched_polygon_reports_every_vertex_including_reentrant() -> None:
    """带底边缺口的七边形: 七个多边形顶点都是几何角点, 含凹角顶点."""
    bm.set_backend("numpy")
    vertices = np.array(
        [[0.0, 0.0], [0.8, 0.0], [1.0, 0.5], [1.2, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    )
    node = bm.array(np.vstack([vertices, [[1.0, 1.0]]]), dtype=bm.float64)
    center = 7
    cell = bm.array(
        [[center, i, (i + 1) % 7] for i in range(7)], dtype=bm.int32
    )
    mesh = TriangleMesh(node, cell)
    expected = vertices[np.lexsort((vertices[:, 1], vertices[:, 0]))]
    np.testing.assert_allclose(_corner_coordinates(mesh), expected)


def test_three_dimensional_mesh_is_rejected() -> None:
    bm.set_backend("numpy")
    from soptx.mesh import TetrahedronMesh

    mesh = TetrahedronMesh.from_box([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=1, ny=1, nz=1)
    with pytest.raises(ValueError, match="二维"):
        mesh.boundary_corner_index()
