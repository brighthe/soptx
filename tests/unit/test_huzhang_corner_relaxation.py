"""Focused tests for two-cell Hu-Zhang corner relaxation topology."""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem import create_huzhang_checkerboard_mesh
from soptx.fem.spaces import HuZhangFESpace


def _box_mesh() -> TriangleMesh:
    bm.set_backend("numpy")
    return create_huzhang_checkerboard_mesh(
        box=(0.0, 1.0, 0.0, 1.0),
        nx=2,
        ny=2,
    )


def _fixed_diagonal_box_mesh() -> TriangleMesh:
    bm.set_backend("numpy")
    return TriangleMesh.from_box(
        box=(0.0, 1.0, 0.0, 1.0),
        nx=2,
        ny=2,
    )


def _box_corners():
    return bm.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=bm.float64,
    )


def test_checkerboard_mesh_has_alternating_diagonals_and_two_cell_corners() -> None:
    mesh = _box_mesh()
    assert mesh.number_of_cells() == 8

    nodes = bm.to_numpy(mesh.entity("node"))
    cells = bm.to_numpy(mesh.entity("cell"))
    for corner in bm.to_numpy(_box_corners()):
        matches = np.flatnonzero(np.all(np.isclose(nodes, corner), axis=1))
        assert matches.size == 1
        assert np.count_nonzero(cells == int(matches[0])) == 2

    edges = bm.to_numpy(mesh.entity("edge"))
    geometric_edges = {
        tuple(sorted(tuple(nodes[node_id]) for node_id in edge))
        for edge in edges
    }
    for ix in range(2):
        for iy in range(2):
            x0, x1 = ix / 2, (ix + 1) / 2
            y0, y1 = iy / 2, (iy + 1) / 2
            diagonal = (
                ((x0, y0), (x1, y1))
                if (ix + iy) % 2 == 0
                else ((x0, y1), (x1, y0))
            )
            assert tuple(sorted(diagonal)) in geometric_edges


@pytest.mark.parametrize(
    ("nx", "ny"),
    [(0, 2), (-2, 2), (1, 2), (2, 0), (2, -2), (2, 1)],
)
def test_checkerboard_mesh_requires_positive_even_subdivisions(
    nx: int,
    ny: int,
) -> None:
    bm.set_backend("numpy")
    with pytest.raises(ValueError, match="positive even integer"):
        create_huzhang_checkerboard_mesh(
            box=(0.0, 1.0, 0.0, 1.0),
            nx=nx,
            ny=ny,
        )


def test_two_cell_corner_topology_relaxes_every_candidate() -> None:
    mesh = _box_mesh()
    conforming = HuZhangFESpace(mesh=mesh, p=2, use_relaxation=False)
    relaxed = HuZhangFESpace(
        mesh=mesh,
        p=2,
        use_relaxation=True,
        corners=_box_corners(),
    )

    assert relaxed.NCP == 4
    assert (
        relaxed.number_of_global_dofs()
        == conforming.number_of_global_dofs() + 4
    )
    np.testing.assert_allclose(
        bm.to_numpy(relaxed.corner["coords"]),
        bm.to_numpy(_box_corners()),
    )
    assert relaxed.cell_to_dof().shape == conforming.cell_to_dof().shape
    assert relaxed.TM.shape == (
        relaxed.number_of_global_dofs(),
        relaxed.number_of_global_dofs(),
    )

    node_ids = bm.to_numpy(relaxed.corner["idx"])
    to_cell = bm.to_numpy(relaxed.corner["to_cell"])
    to_edge = bm.to_numpy(relaxed.corner["to_edge"])
    to_midedge = bm.to_numpy(relaxed.corner["to_midedge"])
    cells = bm.to_numpy(mesh.entity("cell"))
    edges = bm.to_numpy(mesh.entity("edge"))
    cell_to_edge = bm.to_numpy(mesh.cell_to_edge())
    boundary_edges = bm.to_numpy(mesh.boundary_edge_flag())

    for index, node_id in enumerate(node_ids):
        node_id = int(node_id)
        cell_data = to_cell[index]
        edge_data = to_edge[index]
        cell_ids = [int(cell_data[0]), int(cell_data[2])]

        assert cell_ids[0] != cell_ids[1]
        for offset in (0, 2):
            cell_id = int(cell_data[offset])
            local_vertex = int(cell_data[offset + 1])
            edge_id = int(edge_data[offset])
            local_endpoint = int(edge_data[offset + 1])
            assert cells[cell_id, local_vertex] == node_id
            assert boundary_edges[edge_id]
            assert edges[edge_id, local_endpoint] == node_id

        middle_edge = int(to_midedge[index])
        assert not boundary_edges[middle_edge]
        assert node_id in edges[middle_edge]
        assert all(middle_edge in cell_to_edge[cell_id] for cell_id in cell_ids)


def test_single_cell_corner_is_rejected_instead_of_filtered() -> None:
    mesh = _fixed_diagonal_box_mesh()

    with pytest.raises(ValueError, match="has 1 incident cells"):
        HuZhangFESpace(
            mesh=mesh,
            p=2,
            use_relaxation=True,
            corners=_box_corners(),
        )


def test_corner_must_match_exactly_one_mesh_node() -> None:
    mesh = _box_mesh()

    with pytest.raises(ValueError, match="must match exactly one mesh node"):
        HuZhangFESpace(
            mesh=mesh,
            p=2,
            use_relaxation=True,
            corners=bm.array([[2.0, 2.0]], dtype=bm.float64),
        )


def test_duplicate_corner_candidate_is_rejected() -> None:
    mesh = _box_mesh()
    duplicate = bm.array(
        [[0.0, 1.0], [0.0, 1.0]],
        dtype=bm.float64,
    )

    with pytest.raises(ValueError, match="duplicates mesh node"):
        HuZhangFESpace(
            mesh=mesh,
            p=2,
            use_relaxation=True,
            corners=duplicate,
        )


def test_vertex_fan_with_more_than_two_cells_is_rejected() -> None:
    bm.set_backend("numpy")
    node = bm.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.2],
            [0.2, 0.6],
        ],
        dtype=bm.float64,
    )
    cell = bm.array(
        [[0, 1, 3], [0, 3, 4], [0, 4, 2]],
        dtype=bm.int32,
    )
    mesh = TriangleMesh(node, cell)

    with pytest.raises(ValueError, match="has 3 incident cells"):
        HuZhangFESpace(
            mesh=mesh,
            p=2,
            use_relaxation=True,
            corners=node[[0]],
        )
