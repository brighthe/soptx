"""三维子结构宏观角点顺序及 Huang 2023 式 (16) 回归测试."""

from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    SubstructurePrototype,
    build_substructures,
)


def _corner_nodes_from_dofs(corner_dofs: np.ndarray) -> np.ndarray:
    """将节点优先的三维角点自由度恢复成节点编号."""
    reshaped = corner_dofs.reshape(8, 3)
    np.testing.assert_array_equal(
        reshaped % 3, np.tile(np.array([0, 1, 2]), (8, 1))
    )
    return reshaped[:, 0] // 3


def test_3d_macro_corner_indices_are_unique_and_follow_l_column_order() -> None:
    """八角点必须与 L 的 000,001,010,011,100,101,110,111 顺序一致."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        domain_size=(2.0, 1.0, 1.0),
        n_sub=(2, 1, 1),
        n_fine=(1, 1, 1),
    )
    _, sub_meshes, _ = build_substructures(assembler)
    corner_dofs = bm.to_numpy(assembler.macro_corner_indices(sub_meshes))

    expected_nodes = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10, 11],
        ],
        dtype=np.int64,
    )
    actual_nodes = np.stack(
        [_corner_nodes_from_dofs(row) for row in corner_dofs], axis=0
    )

    np.testing.assert_array_equal(actual_nodes, expected_nodes)
    assert all(len(np.unique(row)) == 8 for row in actual_nodes)


def test_adjacent_3d_substructures_share_exactly_four_face_corner_nodes() -> None:
    """沿 x 相邻的两个 Hex 子结构只能共享公共面上的四个宏观角点."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        domain_size=(2.0, 1.0, 1.0),
        n_sub=(2, 1, 1),
        n_fine=(1, 1, 1),
    )
    _, sub_meshes, _ = build_substructures(assembler)
    corner_dofs = bm.to_numpy(assembler.macro_corner_indices(sub_meshes))
    nodes = [_corner_nodes_from_dofs(row) for row in corner_dofs]

    np.testing.assert_array_equal(np.intersect1d(nodes[0], nodes[1]), [4, 5, 6, 7])
    assert len(np.intersect1d(corner_dofs[0], corner_dofs[1])) == 12


def test_3d_exact_condensation_satisfies_huang_equation_16_energy_identity() -> None:
    """实际 H8 子结构应满足细尺度与式 (16) 宏观降阶的能量恒等式."""
    bm.set_backend("numpy")
    prototype = SubstructurePrototype(
        cell_size=(1.0, 1.0, 1.0),
        n_fine=(2, 2, 2),
        E_base=1.0,
        nu=0.3,
    )
    density_grid = bm.asarray(
        np.linspace(0.55, 0.95, 8).reshape(1, 2, 2, 2),
        dtype=bm.float64,
    )
    density_cell = prototype.grid_to_cell_field(density_grid)
    stiffness = prototype.assemble_local_stiffness_batch(density_cell)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    stiffness_s, recovery = condensor.condense(stiffness)

    n_boundary = len(prototype.b_dofs)
    extension = bm.zeros(
        (prototype.n_total_dofs, n_boundary), dtype=bm.float64
    )
    extension = bm.set_at(
        extension,
        (prototype.b_dofs, slice(None)),
        bm.eye(n_boundary, dtype=bm.float64),
    )
    extension = bm.set_at(
        extension,
        (prototype.i_dofs, slice(None)),
        recovery[0],
    )
    interpolation = prototype.linear_boundary_matrix
    fine_to_macro = extension @ interpolation

    lhs = bm.matrix_transpose(fine_to_macro) @ stiffness[0] @ fine_to_macro
    rhs = (
        bm.matrix_transpose(interpolation)
        @ stiffness_s[0]
        @ interpolation
    )
    np.testing.assert_allclose(
        bm.to_numpy(lhs), bm.to_numpy(rhs), rtol=2.0e-12, atol=2.0e-12
    )
