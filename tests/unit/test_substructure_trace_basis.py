"""子结构接口迹空间的线性代数契约测试."""

from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    FullTraceBasis,
    LinearCornerTraceBasis,
    StreamingShapeFunctionCondensation,
    SubstructurePrototype,
)


def test_full_trace_basis_is_identity_for_stiffness_and_displacement() -> None:
    """``full_trace`` 必须保持接口刚度、恢复矩阵和位移不变."""
    bm.set_backend("numpy")
    basis = FullTraceBasis(4)
    stiffness = bm.asarray(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 3.0, -1.0, 0.0],
            [0.0, -1.0, 3.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ],
        dtype=bm.float64,
    )
    recovery = bm.asarray(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
        dtype=bm.float64,
    )
    displacement = bm.asarray(
        [[0.2, -0.1, 0.4, 0.3], [1.0, 2.0, 3.0, 4.0]],
        dtype=bm.float64,
    )

    np.testing.assert_allclose(
        bm.to_numpy(basis.project_stiffness(stiffness)),
        bm.to_numpy(stiffness),
    )
    np.testing.assert_allclose(
        bm.to_numpy(basis.reduce_recovery(recovery)),
        bm.to_numpy(recovery),
    )
    np.testing.assert_allclose(
        bm.to_numpy(basis.expand_displacement(displacement)),
        bm.to_numpy(displacement),
    )


def test_linear_corner_basis_matches_huang_equation_16_in_2d() -> None:
    """``linear_corner`` 的刚度和位移变换必须与原显式矩阵公式一致."""
    bm.set_backend("numpy")
    prototype = SubstructurePrototype(
        cell_size=(1.0, 1.0),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    basis = LinearCornerTraceBasis.from_prototype(prototype)
    interpolation = prototype.linear_boundary_matrix
    n_boundary = int(prototype.n_b)

    raw = np.arange(n_boundary * n_boundary, dtype=np.float64).reshape(
        n_boundary,
        n_boundary,
    )
    stiffness = bm.asarray(raw.T @ raw + np.eye(n_boundary), dtype=bm.float64)
    expected_stiffness = (
        bm.matrix_transpose(interpolation)
        @ stiffness
        @ interpolation
    )

    trace_displacement = bm.asarray(
        np.arange(16, dtype=np.float64).reshape(2, 8) / 10.0,
        dtype=bm.float64,
    )
    expected_boundary = trace_displacement @ bm.matrix_transpose(interpolation)

    assert basis.name == "linear_corner"
    assert basis.n_boundary_dofs == n_boundary
    assert basis.n_trace_dofs == 8
    np.testing.assert_allclose(
        bm.to_numpy(basis.project_stiffness(stiffness)),
        bm.to_numpy(expected_stiffness),
    )
    np.testing.assert_allclose(
        bm.to_numpy(basis.expand_displacement(trace_displacement)),
        bm.to_numpy(expected_boundary),
    )


def test_linear_corner_basis_has_expected_3d_shape_and_partition_of_unity() -> None:
    """三维角点迹应具有 24 列且每个边界位移分量保持单位分解."""
    bm.set_backend("numpy")
    prototype = SubstructurePrototype(
        cell_size=(1.0, 1.0, 1.0),
        n_fine=(2, 2, 2),
        E_base=1.0,
        nu=0.3,
    )
    basis = LinearCornerTraceBasis.from_prototype(prototype)
    matrix = bm.to_numpy(basis.matrix)

    assert matrix.shape == (int(prototype.n_b), 24)
    np.testing.assert_allclose(
        matrix.sum(axis=1),
        np.ones(int(prototype.n_b)),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_streaming_condensation_new_and_legacy_trace_apis_are_equivalent() -> None:
    """旧宏观接口必须与新的 TraceBasis 接口给出相同刚度和恢复结果."""
    bm.set_backend("numpy")
    prototype = SubstructurePrototype(
        cell_size=(1.0, 1.0),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    basis = LinearCornerTraceBasis.from_prototype(prototype)
    n_boundary = int(prototype.n_b)
    n_internal = int(prototype.n_i)

    raw = np.arange(n_boundary * n_boundary, dtype=np.float64).reshape(
        n_boundary,
        n_boundary,
    )
    stiffness_single = raw.T @ raw + np.eye(n_boundary)
    stiffness = bm.asarray(
        np.stack([stiffness_single, 1.5 * stiffness_single]),
        dtype=bm.float64,
    )
    recovery = np.arange(
        2 * n_internal * n_boundary,
        dtype=np.float64,
    ).reshape(2, n_internal, n_boundary) / 100.0
    condensor = StreamingShapeFunctionCondensation(
        prototype.i_dofs,
        prototype.b_dofs,
        Ks_batch=stiffness,
        N_hetero_dict={0: recovery[0], 1: recovery[1]},
        is_homo=bm.asarray([False, False], dtype=bm.bool),
        n_sub_total=2,
    )
    trace_displacement = bm.asarray(
        np.arange(16, dtype=np.float64).reshape(2, 8) / 10.0,
        dtype=bm.float64,
    )

    projected_new = condensor.get_projected_stiffness(basis)
    projected_old = condensor.get_macro_stiffness(basis.matrix)
    boundary_new, internal_new = condensor.recover_from_trace(
        trace_displacement,
        basis,
    )
    boundary_old, internal_old = condensor.recover_from_macro(
        trace_displacement,
        basis.matrix,
    )

    np.testing.assert_allclose(
        bm.to_numpy(projected_new),
        bm.to_numpy(projected_old),
    )
    np.testing.assert_allclose(
        bm.to_numpy(boundary_new),
        bm.to_numpy(boundary_old),
    )
    np.testing.assert_allclose(
        bm.to_numpy(internal_new),
        bm.to_numpy(internal_old),
    )
