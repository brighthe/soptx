"""子结构 cell 场双向映射及灵敏度回归测试."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    SubstructurePrototype,
    build_substructures,
    solve_interface_system,
)


@pytest.mark.parametrize(
    ("domain_size", "n_sub", "n_fine"),
    [
        ((4.0, 6.0), (2, 3), (2, 2)),
        ((4.0, 3.0, 4.0), (2, 1, 2), (2, 3, 2)),
    ],
)
def test_global_local_fe_cell_round_trip_uses_nonuniform_sentinels(
    domain_size: Tuple[float, ...],
    n_sub: Tuple[int, ...],
    n_fine: Tuple[int, ...],
) -> None:
    """2D/3D 非均匀哨兵场应跨三种顺序严格回环."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(domain_size, n_sub, n_fine)
    prototype = SubstructurePrototype(
        tuple(domain_size[d] / n_sub[d] for d in range(len(domain_size))),
        n_fine,
        E_base=1.0,
        nu=0.3,
    )
    total_fine = tuple(n_sub[d] * n_fine[d] for d in range(len(n_sub)))
    sentinel = (17.0 * np.arange(np.prod(total_fine)) + 3.0).reshape(total_fine)

    local_grid = assembler.split_global_cell_field(sentinel.reshape(-1))
    local_fe = prototype.grid_to_cell_field(local_grid)
    restored_local_grid = prototype.cell_to_grid_field(local_fe)
    restored_global = assembler.merge_substructure_cell_field(restored_local_grid)

    np.testing.assert_array_equal(bm.to_numpy(restored_global), sentinel)
    np.testing.assert_array_equal(
        bm.to_numpy(assembler.split_global_cell_field(restored_global)),
        bm.to_numpy(local_grid),
    )

    expected_blocks = []
    for position in np.ndindex(n_sub):
        slices = tuple(
            slice(position[d] * n_fine[d], (position[d] + 1) * n_fine[d])
            for d in range(len(n_sub))
        )
        expected_blocks.append(sentinel[slices])
    np.testing.assert_array_equal(
        bm.to_numpy(local_grid),
        np.stack(expected_blocks, axis=0),
    )


@pytest.mark.parametrize("n_fine", [(3, 2), (2, 3, 2)])
def test_local_grid_and_fe_cell_order_are_exact_inverses(
    n_fine: Tuple[int, ...],
) -> None:
    """局部 grid/FE cell 成对 API 应保留批量前导维并严格互逆."""
    bm.set_backend("numpy")
    prototype = SubstructurePrototype(
        tuple(float(n) for n in n_fine),
        n_fine,
        E_base=1.0,
        nu=0.3,
    )
    sentinel = (
        np.arange(2 * np.prod(n_fine), dtype=np.float64).reshape((2,) + n_fine)
        + np.array([1000.0, 2000.0]).reshape((2,) + (1,) * len(n_fine))
    )

    cell_field = prototype.grid_to_cell_field(sentinel)
    grid_field = prototype.cell_to_grid_field(cell_field)

    assert tuple(cell_field.shape) == (2, int(np.prod(n_fine)))
    np.testing.assert_array_equal(bm.to_numpy(grid_field), sentinel)
    np.testing.assert_array_equal(
        bm.to_numpy(prototype.to_cell_density(sentinel)),
        bm.to_numpy(cell_field),
    )


def _macro_compliance_and_gradient(rho_flat: np.ndarray) -> tuple[float, np.ndarray]:
    """计算一个两子结构小系统的柔度及未过滤解析梯度."""
    assembler = GlobalAssembler(
        domain_size=(2.0, 1.0),
        n_sub=(2, 1),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    prototype.rho_min = 1.0e-3
    prototype.penal = 3.0

    rho = bm.asarray(rho_flat, dtype=bm.float64)
    rho_sub_grid = assembler.split_global_cell_field(rho)
    rho_sub_cell = prototype.grid_to_cell_field(rho_sub_grid)
    stiffness = prototype.assemble_local_stiffness_batch(rho_sub_cell)

    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    stiffness_s, recovery = condensor.condense(stiffness)
    interpolation = prototype.linear_boundary_matrix
    interpolation_t = bm.matrix_transpose(interpolation)
    stiffness_macro = (
        interpolation_t[None, :, :]
        @ stiffness_s
        @ interpolation[None, :, :]
    )
    system = assembler.assemble_macro_system(sub_meshes, stiffness_macro)

    load = bm.zeros((assembler.total_macro_dofs,), dtype=bm.float64)
    load = bm.set_at(load, 11, -1.0)
    # 左下角固定两个分量, 左上角只固定 x 分量, 消除三个刚体模态.
    fixed = bm.asarray([0, 1, 2], dtype=bm.int64)
    displacement = solve_interface_system(system, load, fixed)

    corner_dofs = assembler.macro_corner_indices(sub_meshes)
    u_boundary = displacement[corner_dofs] @ interpolation_t
    u_internal = bm.einsum("bij,bj->bi", recovery, u_boundary)
    u_local = bm.zeros(
        (len(sub_meshes), prototype.n_total_dofs), dtype=bm.float64
    )
    u_local = bm.set_at(
        u_local, (slice(None), prototype.b_dofs), u_boundary
    )
    u_local = bm.set_at(
        u_local, (slice(None), prototype.i_dofs), u_internal
    )

    u_element = u_local[:, prototype.cell2dof]
    energy_cell = bm.sum(
        (u_element @ prototype.KE_unit[0]) * u_element,
        axis=-1,
    )
    energy_grid = prototype.cell_to_grid_field(energy_cell)
    energy_global = assembler.merge_substructure_cell_field(energy_grid)

    derivative = (
        prototype.penal
        * (1.0 - prototype.rho_min)
        * rho ** (prototype.penal - 1.0)
    )
    gradient = -derivative * bm.reshape(energy_global, (-1,))
    compliance = float(bm.dot(load, displacement))
    return compliance, bm.to_numpy(gradient)


def test_mapped_2d_compliance_gradient_matches_finite_difference() -> None:
    """映射后的 2D SIMP 灵敏度应与中心有限差分一致."""
    bm.set_backend("numpy")
    rho = np.array([0.71, 0.83, 0.64, 0.92, 0.77, 0.68, 0.88, 0.74])
    _, analytic = _macro_compliance_and_gradient(rho)
    selected = np.array([0, 3, 4, 7])
    epsilon = 1.0e-6
    finite_difference = []
    for cell in selected:
        rho_plus = rho.copy()
        rho_minus = rho.copy()
        rho_plus[cell] += epsilon
        rho_minus[cell] -= epsilon
        c_plus, _ = _macro_compliance_and_gradient(rho_plus)
        c_minus, _ = _macro_compliance_and_gradient(rho_minus)
        finite_difference.append((c_plus - c_minus) / (2.0 * epsilon))

    np.testing.assert_allclose(
        analytic[selected],
        np.asarray(finite_difference),
        rtol=2.0e-5,
        atol=2.0e-7,
    )
