"""Elasticity problem 到子结构宏观系统的公共契约适配测试."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    build_substructures,
    project_problem_conditions_to_full_system,
    project_problem_conditions_to_macro_system,
    project_problem_conditions_to_nodes,
    solve_interface_system,
)
from soptx.problems.elasticity import FullMBBBeam2d, FullMBBBeam3d
from soptx.problems.loads import BodyForceLoad, BoundaryTractionLoad


class _LoadObjectOnlyMBB:
    """旧集中力接口不可调用、但 ``loads()`` 完整可用的测试 Problem."""

    def __init__(self, *, P: float = -1.0) -> None:
        self._base = FullMBBBeam2d(P=P)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def loads(self):
        return self._base.loads()

class _UniformTopTractionProblem:
    dimension = 2
    domain = (0.0, 2.0, 0.0, 1.0)

    def is_dirichlet_boundary(self):
        return (None, None)

    def loads(self):
        return (
            BoundaryTractionLoad(
                dimension=2,
                marker=lambda points: np.isclose(points[..., 1], 1.0),
                value=lambda points: np.broadcast_to(
                    (0.0, -2.0),
                    points.shape,
                ),
            ),
        )


class _UniformBodyForceProblem:
    dimension = 2
    domain = (0.0, 2.0, 0.0, 1.0)

    def is_dirichlet_boundary(self):
        return (None, None)

    def loads(self):
        return (
            BodyForceLoad(
                dimension=2,
                value=lambda points: np.broadcast_to(
                    (1.0, -3.0),
                    points.shape,
                ),
            ),
        )


def _rigid_mode_rank(
    coordinates: np.ndarray,
    fixed_dofs: np.ndarray,
) -> int:
    """计算受约束刚体模态矩阵的秩."""
    dim = coordinates.shape[1]
    n_nodes = coordinates.shape[0]
    if dim == 2:
        modes = np.zeros((2 * n_nodes, 3))
        modes[0::2, 0] = 1.0
        modes[1::2, 1] = 1.0
        modes[0::2, 2] = -coordinates[:, 1]
        modes[1::2, 2] = coordinates[:, 0]
    else:
        x, y, z = coordinates.T
        modes = np.zeros((3 * n_nodes, 6))
        modes[0::3, 0] = 1.0
        modes[1::3, 1] = 1.0
        modes[2::3, 2] = 1.0
        modes[1::3, 3] = -z
        modes[2::3, 3] = y
        modes[0::3, 4] = z
        modes[2::3, 4] = -x
        modes[0::3, 5] = -y
        modes[1::3, 5] = x
    return int(np.linalg.matrix_rank(modes[fixed_dofs]))


def test_full_mbb_2d_macro_contract_has_exact_dofs_and_load() -> None:
    """2D 宏观系统应只有左下 ux 与左右下 uy, 并施加标准集中力."""
    bm.set_backend("numpy")
    problem = FullMBBBeam2d(domain=(2.0, 14.0, -1.0, 1.0), P=-2.5)
    assembler = GlobalAssembler((12.0, 2.0), (4, 2), (1, 1))

    force, fixed = project_problem_conditions_to_macro_system(problem, assembler)
    force_np = bm.to_numpy(force)
    fixed_np = bm.to_numpy(fixed)

    np.testing.assert_array_equal(fixed_np, np.array([0, 1, 25]))
    np.testing.assert_array_equal(np.flatnonzero(force_np), np.array([17]))
    np.testing.assert_allclose(force_np[17], -2.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        force_np.reshape(-1, 2).sum(axis=0),
        np.array([0.0, -2.5]),
        rtol=0.0,
        atol=0.0,
    )

    physical_coordinates = (
        bm.to_numpy(assembler.macro_node_coordinates()) + np.array([2.0, -1.0])
    )
    assert _rigid_mode_rank(physical_coordinates, fixed_np) == 3


def test_full_mbb_3d_macro_contract_has_exact_dofs_and_load() -> None:
    """3D 宏观系统应精确投影三类标准约束与顶面中心集中力."""
    bm.set_backend("numpy")
    problem = FullMBBBeam3d(
        domain=(2.0, 14.0, -1.0, 1.0, 3.0, 5.0), P=-3.25
    )
    assembler = GlobalAssembler((12.0, 2.0, 2.0), (4, 2, 2), (1, 1, 1))

    force, fixed = project_problem_conditions_to_macro_system(problem, assembler)
    force_np = bm.to_numpy(force)
    fixed_np = bm.to_numpy(fixed)
    expected_fixed = np.array(
        [0, 1, 3, 4, 5, 6, 7, 32, 59, 86, 109, 112, 113, 115]
    )

    np.testing.assert_array_equal(fixed_np, expected_fixed)
    np.testing.assert_array_equal(np.flatnonzero(force_np), np.array([76]))
    np.testing.assert_allclose(force_np[76], -3.25, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        force_np.reshape(-1, 3).sum(axis=0),
        np.array([0.0, -3.25, 0.0]),
        rtol=0.0,
        atol=0.0,
    )

    physical_coordinates = (
        bm.to_numpy(assembler.macro_node_coordinates())
        + np.array([2.0, -1.0, 3.0])
    )
    assert _rigid_mode_rank(physical_coordinates, fixed_np) == 6


def test_full_mbb_3d_odd_cross_section_preserves_center_load_and_symmetry() -> None:
    """奇数横向分段应把中心载荷对称分配到最近节点并保持总力."""
    bm.set_backend("numpy")
    problem = FullMBBBeam3d(
        domain=(0.0, 6.0, 0.0, 1.0, 0.0, 1.0), P=-1.0
    )
    assembler = GlobalAssembler((6.0, 1.0, 1.0), (6, 3, 3), (1, 1, 1))

    force, fixed = project_problem_conditions_to_macro_system(problem, assembler)
    force_np = bm.to_numpy(force).reshape(-1, 3)
    fixed_np = bm.to_numpy(fixed)
    coordinates = bm.to_numpy(assembler.macro_node_coordinates())

    load_nodes = np.flatnonzero(force_np[:, 1])
    np.testing.assert_allclose(
        coordinates[load_nodes],
        np.array([[3.0, 1.0, 1.0 / 3.0], [3.0, 1.0, 2.0 / 3.0]]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        force_np[load_nodes, 1],
        np.array([-0.5, -0.5]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        force_np.sum(axis=0),
        np.array([0.0, -1.0, 0.0]),
        rtol=0.0,
        atol=1.0e-14,
    )

    z_fixed_nodes = fixed_np[fixed_np % 3 == 2] // 3
    assert z_fixed_nodes.size == 14
    np.testing.assert_allclose(coordinates[z_fixed_nodes, 1], 0.0)
    np.testing.assert_allclose(
        np.unique(coordinates[z_fixed_nodes, 2]),
        np.array([1.0 / 3.0, 2.0 / 3.0]),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert _rigid_mode_rank(coordinates, fixed_np) == 6


def test_nodal_adapter_accepts_numpy_coordinates_and_preserves_total_load() -> None:
    """底层适配器应接受 NumPy 坐标并保持集中载荷合力."""
    bm.set_backend("numpy")
    problem = FullMBBBeam2d(P=-4.0)
    coordinates = np.array(
        [[0.0, 0.0], [60.0, 20.0], [120.0, 0.0]], dtype=np.float64
    )

    force, fixed = project_problem_conditions_to_nodes(problem, coordinates)

    np.testing.assert_array_equal(bm.to_numpy(fixed), np.array([0, 1, 5]))
    np.testing.assert_array_equal(np.flatnonzero(bm.to_numpy(force)), np.array([3]))
    assert float(bm.sum(force)) == pytest.approx(-4.0)


def test_substructure_adapter_consumes_load_objects_not_legacy_point_api() -> None:
    bm.set_backend("numpy")
    problem = _LoadObjectOnlyMBB(P=-2.0)
    assembler = GlobalAssembler((120.0, 20.0), (4, 2), (1, 1))

    force, _fixed = project_problem_conditions_to_macro_system(
        problem,
        assembler,
    )

    np.testing.assert_allclose(
        bm.to_numpy(force).reshape(-1, 2).sum(axis=0),
        np.array((0.0, -2.0)),
        rtol=0.0,
        atol=0.0,
    )


def test_full_system_integrates_boundary_traction_load_object() -> None:
    bm.set_backend("numpy")
    problem = _UniformTopTractionProblem()
    assembler = GlobalAssembler((2.0, 1.0), (2, 1), (1, 1))

    force, fixed = project_problem_conditions_to_full_system(problem, assembler)
    nodal_force = bm.to_numpy(force).reshape(-1, 2)

    np.testing.assert_array_equal(bm.to_numpy(fixed), np.array([], dtype=np.int64))
    np.testing.assert_allclose(
        nodal_force.sum(axis=0),
        np.array((0.0, -4.0)),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_full_system_integrates_body_force_load_object() -> None:
    bm.set_backend("numpy")
    problem = _UniformBodyForceProblem()
    assembler = GlobalAssembler((2.0, 1.0), (2, 1), (1, 1))

    force, _fixed = project_problem_conditions_to_full_system(problem, assembler)

    np.testing.assert_allclose(
        bm.to_numpy(force).reshape(-1, 2).sum(axis=0),
        np.array((2.0, -6.0)),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_uniform_2d_macro_system_with_problem_contract_is_solvable() -> None:
    """标准载荷与约束下的均匀 2D 小系统应可解且柔度有限为正."""
    bm.set_backend("numpy")
    problem = FullMBBBeam2d(domain=(0.0, 2.0, 0.0, 1.0), P=-1.0)
    assembler = GlobalAssembler(
        (2.0, 1.0), (2, 1), (2, 2), E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    density = bm.full((8,), 0.5, dtype=bm.float64)
    density_cell = prototype.grid_to_cell_field(
        assembler.split_global_cell_field(density)
    )
    stiffness = prototype.assemble_local_stiffness_batch(density_cell)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    stiffness_s, _ = condensor.condense(stiffness)
    interpolation = prototype.linear_boundary_matrix
    stiffness_macro = (
        bm.matrix_transpose(interpolation)[None, :, :]
        @ stiffness_s
        @ interpolation[None, :, :]
    )
    system = assembler.assemble_macro_system(sub_meshes, stiffness_macro)
    force, fixed = project_problem_conditions_to_macro_system(problem, assembler)

    displacement = solve_interface_system(system, force, fixed)
    compliance = float(bm.dot(force, displacement))

    assert np.all(np.isfinite(bm.to_numpy(displacement)))
    np.testing.assert_allclose(
        bm.to_numpy(displacement[fixed]), 0.0, rtol=0.0, atol=1.0e-14
    )
    assert np.isfinite(compliance)
    assert compliance > 0.0


@pytest.mark.parametrize(
    ("case_ids", "problem_class"),
    [
        (("mbb_piml_route_a", "mbb_fea_baseline"), FullMBBBeam2d),
        (("mbb_3d_piml_route_a", "mbb_3d_fea_baseline"), FullMBBBeam3d),
    ],
)
def test_registered_piml_and_fea_cases_project_identical_system_data(
    case_ids: tuple[str, str],
    problem_class: type,
) -> None:
    """注册的 2D/3D PIML 与 FEA 路线必须共享 f 与 fixed dofs."""
    bm.set_backend("numpy")
    repository_root = Path(__file__).resolve().parents[2]
    cases_path = (
        repository_root / "experiments" / "piml_substructure_topopt" / "cases.toml"
    )
    with cases_path.open("rb") as stream:
        cases = {case["id"]: case for case in tomllib.load(stream)["cases"]}

    projected = []
    for case_id in case_ids:
        case = cases[case_id]
        domain = tuple(case["domain"])
        problem = problem_class(domain=domain, P=case["p_load"])
        domain_size = tuple(
            domain[2 * d + 1] - domain[2 * d] for d in range(problem.dimension)
        )
        assembler = GlobalAssembler(
            domain_size, tuple(case["n_sub"]), tuple(case["n_fine"])
        )
        projected.append(
            project_problem_conditions_to_macro_system(problem, assembler)
        )

    np.testing.assert_array_equal(
        bm.to_numpy(projected[0][0]), bm.to_numpy(projected[1][0])
    )
    np.testing.assert_array_equal(
        bm.to_numpy(projected[0][1]), bm.to_numpy(projected[1][1])
    )
