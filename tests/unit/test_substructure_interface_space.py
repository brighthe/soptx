"""接口空间 ``InterfaceSpace`` 与既有逐种类 API 的一致性测试."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    ExactSchurReduction,
    FullTraceBasis,
    GlobalAssembler,
    InterfaceDofsView,
    InterfaceSpace,
    LinearCornerTraceBasis,
    TraceBasis,
    build_interface_space,
    build_substructures,
    build_interface_pattern,
    iter_exact_trace_stiffness_batches,
    project_problem_conditions_to_interface_system,
    project_problem_conditions_to_macro_system,
    solve_constrained_system,
    solve_interface_system,
)
from soptx.problems.elasticity import FullMBBBeam2d


def _components():
    """3x1 子结构, 每块 2x2 细单元的二维 MBB 工况."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        domain_size=(3.0, 1.0),
        n_sub=(3, 1),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    prototype.penal = 3.0
    prototype.rho_min = 1.0e-3
    problem = FullMBBBeam2d(domain=(0.0, 3.0, 0.0, 1.0), P=-1.0)
    density = np.linspace(0.3, 0.95, 12).reshape(3, 2, 2)
    return assembler, prototype, sub_meshes, problem, density


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
def test_local_dofs_match_layout_numbering(kind: str) -> None:
    """``local_dofs`` 应与布局中对应种类的编号方法逐位一致."""
    assembler, prototype, sub_meshes, _, _ = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)

    if kind == "linear_corner":
        expected_local = assembler.macro_corner_indices(sub_meshes)
        expected_global = bm.arange(assembler.total_macro_dofs, dtype=bm.int64)
        assert isinstance(space.trace_basis, LinearCornerTraceBasis)
    else:
        expected_global = assembler.build_interface_dofs(sub_meshes)
        expected_local = assembler.interface_indices(sub_meshes, expected_global)
        assert isinstance(space.trace_basis, FullTraceBasis)

    assert space.name == kind
    assert space.n_substructures == len(sub_meshes)
    assert space.n_global == int(len(expected_global))
    assert space.n_trace_dofs == space.trace_basis.n_trace_dofs
    assert space.n_boundary_dofs == int(prototype.n_b)
    np.testing.assert_array_equal(
        bm.to_numpy(space.local_dofs), bm.to_numpy(expected_local)
    )
    np.testing.assert_array_equal(
        bm.to_numpy(space.global_dofs), bm.to_numpy(expected_global)
    )


def test_numbering_conventions_are_node_first() -> None:
    """两种空间的 ``local_dofs`` 列序都应为节点优先: dof = dim * node + k."""
    assembler, prototype, sub_meshes, _, _ = _components()
    for kind in ("full_trace", "linear_corner"):
        space = build_interface_space(kind, assembler, sub_meshes, prototype)
        local = bm.to_numpy(space.local_dofs).reshape(len(sub_meshes), -1, 2)
        assert np.all(local[:, :, 0] % 2 == 0), kind
        np.testing.assert_array_equal(local[:, :, 1], local[:, :, 0] + 1)


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_assemble_matches_kind_specific_entry(kind: str, chunk_size: int) -> None:
    """``space.assemble`` 应与对应种类的逐块装配入口给出同一系统."""
    assembler, prototype, sub_meshes, _, density = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)

    def batches():
        return iter_exact_trace_stiffness_batches(
            prototype, density, space.trace_basis, chunk_size=chunk_size,
        )

    actual = space.assemble(batches())
    if kind == "linear_corner":
        expected = assembler.assemble_macro_system_batches(sub_meshes, batches())
    else:
        expected = assembler.assemble_interface_system_batches(
            sub_meshes, batches()
        )

    np.testing.assert_array_equal(
        bm.to_numpy(actual.global_dofs), bm.to_numpy(expected.global_dofs)
    )
    np.testing.assert_allclose(
        bm.to_numpy(actual.stiffness.to_dense()),
        bm.to_numpy(expected.stiffness.to_dense()),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_full_trace_assembly_matches_legacy_batch_assembly() -> None:
    """``full_trace`` 的流式装配应与整批 ``assemble_interface_system`` 一致."""
    assembler, prototype, sub_meshes, _, density = _components()
    space = build_interface_space("full_trace", assembler, sub_meshes, prototype)

    local_full = prototype.assemble_local_stiffness_batch(density)
    reduced = ExactSchurReduction(
        prototype.i_dofs, prototype.b_dofs
    ).reduce_many(local_full)
    expected = assembler.assemble_interface_system(sub_meshes, reduced)
    actual = space.assemble(
        iter_exact_trace_stiffness_batches(
            prototype, density, space.trace_basis, chunk_size=2,
        )
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual.stiffness.to_dense()),
        bm.to_numpy(expected.stiffness.to_dense()),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
def test_project_conditions_match_kind_specific_functions(kind: str) -> None:
    """``project_conditions`` 应与对应种类的投影函数给出同一载荷与约束."""
    assembler, prototype, sub_meshes, problem, _ = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)

    load, fixed = space.project_conditions(problem)
    if kind == "linear_corner":
        expected_load, expected_fixed = (
            project_problem_conditions_to_macro_system(problem, assembler)
        )
    else:
        conditions = project_problem_conditions_to_interface_system(
            problem,
            assembler,
            InterfaceDofsView(global_dofs=space.global_dofs),
        )
        expected_load = conditions.interface_force
        expected_fixed = conditions.interface_fixed_dofs

    assert tuple(load.shape) == (space.n_global,)
    np.testing.assert_allclose(bm.to_numpy(load), bm.to_numpy(expected_load))
    np.testing.assert_array_equal(bm.to_numpy(fixed), bm.to_numpy(expected_fixed))
    assert len(fixed) > 0
    assert int(bm.max(fixed)) < space.n_global


def test_global_trace_map_matches_legacy_projection_and_compatibility() -> None:
    """``P_q`` 应与 ``build_linear_corner_projection`` 一致, 并满足式 (3.2)."""
    assembler, prototype, sub_meshes, _, _ = _components()
    corner = build_interface_space("linear_corner", assembler, sub_meshes, prototype)
    full = build_interface_space("full_trace", assembler, sub_meshes, prototype)

    P = corner.global_trace_map()
    expected = assembler.build_linear_corner_projection(
        sub_meshes,
        InterfaceDofsView(global_dofs=full.global_dofs),
        corner.trace_basis,
    )
    assert isinstance(P, csr_matrix)
    assert P.shape == (full.n_global, corner.n_global)
    np.testing.assert_allclose(P.toarray(), expected.toarray(), atol=1.0e-12)

    # 式 (3.2): A_b^j P_q == Psi^j A_q^j, 逐子结构核对.
    P_dense = P.toarray()
    Psi = bm.to_numpy(corner.trace_basis.matrix)
    A_b = bm.to_numpy(full.local_dofs)
    A_q = bm.to_numpy(corner.local_dofs)
    for j in range(corner.n_substructures):
        lhs = P_dense[A_b[j]]
        rhs = np.zeros((full.n_boundary_dofs, corner.n_global))
        rhs[:, A_q[j]] = Psi
        np.testing.assert_allclose(lhs, rhs, atol=1.0e-12)

    identity = full.global_trace_map()
    np.testing.assert_allclose(identity.toarray(), np.eye(full.n_global))


def test_displacement_extraction_follows_local_dofs() -> None:
    """``trace_displacement`` 与 ``boundary_displacement`` 应按 ``A_q^j`` 与 ``Psi`` 提取."""
    assembler, prototype, sub_meshes, _, _ = _components()
    space = build_interface_space("linear_corner", assembler, sub_meshes, prototype)
    Q = bm.asarray(np.arange(space.n_global, dtype=np.float64))

    q = space.trace_displacement(Q)
    assert tuple(q.shape) == (space.n_substructures, space.n_trace_dofs)
    np.testing.assert_array_equal(
        bm.to_numpy(q), bm.to_numpy(Q)[bm.to_numpy(space.local_dofs)]
    )

    u_b = space.boundary_displacement(Q)
    assert tuple(u_b.shape) == (space.n_substructures, space.n_boundary_dofs)
    np.testing.assert_allclose(
        bm.to_numpy(u_b),
        bm.to_numpy(q) @ bm.to_numpy(space.trace_basis.matrix).T,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="global_displacement 的形状"):
        space.trace_displacement(Q[:-1])


def test_factory_rejects_unknown_kind_and_unmapped_basis() -> None:
    """未知名称与无全局映射的迹基应被明确拒绝."""
    assembler, prototype, sub_meshes, _, _ = _components()
    with pytest.raises(ValueError, match="未知的接口空间"):
        build_interface_space("quadratic_edge", assembler, sub_meshes, prototype)

    custom = TraceBasis(bm.eye(int(prototype.n_b), dtype=bm.float64))
    with pytest.raises(TypeError, match="仅支持 FullTraceBasis 或 LinearCornerTraceBasis"):
        InterfaceSpace.from_trace_basis(custom, assembler, sub_meshes)


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
def test_interface_conditions_always_target_full_interface(kind: str) -> None:
    """``interface_conditions`` 在两种空间下都应给出完整接口上的条件."""
    assembler, prototype, sub_meshes, problem, _ = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)
    full = build_interface_space("full_trace", assembler, sub_meshes, prototype)

    force, fixed = space.interface_conditions(problem)
    expected_force, expected_fixed = full.project_conditions(problem)

    assert tuple(force.shape) == (full.n_global,)
    np.testing.assert_allclose(bm.to_numpy(force), bm.to_numpy(expected_force))
    np.testing.assert_array_equal(bm.to_numpy(fixed), bm.to_numpy(expected_fixed))


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
def test_constrained_conditions_follow_global_trace_map(kind: str) -> None:
    """``F_Q = P_q^T F_Gamma`` 与 ``C_D = P_q[D, :]`` 应逐位成立."""
    assembler, prototype, sub_meshes, problem, _ = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)

    force_gamma, fixed = space.interface_conditions(problem)
    load, constraints = space.constrained_conditions(problem)
    P = space.global_trace_map().toarray()
    force_np = bm.to_numpy(force_gamma)
    fixed_np = bm.to_numpy(fixed)

    assert tuple(load.shape) == (space.n_global,)
    assert isinstance(constraints, csr_matrix)
    assert constraints.shape == (len(fixed_np), space.n_global)
    np.testing.assert_allclose(bm.to_numpy(load), P.T @ force_np, atol=1.0e-12)
    np.testing.assert_allclose(constraints.toarray(), P[fixed_np], atol=1.0e-12)
    if kind == "full_trace":
        expected_rows = np.zeros((len(fixed_np), space.n_global))
        expected_rows[np.arange(len(fixed_np)), fixed_np] = 1.0
        np.testing.assert_array_equal(constraints.toarray(), expected_rows)
        np.testing.assert_allclose(bm.to_numpy(load), force_np)


def test_constrained_solve_matches_macro_projection_route() -> None:
    """``linear_corner`` 下文档通用约束形式与宏观节点投影路线应解出同一 ``Q``."""
    assembler, prototype, sub_meshes, problem, density = _components()
    space = build_interface_space("linear_corner", assembler, sub_meshes, prototype)
    system = space.assemble(
        iter_exact_trace_stiffness_batches(
            prototype, density, space.trace_basis, chunk_size=2,
        )
    )

    load_q, constraints = space.constrained_conditions(problem)
    solved = solve_constrained_system(system, load_q, constraints)

    load_macro, fixed_macro = space.project_conditions(problem)
    expected = solve_interface_system(system, load_macro, fixed_macro)

    assert solved.constraint_rank == len(bm.to_numpy(fixed_macro))
    np.testing.assert_allclose(bm.to_numpy(load_q), bm.to_numpy(load_macro), atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(bm.to_numpy(solved.displacement), dtype=np.float64),
        np.asarray(bm.to_numpy(expected), dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
def test_assemble_matches_dense_scatter_reference(kind: str) -> None:
    """通用 dofmap 组装应与逐子结构稠密散加的参考结果逐位一致, 且可复用模式反复组装."""
    assembler, prototype, sub_meshes, _, density = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)
    local = bm.to_numpy(space.local_dofs)

    def reference(rho):
        batches = list(
            iter_exact_trace_stiffness_batches(
                prototype, rho, space.trace_basis, chunk_size=1,
            )
        )
        K = np.zeros((space.n_global, space.n_global))
        for batch in batches:
            K_r = bm.to_numpy(batch.stiffness)[0]
            K[np.ix_(local[batch.start], local[batch.start])] += K_r
        return K

    first = space.assemble(
        iter_exact_trace_stiffness_batches(
            prototype, density, space.trace_basis, chunk_size=2,
        )
    )
    np.testing.assert_allclose(
        bm.to_numpy(first.stiffness.to_dense()), reference(density),
        rtol=1.0e-12, atol=1.0e-12,
    )

    # 密度更新后复用同一符号模式重新组装, 数值缓冲区相互独立.
    pattern = space.pattern
    density_2 = density[::-1].copy()
    second = space.assemble(
        iter_exact_trace_stiffness_batches(
            prototype, density_2, space.trace_basis, chunk_size=3,
        )
    )
    assert space.pattern is pattern
    np.testing.assert_allclose(
        bm.to_numpy(second.stiffness.to_dense()), reference(density_2),
        rtol=1.0e-12, atol=1.0e-12,
    )
    np.testing.assert_allclose(
        bm.to_numpy(first.stiffness.to_dense()), reference(density),
        rtol=1.0e-12, atol=1.0e-12,
    )


@pytest.mark.parametrize("kind", ["full_trace", "linear_corner"])
def test_node_first_pattern_matches_generic_pattern(kind: str) -> None:
    """按节点优先标量映射建的模式应与完整自由度映射建的模式给出同一矩阵."""
    assembler, prototype, sub_meshes, _, density = _components()
    space = build_interface_space(kind, assembler, sub_meshes, prototype)
    generic = build_interface_pattern(space.local_dofs, space.n_global, dof_numel=1)
    grouped = build_interface_pattern(space.local_dofs, space.n_global, dof_numel=2)
    assert grouped.nnz == generic.nnz
    assert space.pattern.dof_numel == 2

    def batches():
        return iter_exact_trace_stiffness_batches(
            prototype, density, space.trace_basis, chunk_size=2,
        )

    from soptx.fem.substructure import assemble_interface_stiffness
    K_generic = assemble_interface_stiffness(
        space.local_dofs, space.n_global, batches(), pattern=generic,
    )
    K_grouped = assemble_interface_stiffness(
        space.local_dofs, space.n_global, batches(), pattern=grouped,
    )
    np.testing.assert_allclose(
        bm.to_numpy(K_grouped.to_dense()), bm.to_numpy(K_generic.to_dense()),
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_node_first_pattern_rejects_component_major_mapping() -> None:
    """分量优先排列的映射不能按 ``dof_numel > 1`` 建模式."""
    local = bm.asarray([[0, 2, 1, 3], [2, 4, 3, 5]], dtype=bm.int64)
    with pytest.raises(ValueError, match="节点优先"):
        build_interface_pattern(local, 6, dof_numel=2)
    # 通用路径不受排列约束.
    assert build_interface_pattern(local, 6, dof_numel=1).nnz > 0
