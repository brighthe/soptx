"""隐式 Schur 与限制预条件子的独立代数验收."""

import numpy as np
import pytest
from scipy.linalg import block_diag

from soptx.fem.substructure.implicit import (
    InteriorBlockSolver, ImplicitSchurOperator, RestrictedPreconditioner,
)


def make_problem():
    """构造已知 Schur 补的两个内部块与两个自由接口."""
    blocks = [np.array([[3., .4], [.4, 2.]]),
              np.array([[2., -.3], [-.3, 4.]])]
    d = block_diag(*blocks)
    b = np.array([[.2, -.3], [.4, .1], [-.2, .5], [.3, .4]])
    expected = np.array([[2., .2], [.2, 1.5]])
    g = expected + b.T @ np.linalg.solve(d, b)
    full = block_diag(np.ones((1, 1)), np.block([[g, b.T], [b, d]]))
    interior = InteriorBlockSolver(np.array([[3, 4], [5, 6]]), 7)
    interior.factor_batch(0, np.stack(blocks))
    schur = ImplicitSchurOperator(full, interior, np.array([1, 2]), np.array([0]))
    return full, schur, expected


def test_schur_action_and_recovery_match_independent_block_formula():
    """核对作用、内部平衡与全场直接解, 不以实现自身作为参考."""
    full, schur, expected = make_problem()
    q = np.array([.7, -.2])
    np.testing.assert_allclose(schur @ q, expected @ q, rtol=1e-13, atol=1e-14)
    u = schur.recover(q)
    np.testing.assert_allclose((full @ u)[3:], 0, atol=1e-14)
    force = np.zeros(7)
    force[1:3] = expected @ q
    np.testing.assert_allclose(u, np.linalg.solve(full, force), rtol=1e-13, atol=1e-14)
    assert u[0] == 0


def test_restricted_exact_inverse_is_schur_inverse():
    """精确全场逆的接口主块必须等于 Schur 逆."""
    full, schur, expected = make_problem()
    restricted = RestrictedPreconditioner(np.linalg.inv(full), schur.free_interface, 7)
    rhs = np.array([.2, -.6])
    np.testing.assert_allclose(restricted @ rhs, np.linalg.solve(expected, rhs),
                               rtol=1e-13, atol=1e-14)


def test_restricted_spd_preconditioner_preserves_symmetry_and_positivity():
    """完整空间 SPD 预条件子的接口限制仍为 SPD."""
    _, schur, _ = make_problem()
    rng = np.random.default_rng(8)
    a = rng.normal(size=(7, 7))
    b = a.T @ a + np.eye(7)
    restricted = RestrictedPreconditioner(b, schur.free_interface, 7)
    columns = np.column_stack([restricted @ e for e in np.eye(2)])
    np.testing.assert_allclose(columns, columns.T, atol=1e-14)
    assert np.linalg.eigvalsh(columns).min() > 0


def test_rejects_overlap_and_incomplete_factorization():
    """拒绝重叠编号与尚未完成的分解缓存."""
    with pytest.raises(ValueError, match="重叠"):
        InteriorBlockSolver(np.array([[1, 2], [2, 3]]), 5)
    interior = InteriorBlockSolver(np.array([[1, 2]]), 4)
    with pytest.raises(RuntimeError, match="尚未"):
        interior.solve_into(np.ones(4), np.zeros(4))
    interior.factor_batch(0, np.eye(2)[None])
    with pytest.raises(ValueError, match="划分"):
        ImplicitSchurOperator(np.eye(4), interior, np.array([0, 1]), np.array([], dtype=int))


def test_rejects_nonpositive_internal_block():
    """不能用非正定内部块进行精确静力缩聚."""
    interior = InteriorBlockSolver(np.array([[0, 1]]), 3)
    with pytest.raises(np.linalg.LinAlgError):
        interior.factor_batch(0, np.array([[[1., 0.], [0., -1.]]]))


def test_hex_schur_mg_matches_independent_full_assembly():
    """在独立全矩阵装配基准上核对六面体网格映射、MG 和恢复场."""
    from soptx.backend import backend_manager as bm
    from soptx.fem.kernels import ElementRestriction
    from soptx.fem.levels import SharedReferenceElementAssembly
    from soptx.fem.matrix import build_csr_pattern, assemble_csr
    from soptx.fem.multigrid import StructuredHexHierarchy
    from soptx.fem.operators import ConstrainedOperator
    from soptx.fem.substructure import (
        StructuredSubstructureLayout, GlobalAssembler, InterfaceDofsView,
        build_substructures, project_problem_conditions_to_interface_system,
    )
    from soptx.problems.elasticity import FullMBBBeam3d
    from soptx.solvers import CGSolver

    previous_backend = bm.backend_name
    bm.set_backend("numpy")
    mg = None
    try:
        layout = StructuredSubstructureLayout(
            domain_size=(4., 2., 2.), n_sub=(2, 1, 1), n_fine=(2, 2, 2),
            E_base=1., nu=.3, hypothesis="3D")
        assembler = GlobalAssembler(layout)
        prototype, meshes, positions = build_substructures(
            assembler, integration_order=2, penal=3., rho_min=1e-7)
        rho = np.random.default_rng(31).uniform(.2, .9, (4, 2, 2))
        interface = layout.build_interface_dofs(meshes)
        problem = FullMBBBeam3d(domain=(0., 4., 0., 2., 0., 2.),
                               E=1., nu=.3, P=-1., support="end_lines",
                               load_subdivisions=(4, 2))
        conditions = project_problem_conditions_to_interface_system(
            problem, assembler, InterfaceDofsView(global_dofs=interface))
        fixed = np.asarray(conditions.full_fixed_dofs)
        free_interface = np.setdiff1d(interface, fixed)
        space = layout.space_full
        n = layout.total_full_dofs
        mapping = np.stack([layout.get_substructure_global_dofs(pos, mesh)[prototype.i_dofs]
                            for pos, mesh in zip(positions, meshes)])
        interior = InteriorBlockSolver(mapping, n)
        local_rho = layout.split_global_cell_field(rho.reshape(-1))
        for begin, end, local in prototype.iter_local_stiffness_batches(local_rho, chunk_size=1):
            interior.factor_batch(begin, local[:, prototype.i_dofs[:, None], prototype.i_dofs])
        coef = 1e-7 + (1-1e-7) * rho.reshape(-1)**3
        ke = prototype.KE_unit[:1]
        mask = np.zeros(n, dtype=bool)
        mask[fixed] = True
        ea = SharedReferenceElementAssembly(
            space, ElementRestriction(space.cell_to_dof(), n), ke, scale=coef)
        full_operator = ConstrainedOperator(ea, isDDof=mask)
        schur = ImplicitSchurOperator(full_operator, interior, free_interface, fixed)

        pattern = build_csr_pattern(space)
        matrix = assemble_csr(np.broadcast_to(ke, (rho.size, 24, 24)), pattern,
                              scale=coef).to_scipy().toarray()
        matrix[fixed, :] = 0
        matrix[:, fixed] = 0
        matrix[fixed, fixed] = 1
        force = np.asarray(conditions.full_force).copy()
        force[fixed] = 0
        expected = np.linalg.solve(matrix, force)

        hierarchy = StructuredHexHierarchy(space, mask, ke[0], coarse_max_dofs=30)
        hierarchy.update(coef)
        mg = hierarchy.build_multigrid()
        mg.setup(full_operator)
        preconditioner = RestrictedPreconditioner(mg, free_interface, n)
        rhs = force[free_interface]
        q, info = CGSolver(M=preconditioner, atol=1e-10*np.linalg.norm(rhs),
                           rtol=0., maxit=1000, norm_type="unpreconditioned").setup(schur).solve(rhs)
        recovered = schur.recover(q)
        assert info["converged"]
        assert np.linalg.norm(matrix @ recovered-force)/np.linalg.norm(force) < 1e-9
        np.testing.assert_allclose(recovered, expected, rtol=1e-7, atol=1e-8)
    finally:
        if mg is not None:
            mg.coarse_solver.close()
        bm.set_backend(previous_backend)
