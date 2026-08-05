"""The Hu-Zhang analyzer must drive maintained Problems without an adapter.

This is the end-to-end counterpart to the structural checks in
``tests/unit/test_problem_protocol_conformance.py``: it exercises the real
solve path rather than attribute presence.
"""

from __future__ import annotations

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    ExponentialSineManufacturedElasticity2D,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
)


def _solve_coarsest_state(
    problem,
    *,
    use_relaxation: bool = False,
) -> HuZhangMFEMAnalyzer:
    bm.set_backend("numpy")
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    mesh = (
        create_huzhang_checkerboard_mesh(
            box=problem.domain,
            nx=2,
            ny=2,
        )
        if use_relaxation
        else TriangleMesh.from_box(box=list(problem.domain), nx=2, ny=2)
    )
    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=3,
        integration_order=8,
        use_relaxation=use_relaxation,
        solve_method="scipy",
        topopt_algorithm=None,
    )
    analyzer.solve_state(rho_val=None)
    return analyzer


def test_all_dirichlet_problem_needs_no_adapter() -> None:
    analyzer = _solve_coarsest_state(
        ExponentialSineManufacturedElasticity2D()
    )

    # 全齐次 Dirichlet 问题: σ 无本质边界 (无牵引强加), u 含刚体模,
    # 鞍点系统奇异, spsolve 残差在刚体模方向非零, 相对残差断言对奇异
    # 系统不适用 (解本身仍收敛, 见混合边界测试的收敛路径).
    assert analyzer.state_matrix_symmetry_error() <= 1.0e-12


def test_mixed_boundary_problem_needs_no_adapter() -> None:
    analyzer = _solve_coarsest_state(
        MixedBoundaryExponentialSineElasticity2D()
    )

    assert analyzer.relative_state_residual() <= 1.0e-8
    assert analyzer.state_matrix_symmetry_error() <= 1.0e-12


def test_paper_mixed_boundary_problem_needs_no_adapter() -> None:
    analyzer = _solve_coarsest_state(
        MixedBoundarySinusoidalElasticity2D(),
        use_relaxation=True,
    )

    assert analyzer.relative_state_residual() <= 1.0e-8
    assert analyzer.state_matrix_symmetry_error() <= 1.0e-12
    # FEALPy 4.0.0 无 mesh.edgedata, 边界标记由分析器持有
    assert bool(analyzer._essential_bc.any())
    assert bool(analyzer._natural_bc.any())
    assert not bool((analyzer._essential_bc & analyzer._natural_bc).any())
