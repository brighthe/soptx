"""The Hu-Zhang analyzer must drive maintained Problems without an adapter.

This is the end-to-end counterpart to the structural checks in
``tests/unit/test_problem_protocol_conformance.py``: it exercises the real
solve path rather than attribute presence.
"""

from __future__ import annotations

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem import HuZhangMFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    ExponentialSineManufacturedElasticity2D,
    MixedBoundaryExponentialSineElasticity2D,
)


def _solve_coarsest_state(problem) -> HuZhangMFEMAnalyzer:
    bm.set_backend("numpy")
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    mesh = TriangleMesh.from_box(box=list(problem.domain), nx=2, ny=2)
    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=3,
        integration_order=8,
        use_relaxation=False,
        solve_method="scipy",
        topopt_algorithm=None,
    )
    analyzer.solve_state(rho_val=None)
    return analyzer


def test_all_dirichlet_problem_needs_no_adapter() -> None:
    analyzer = _solve_coarsest_state(
        ExponentialSineManufacturedElasticity2D()
    )

    assert analyzer.relative_state_residual() <= 1.0e-8
    assert analyzer.state_matrix_symmetry_error() <= 1.0e-12


def test_mixed_boundary_problem_needs_no_adapter() -> None:
    analyzer = _solve_coarsest_state(
        MixedBoundaryExponentialSineElasticity2D()
    )

    assert analyzer.relative_state_residual() <= 1.0e-8
    assert analyzer.state_matrix_symmetry_error() <= 1.0e-12
