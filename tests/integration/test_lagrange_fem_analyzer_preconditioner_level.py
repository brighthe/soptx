"""``preconditioner_level`` of ``LagrangeFEMAnalyzer``.

``operator_level`` fixes how the main operator is stored; ``preconditioner_level``
fixes how the preconditioner's operator is stored.  They are independent: a
matrix-free main operator may be preconditioned from a fully assembled matrix,
which is the only way a ``CAP_MATRIX`` backend (a direct solver here, algebraic
multigrid later) can reach a ``'pa'`` system at all.

The direct solver used as a preconditioner is an exact inverse, so CG must stop
after one iteration.  That doubles as the strongest available check that the two
levels really are the same discrete operator after boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import SinusoidalPlaneStrainElasticity2D

DEGREE = 1
INTEGRATION_ORDER = DEGREE + 3
RESOLUTION = 8


def make_analyzer(
    operator_level: str,
    solve_method: str = "cg",
    preconditioner_level: str | None = None,
    resolution: int = RESOLUTION,
) -> LagrangeFEMAnalyzer:
    problem = SinusoidalPlaneStrainElasticity2D()
    mesh = TriangleMesh.from_box(
        list(problem.domain),
        nx=resolution,
        ny=resolution,
    )
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis="plane_strain",
        device=bm.get_device(mesh),
    )
    return LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=DEGREE,
        integration_order=INTEGRATION_ORDER,
        assembly_method="standard",
        operator_level=operator_level,
        preconditioner_level=preconditioner_level,
        solve_method=solve_method,
        topopt_algorithm=None,
    )


def relative_difference(left, right) -> float:
    return float(bm.max(bm.abs(left - right))) / float(bm.max(bm.abs(right)))


CG_OPTIONS = {"rtol": 1.0e-12, "atol": 1.0e-14, "maxiter": 2000}


# --------------------------------------------------------------------------
# the axis exists and defaults to the old behaviour
# --------------------------------------------------------------------------
def test_the_axis_defaults_to_the_main_operator() -> None:
    """Leaving it unset must not perturb the existing jacobi path."""

    without = make_analyzer("pa").solve_state(precond="jacobi", **CG_OPTIONS)
    explicit_none = make_analyzer("pa", preconditioner_level=None).solve_state(
        precond="jacobi", **CG_OPTIONS
    )

    assert explicit_none["solver"]["niter"] == without["solver"]["niter"]
    assert explicit_none["solver"]["relres"] == without["solver"]["relres"]
    assert relative_difference(
        explicit_none["displacement"][:], without["displacement"][:]
    ) == 0.0


def test_the_level_is_reported() -> None:
    assert make_analyzer("pa").preconditioner_level is None
    assert make_analyzer("pa", preconditioner_level="fa").preconditioner_level == "fa"


def test_an_unknown_preconditioner_level_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="预条件子层级"):
        make_analyzer("pa", preconditioner_level="fla")


# --------------------------------------------------------------------------
# a CAP_MATRIX preconditioner under a matrix-free main operator
# --------------------------------------------------------------------------
def test_a_direct_preconditioner_reaches_a_matrix_free_operator() -> None:
    """'pa' main operator, 'fa' preconditioner: CG stops after one iteration.

    Without the axis this combination is unreachable -- ``DirectSolver`` needs an
    explicit matrix and the 'pa' operator has none, so ``solve_system`` used to
    refuse the whole solve.
    """

    analyzer = make_analyzer("pa", preconditioner_level="fa")
    state = analyzer.solve_state(precond="scipy", **CG_OPTIONS)

    assert state["solver"]["converged"] is True
    assert state["solver"]["niter"] == 1

    reference = make_analyzer("fa", solve_method="scipy").solve_state()
    assert (
        relative_difference(state["displacement"][:], reference["displacement"][:])
        < 1.0e-10
    )


def test_the_direct_preconditioner_also_works_under_ea() -> None:
    """The axis is about storage form, so 'ea' behaves exactly like 'pa'."""

    state = make_analyzer("ea", preconditioner_level="fa").solve_state(
        precond="scipy", **CG_OPTIONS
    )

    assert state["solver"]["niter"] == 1
    assert state["solver"]["converged"] is True


def test_a_matrix_free_preconditioner_level_still_refuses_a_direct_backend() -> None:
    """Capability negotiation must bite on the preconditioner side too."""

    analyzer = make_analyzer("pa", preconditioner_level="pa")

    with pytest.raises(RuntimeError, match="preconditioner_level='pa'"):
        analyzer.solve_state(precond="scipy", **CG_OPTIONS)


# --------------------------------------------------------------------------
# the second level describes the same system as the main operator
# --------------------------------------------------------------------------
def test_jacobi_is_unchanged_when_sourced_from_another_level() -> None:
    """The diagonal of the 'fa' system equals the diagonal of the 'pa' system."""

    sourced = make_analyzer("pa", preconditioner_level="fa").solve_state(
        precond="jacobi", **CG_OPTIONS
    )
    inherent = make_analyzer("pa").solve_state(precond="jacobi", **CG_OPTIONS)

    assert sourced["solver"]["niter"] == inherent["solver"]["niter"]
    assert (
        relative_difference(
            sourced["displacement"][:], inherent["displacement"][:]
        )
        < 1.0e-10
    )


def test_the_preconditioner_operator_carries_the_boundary_conditions() -> None:
    """Dirichlet rows must be eliminated, else the factorization sees a singular
    matrix and the diagonal picks up zeros."""

    from soptx.solvers import DirectSolver, operator_diagonal

    analyzer = make_analyzer("pa", preconditioner_level="fa")
    analyzer.assemble_stiff_matrix()
    operator = analyzer._preconditioner_operator()

    assert np.all(bm.to_numpy(operator_diagonal(operator)) > 0.0)

    # a singular matrix would surface here rather than as a silently wrong solve
    DirectSolver(backend="scipy").setup(operator).close()
