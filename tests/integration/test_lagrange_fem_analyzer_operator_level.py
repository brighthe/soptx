"""FA / EA operator levels of ``LagrangeFEMAnalyzer``.

``operator_level='fa'`` assembles a global sparse matrix; ``'ea'`` keeps the
element matrices and applies them one cell at a time.  Both describe the same
discrete operator ``K = sum_e R_e^T K_e R_e``, so every quantity below must
agree to round-off.
"""

from __future__ import annotations

from math import log

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TriangleMesh

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import SinusoidalPlaneStrainElasticity2D

DEGREE = 1
INTEGRATION_ORDER = DEGREE + 3


def make_analyzer(
    resolution: int,
    operator_level: str,
    solve_method: str,
    tensor_space=None,
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
    if tensor_space == "external":
        scalar_space = LagrangeFESpace(mesh, p=DEGREE, ctype="C")
        tensor_space = TensorFunctionSpace(
            scalar_space=scalar_space,
            shape=(-1, mesh.geo_dimension()),
        )
    return LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=DEGREE,
        integration_order=INTEGRATION_ORDER,
        assembly_method="standard",
        operator_level=operator_level,
        solve_method=solve_method,
        tensor_space=tensor_space,
        topopt_algorithm=None,
    )


def random_vector(analyzer: LagrangeFEMAnalyzer, seed: int = 2026):
    gdof = analyzer.tensor_space.number_of_global_dofs()
    generator = np.random.default_rng(seed)
    return bm.asarray(generator.standard_normal(gdof))


def relative_difference(left, right) -> float:
    return float(bm.max(bm.abs(left - right))) / float(bm.max(bm.abs(right)))


def relative_l2_error(analyzer: LagrangeFEMAnalyzer, solution) -> float:
    problem = analyzer.pde

    def zero_field(points):
        return bm.zeros_like(problem.disp_solution(points))

    absolute = float(
        analyzer.disp_mesh.error(
            problem.disp_solution,
            solution,
            q=INTEGRATION_ORDER,
        )
    )
    exact_norm = float(
        analyzer.disp_mesh.error(
            problem.disp_solution,
            zero_field,
            q=INTEGRATION_ORDER,
        )
    )
    return absolute / exact_norm


def test_raw_matvec_agrees_between_operator_levels() -> None:
    """Before boundary conditions the two levels are the same operator."""

    fa = make_analyzer(6, "fa", "scipy")
    ea = make_analyzer(6, "ea", "cg")

    matrix = fa.assemble_stiff_matrix()
    operator = ea.assemble_stiff_matrix()
    vector = random_vector(fa)

    assert relative_difference(operator @ vector, matrix.matmul(vector)) < 1.0e-12


def test_dirichlet_matvec_agrees_between_operator_levels() -> None:
    """Symmetric elimination and DirichletBCOperator define one system."""

    fa = make_analyzer(6, "fa", "scipy")
    ea = make_analyzer(6, "ea", "cg")

    fa_matrix, fa_load = fa.apply_bc(
        fa.assemble_stiff_matrix(),
        fa.assemble_body_force_vector(),
    )
    ea_operator, ea_load = ea.apply_bc(
        ea.assemble_stiff_matrix(),
        ea.assemble_body_force_vector(),
    )
    vector = random_vector(fa)

    assert relative_difference(ea_load, fa_load) < 1.0e-12
    assert (
        relative_difference(ea_operator @ vector, fa_matrix.matmul(vector))
        < 1.0e-12
    )


def test_solutions_agree_between_operator_levels() -> None:
    """EA with CG reproduces the FA direct solution."""

    fa = make_analyzer(8, "fa", "scipy")
    ea = make_analyzer(8, "ea", "cg")

    fa_solution = fa.solve_state()["displacement"]
    ea_solution = ea.solve_state()["displacement"]

    assert relative_difference(ea_solution[:], fa_solution[:]) < 1.0e-8


def test_ea_converges_on_the_manufactured_solution() -> None:
    """The matrix-free path keeps the expected P1 convergence order."""

    errors = []
    for resolution in (8, 16):
        analyzer = make_analyzer(resolution, "ea", "cg")
        state = analyzer.solve_state()
        errors.append(relative_l2_error(analyzer, state["displacement"]))

    assert errors[1] < errors[0]
    assert log(errors[0] / errors[1]) / log(2.0) >= 1.5


def test_ea_rejects_a_direct_solver() -> None:
    """There is no factorizable matrix in EA, so this must fail loudly."""

    analyzer = make_analyzer(4, "ea", "mumps")

    with pytest.raises(RuntimeError, match="ea"):
        analyzer.solve_state()


def test_ea_rejects_an_adjoint_right_hand_side() -> None:
    """The batched adjoint right-hand side is FA-only for now."""

    analyzer = make_analyzer(4, "ea", "cg")

    with pytest.raises(RuntimeError, match="fa"):
        analyzer.apply_bc(
            analyzer.assemble_stiff_matrix(),
            analyzer.assemble_body_force_vector(),
            adjoint=True,
        )


def test_an_externally_built_space_gives_the_same_solution() -> None:
    """The injection seam kept for distributed spaces must be transparent."""

    internal = make_analyzer(8, "ea", "cg")
    external = make_analyzer(8, "ea", "cg", tensor_space="external")

    internal_solution = internal.solve_state()["displacement"]
    external_solution = external.solve_state()["displacement"]

    assert relative_difference(external_solution[:], internal_solution[:]) < 1.0e-12


def test_an_unknown_operator_level_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="算子层级"):
        make_analyzer(4, "eba", "cg")
