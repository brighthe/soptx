"""Standard-FEM (non-topology-optimization) path of ``LagrangeFEMAnalyzer``.

``topopt_algorithm=None`` means no material interpolation: the integrator
uses the solid constitutive matrix directly.  These tests guard that path,
which is the one consumed by plain linear-elasticity analysis rather than by
density-based optimization.
"""

from __future__ import annotations

from math import log

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import SinusoidalPlaneStrainElasticity2D

DEGREE = 1
INTEGRATION_ORDER = DEGREE + 3


def make_analyzer(resolution: int) -> LagrangeFEMAnalyzer:
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
        solve_method="scipy",
        topopt_algorithm=None,
    )


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


def test_standard_mode_assembles_without_density() -> None:
    """Assembly must not require an interpolated density coefficient."""

    analyzer = make_analyzer(resolution=2)
    matrix = analyzer.assemble_stiff_matrix().to_dense()

    gdof = analyzer.tensor_space.number_of_global_dofs()
    assert matrix.shape == (gdof, gdof)
    assert float(bm.max(bm.abs(matrix - matrix.T))) < 1.0e-14
    assert float(bm.max(bm.abs(matrix))) > 0.0


def test_standard_mode_ignores_a_supplied_density() -> None:
    """A density argument is warned about and left out of the operator."""

    analyzer = make_analyzer(resolution=2)
    number_of_cells = analyzer.disp_mesh.number_of_cells()
    density = 0.5 * bm.ones(number_of_cells, dtype=bm.float64)

    without_density = analyzer.assemble_stiff_matrix().to_dense()
    with_density = analyzer.assemble_stiff_matrix(rho_val=density).to_dense()

    difference = float(bm.max(bm.abs(with_density - without_density)))
    assert difference < 1.0e-14


def test_standard_mode_converges_on_the_manufactured_solution() -> None:
    """P1 displacement error must decrease at close to second order."""

    errors = []
    for resolution in (8, 16):
        analyzer = make_analyzer(resolution)
        state = analyzer.solve_state()
        errors.append(relative_l2_error(analyzer, state["displacement"]))

    assert errors[1] < errors[0]
    observed_order = log(errors[0] / errors[1]) / log(2.0)
    assert observed_order >= 1.5
