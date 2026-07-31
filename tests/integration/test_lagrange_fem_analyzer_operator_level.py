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


class TransparentWrapper:
    """Stands in for distributed.OverlapOperator: forwards without changing it."""

    def __init__(self, form) -> None:
        self.form = form
        self.matvec_calls = 0

    def __matmul__(self, vector):
        self.matvec_calls += 1
        return self.form @ vector

    def __getattr__(self, name):
        return getattr(self.form, name)


class SeamRecordingAnalyzer(LagrangeFEMAnalyzer):
    """Overrides the two distributed extension points, keeping them identities."""

    wrapped = None
    reduced_loads = 0

    def wrap_operator(self, form):
        self.wrapped = TransparentWrapper(form)
        return self.wrapped

    def reduce_load(self, F):
        self.reduced_loads += 1
        return F


def test_distributed_seams_are_used_by_the_ea_path() -> None:
    """A distributed subclass must be able to reach both extension points."""

    reference = make_analyzer(8, "ea", "cg")
    baseline = reference.solve_state()["displacement"]

    problem = SinusoidalPlaneStrainElasticity2D()
    mesh = TriangleMesh.from_box(list(problem.domain), nx=8, ny=8)
    analyzer = SeamRecordingAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=IsotropicLinearElasticMaterial(
            youngs_modulus=problem.E,
            poisson_ratio=problem.nu,
            hypothesis="plane_strain",
            device=bm.get_device(mesh),
        ),
        space_degree=DEGREE,
        integration_order=INTEGRATION_ORDER,
        operator_level="ea",
        solve_method="cg",
        topopt_algorithm=None,
    )
    solution = analyzer.solve_state()["displacement"]

    assert analyzer.reduced_loads == 1
    assert analyzer.wrapped is not None
    # 包装必须真的参与 matvec, 而不是被 DirichletBCOperator 绕过
    assert analyzer.wrapped.matvec_calls > 0
    # 恒等包装不得改变结果
    assert relative_difference(solution[:], baseline[:]) < 1.0e-12


def test_solve_system_does_not_depend_on_apply_bc_state() -> None:
    """The initial guess is an argument, not state left behind by apply_bc.

    Called with no x0 the solve must still land on the same displacement; the
    boundary rows are identities, so the guess only affects iteration count.
    """

    analyzer = make_analyzer(8, "ea", "cg")
    operator, load = analyzer.apply_bc(
        analyzer.assemble_stiff_matrix(),
        analyzer.assemble_body_force_vector(),
    )

    from_zero = bm.zeros_like(load)
    analyzer.solve_system(operator, load, from_zero, rtol=1.0e-12, atol=1.0e-14)

    from_prescribed = bm.zeros_like(load)
    analyzer.solve_system(
        operator, load, from_prescribed,
        x0=analyzer.prescribed_solution,
        rtol=1.0e-12, atol=1.0e-14,
    )

    assert relative_difference(from_zero, from_prescribed) < 1.0e-8


def test_dof_comm_is_readable_from_the_base_class() -> None:
    """A distributed subclass must not have to re-expose what it was given."""

    communicator = FakeDofComm(mpi_size=2)
    analyzer = make_analyzer(4, "ea", "cg")
    assert analyzer.dof_comm is None

    analyzer._dof_comm = communicator
    assert analyzer.dof_comm is communicator


def test_cg_reports_solver_diagnostics() -> None:
    """The example's numerical gates are built on these fields."""

    analyzer = make_analyzer(8, "ea", "cg")
    info = analyzer.solve_state(rtol=1.0e-10, atol=1.0e-12, maxiter=1000)["solver"]

    assert info["name"] == "cg"
    assert info["converged"] is True
    assert 0 < info["niter"] < info["maxit"]
    assert info["recursive_residual"] > 0.0


def test_a_direct_solve_does_not_fabricate_iteration_counts() -> None:
    analyzer = make_analyzer(8, "fa", "scipy")
    info = analyzer.solve_state()["solver"]

    assert info["name"] == "scipy"
    assert "niter" not in info


class FakeDofComm:
    """Just enough of EntityMPI for the rank-count guards."""

    def __init__(self, mpi_size: int) -> None:
        self.mpi_size = mpi_size


def test_fa_refuses_a_multi_rank_system() -> None:
    """Symmetric elimination has no seam for the overlap reduction.

    Without this guard a multi-rank FA run would not fail -- each rank would
    quietly solve its own local matrix.
    """

    analyzer = make_analyzer(4, "fa", "scipy")
    analyzer._dof_comm = FakeDofComm(mpi_size=2)

    with pytest.raises(RuntimeError, match="ea"):
        analyzer.apply_bc(
            analyzer.assemble_stiff_matrix(),
            analyzer.assemble_body_force_vector(),
        )


def test_fa_accepts_a_single_rank_communicator() -> None:
    """Overlap reduction is the identity on one rank, so FA stays usable.

    The example builds a dof_comm even for one rank, and its FA reference runs
    through this path.
    """

    analyzer = make_analyzer(4, "fa", "scipy")
    analyzer._dof_comm = FakeDofComm(mpi_size=1)

    matrix, load = analyzer.apply_bc(
        analyzer.assemble_stiff_matrix(),
        analyzer.assemble_body_force_vector(),
    )

    assert matrix is not None
    assert load is not None


def test_ea_accepts_a_multi_rank_system_at_the_boundary_step() -> None:
    """The EA path must stay open: it is the supported distributed level."""

    analyzer = make_analyzer(4, "ea", "cg")
    analyzer._dof_comm = FakeDofComm(mpi_size=2)

    operator, load = analyzer.apply_bc(
        analyzer.assemble_stiff_matrix(),
        analyzer.assemble_body_force_vector(),
    )

    assert operator is not None
    assert load is not None


def test_a_serial_solver_refuses_a_distributed_system() -> None:
    """dof_comm must be rejected by the solver, not by solve_state."""

    analyzer = make_analyzer(4, "ea", "cg")
    analyzer._dof_comm = object()

    with pytest.raises(NotImplementedError, match="solve_system"):
        analyzer.solve_state()


def test_overriding_the_solver_lifts_the_distributed_guard() -> None:
    """A distributed subclass supplies its own solver and must not be blocked."""

    class OwnSolverAnalyzer(LagrangeFEMAnalyzer):
        def solve_system(self, K, F, out, **kwargs):
            from fealpy.solver import cg

            out[:], _ = cg(K, F[:], x0=self._prescribed_solution,
                           batch_first=False, atol=1e-12, rtol=1e-12,
                           maxit=1000, returninfo=True)
            return out, {"name": "overridden"}

    reference = make_analyzer(8, "ea", "cg")
    baseline = reference.solve_state()["displacement"]

    problem = SinusoidalPlaneStrainElasticity2D()
    mesh = TriangleMesh.from_box(list(problem.domain), nx=8, ny=8)
    analyzer = OwnSolverAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=IsotropicLinearElasticMaterial(
            youngs_modulus=problem.E,
            poisson_ratio=problem.nu,
            hypothesis="plane_strain",
            device=bm.get_device(mesh),
        ),
        space_degree=DEGREE,
        integration_order=INTEGRATION_ORDER,
        operator_level="ea",
        solve_method="cg",
        dof_comm=object(),
        topopt_algorithm=None,
    )
    state = analyzer.solve_state()

    assert state["solver"]["name"] == "overridden"
    assert relative_difference(state["displacement"][:], baseline[:]) < 1.0e-12
