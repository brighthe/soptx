"""Poisson-ratio interpolation path of ``LagrangeFEMAnalyzer``.

For nearly incompressible materials the interpolation scheme returns
``(E(rho), nu(rho))`` and the element stiffness is no longer a scalar multiple
of the solid one.  The analyzer then hands a per-element constitutive matrix to
``LinearElasticIntegrator.assembly('standard')`` and differentiates through the
Lame split ``K_e = lambda_e K_e^lambda + mu_e K_e^mu``.  These tests pin down:

1. the E-only path is bitwise unchanged;
2. ``nu_void = nu_0`` degenerates to the E-only stiffness;
3. the Lame split reproduces the assembled element matrices;
4. the manual sensitivity matches a central finite difference of the
   compliance ``f^T u`` on the bearing problem;
5. ``rho = 1`` reproduces the plain (non-topology) stiffness matrix;
6. unsupported assembly methods are rejected instead of silently ignoring nu.
"""

from __future__ import annotations

import numpy as np
import pytest
from fealpy.backend import backend_manager as bm

from soptx.fem import LagrangeFEMAnalyzer, create_huzhang_checkerboard_mesh
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import BearingDevice2d
from soptx.topology.interpolation import MaterialInterpolationScheme

NX, NY = 12, 4
ORDER = 2
INTEGRATION_ORDER = 2 * ORDER + 2
E0 = 1.0
E_MIN = 1.0e-9
P_E = 3.0
NU_VOID = 0.3


def make_problem(nu: float, plane_type: str = "plane_strain") -> BearingDevice2d:
    return BearingDevice2d(t=-8.0e-2, E=E0, nu=nu, plane_type=plane_type)


def make_mesh(problem: BearingDevice2d):
    return create_huzhang_checkerboard_mesh(box=problem.domain, nx=NX, ny=NY)


def make_material(problem: BearingDevice2d) -> IsotropicLinearElasticMaterial:
    return IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )


def make_interpolation(target_variables, nu_void: float = NU_VOID) -> MaterialInterpolationScheme:
    return MaterialInterpolationScheme(
        density_location="element",
        interpolation_method="msimp",
        options={
            "penalty_factor": P_E,
            "void_youngs_modulus": E_MIN,
            "target_variables": list(target_variables),
            "nu_penalty_factor": 1.0,
            "void_poisson_ratio": nu_void,
        },
        enable_logging=False,
    )


def make_analyzer(
    nu: float,
    target_variables=("E", "nu"),
    nu_void: float = NU_VOID,
    plane_type: str = "plane_strain",
    assembly_method: str = "standard",
    topopt: bool = True,
) -> LagrangeFEMAnalyzer:
    problem = make_problem(nu, plane_type)
    mesh = make_mesh(problem)
    material = make_material(problem)
    kwargs = dict(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=ORDER,
        integration_order=INTEGRATION_ORDER,
        assembly_method=assembly_method,
        solve_method="scipy",
    )
    if topopt:
        kwargs.update(
            topopt_algorithm="density_based",
            interpolation_scheme=make_interpolation(target_variables, nu_void),
        )
    else:
        kwargs.update(topopt_algorithm=None)
    return LagrangeFEMAnalyzer(**kwargs)


def random_density(analyzer: LagrangeFEMAnalyzer, seed: int = 7):
    NC = analyzer.disp_mesh.number_of_cells()
    rng = np.random.default_rng(seed)
    rho = 0.3 + 0.6 * rng.random(NC)
    return bm.tensor(rho, dtype=bm.float64)


def dense_stiffness(analyzer: LagrangeFEMAnalyzer, rho):
    return bm.to_numpy(analyzer.assemble_stiff_matrix(rho_val=rho).to_dense())


def relative_difference(a, b) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def compliance(analyzer: LagrangeFEMAnalyzer, rho) -> float:
    state = analyzer.solve_state(rho_val=rho)
    uh = bm.to_numpy(state["displacement"][:])
    F = bm.to_numpy(analyzer.force_vector[:])
    return float(uh @ F)


# ---------------------------------------------------------------- 1. E-only path


def test_e_only_path_is_bitwise_unchanged() -> None:
    """Compressible material: 'nu' in target_variables must be a no-op."""
    reference = make_analyzer(nu=0.3, target_variables=("E",))
    candidate = make_analyzer(nu=0.3, target_variables=("E", "nu"))
    rho = random_density(reference)

    K_ref = dense_stiffness(reference, rho)
    K_new = dense_stiffness(candidate, rho)

    assert candidate.poisson_ratio_interpolated is False
    assert candidate._integrator.coef.shape == (reference.disp_mesh.number_of_cells(),)
    assert np.array_equal(K_ref, K_new)

    dK_ref = bm.to_numpy(reference.compute_stiffness_matrix_derivative(rho_val=rho))
    dK_new = bm.to_numpy(candidate.compute_stiffness_matrix_derivative(rho_val=rho))
    assert np.array_equal(dK_ref, dK_new)


# ---------------------------------------------------------- 2. degenerate nu_void


def test_nu_void_equal_to_nu0_degenerates_to_e_only() -> None:
    """nu(rho) == nu_0 for all rho: per-element D path must equal the scalar path."""
    nu0 = 0.4999
    reference = make_analyzer(nu=nu0, target_variables=("E",))
    candidate = make_analyzer(nu=nu0, target_variables=("E", "nu"), nu_void=nu0)
    rho = random_density(reference)

    K_ref = dense_stiffness(reference, rho)
    K_new = dense_stiffness(candidate, rho)

    assert candidate.poisson_ratio_interpolated is True
    assert relative_difference(K_new, K_ref) < 1.0e-12

    dK_ref = bm.to_numpy(reference.compute_stiffness_matrix_derivative(rho_val=rho))
    dK_new = bm.to_numpy(candidate.compute_stiffness_matrix_derivative(rho_val=rho))
    assert relative_difference(dK_new, dK_ref) < 1.0e-12


# ------------------------------------------------------------ 3. Lame split


@pytest.mark.parametrize("plane_type", ["plane_strain", "plane_stress"])
def test_lame_split_reproduces_element_matrices(plane_type: str) -> None:
    analyzer = make_analyzer(nu=0.4999, plane_type=plane_type)
    rho = random_density(analyzer)

    analyzer.assemble_stiff_matrix(rho_val=rho)
    KE = bm.to_numpy(analyzer._integrator.assembly(space=analyzer.tensor_space))

    E_rho = analyzer._cached_stiffness_absolute
    nu_rho = analyzer._cached_nu_rho
    lam, mu = analyzer._lame_parameters(E_rho, nu_rho)[:2]
    ke_lam, ke_mu = analyzer.compute_lame_basis_matrices()
    KE_split = (
        np.einsum("c, cij -> cij", bm.to_numpy(lam), bm.to_numpy(ke_lam))
        + np.einsum("c, cij -> cij", bm.to_numpy(mu), bm.to_numpy(ke_mu))
    )
    assert relative_difference(KE_split, KE) < 1.0e-12

    # the per-element constitutive matrix must agree with the material class at rho = 1
    ones = bm.ones_like(rho)
    analyzer.assemble_stiff_matrix(rho_val=ones)
    D_rho1 = bm.to_numpy(analyzer._integrator.coef)
    D0 = bm.to_numpy(analyzer.material.elastic_matrix()[0, 0])
    assert relative_difference(D_rho1, np.broadcast_to(D0, D_rho1.shape)) < 1.0e-12


# ------------------------------------------------------- 4. finite difference


@pytest.mark.parametrize("plane_type", ["plane_strain", "plane_stress"])
def test_sensitivity_matches_central_finite_difference(plane_type: str) -> None:
    analyzer = make_analyzer(nu=0.4999, plane_type=plane_type)
    rho = random_density(analyzer)

    state = analyzer.solve_state(rho_val=rho)
    uh = state["displacement"][:]
    cell2dof = analyzer.tensor_space.cell_to_dof()
    uhe = uh[cell2dof]
    dKE = analyzer.compute_stiffness_matrix_derivative(rho_val=rho)
    dc = bm.to_numpy(-bm.einsum("ci, cij, cj -> c", uhe, dKE, uhe))

    NC = analyzer.disp_mesh.number_of_cells()
    rng = np.random.default_rng(11)
    cells = rng.choice(NC, size=6, replace=False)
    h = 1.0e-6
    rho_np = bm.to_numpy(rho).copy()
    for e in cells:
        rho_p = rho_np.copy()
        rho_p[e] += h
        rho_m = rho_np.copy()
        rho_m[e] -= h
        c_p = compliance(analyzer, bm.tensor(rho_p, dtype=bm.float64))
        c_m = compliance(analyzer, bm.tensor(rho_m, dtype=bm.float64))
        fd = (c_p - c_m) / (2.0 * h)
        assert abs(fd - dc[e]) <= 1.0e-5 * max(abs(fd), abs(dc[e])), (e, fd, dc[e])


# ------------------------------------------------------------ 5. rho = 1


def test_full_density_matches_standard_fem() -> None:
    topopt = make_analyzer(nu=0.4999)
    plain = make_analyzer(nu=0.4999, topopt=False)
    ones = bm.ones((topopt.disp_mesh.number_of_cells(),), dtype=bm.float64)

    K_topopt = dense_stiffness(topopt, ones)
    K_plain = bm.to_numpy(plain.assemble_stiff_matrix().to_dense())
    assert relative_difference(K_topopt, K_plain) < 1.0e-12


# ---------------------------------------------------------- 6. guarded combos


@pytest.mark.parametrize("assembly_method", ["voigt", "fast"])
def test_unsupported_assembly_method_is_rejected(assembly_method: str) -> None:
    analyzer = make_analyzer(nu=0.4999, assembly_method=assembly_method)
    rho = random_density(analyzer)
    with pytest.raises(RuntimeError):
        analyzer.assemble_stiff_matrix(rho_val=rho)
