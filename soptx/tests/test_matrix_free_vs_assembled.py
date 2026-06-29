import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TetrahedronMesh, TriangleMesh
from scipy.sparse.linalg import spsolve

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.analysis.matrix_free import MatrixFreeCGSolver, MatrixFreeElasticityOperator
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
from soptx.model.mbb_beam_2d_lfem import HalfMBBBeamRight2d


def test_matrix_free_matvec_matches_assembled_matrix():
    bm.set_backend("numpy")

    mesh = TriangleMesh.from_box(box=[0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    tensor_space = TensorFunctionSpace(scalar_space, shape=(-1, mesh.geo_dimension()))

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        plane_type="plane_stress",
    )
    integrator = LinearElasticIntegrator(material=material, q=4, method="standard")

    rho = bm.linspace(0.4, 1.0, mesh.number_of_cells())
    integrator.coef = rho

    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly(format="coo")
    integrator._disable_action_assembly_fallback = True

    op = MatrixFreeElasticityOperator(
        space=tensor_space,
        integrator=integrator,
        rho=rho,
    )

    x = bm.linspace(0.1, 1.0, tensor_space.number_of_global_dofs())
    y_ref = K.matmul(x)
    y_mf = op.matvec(x)

    rel_error = bm.linalg.norm(y_ref - y_mf) / bm.linalg.norm(y_ref)
    assert rel_error < 1.0e-12


def test_matrix_free_matvec_matches_assembled_matrix_3d():
    bm.set_backend("numpy")

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
        nz=1,
    )
    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    tensor_space = TensorFunctionSpace(scalar_space, shape=(-1, mesh.geo_dimension()))

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        plane_type="3d",
    )
    integrator = LinearElasticIntegrator(material=material, q=4, method="standard")

    coef = bm.linspace(0.4, 1.0, mesh.number_of_cells())
    integrator.coef = coef

    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly(format="coo")
    integrator._disable_action_assembly_fallback = True

    op = MatrixFreeElasticityOperator(
        space=tensor_space,
        integrator=integrator,
        rho=coef,
    )

    x = bm.linspace(0.1, 1.0, tensor_space.number_of_global_dofs())
    y_ref = K.matmul(x)
    y_mf = op.matvec(x)

    rel_error = bm.linalg.norm(y_ref - y_mf) / bm.linalg.norm(y_ref)
    assert rel_error < 1.0e-12


def test_matrix_free_cg_solution_matches_assembled_solution():
    bm.set_backend("numpy")

    mesh = TriangleMesh.from_box(box=[0.0, 1.0, 0.0, 1.0], nx=3, ny=2)
    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    tensor_space = TensorFunctionSpace(scalar_space, shape=(-1, mesh.geo_dimension()))

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        plane_type="plane_stress",
    )
    integrator = LinearElasticIntegrator(material=material, q=4, method="standard")

    coef = bm.linspace(0.4, 1.0, mesh.number_of_cells())
    integrator.coef = coef

    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly(format="coo")
    integrator._disable_action_assembly_fallback = True

    fixed_dofs = bm.where(
        tensor_space.is_boundary_dof(
            threshold=lambda p: bm.abs(p[..., 0]) < 1.0e-12,
            method="interp",
        )
    )[0]
    rhs = bm.linspace(0.2, 1.1, tensor_space.number_of_global_dofs())
    rhs = bm.set_at(rhs, fixed_dofs, 0.0)

    K_bc = K.to_scipy().tolil()
    fixed_dofs_np = np.asarray(fixed_dofs, dtype=int)
    K_bc[fixed_dofs_np, :] = 0.0
    K_bc[:, fixed_dofs_np] = 0.0
    K_bc[fixed_dofs_np, fixed_dofs_np] = 1.0
    u_ref = spsolve(K_bc.tocsr(), np.asarray(rhs))

    op = MatrixFreeElasticityOperator(
        space=tensor_space,
        integrator=integrator,
        rho=coef,
        dirichlet_dofs=fixed_dofs,
    )
    solver = MatrixFreeCGSolver(tol=1.0e-12, maxiter=200)
    u_mf, info = solver.solve(op, rhs)

    assert info.converged
    rel_error = bm.linalg.norm(u_ref - u_mf) / bm.linalg.norm(u_ref)
    assert rel_error < 1.0e-12
    assert bm.linalg.norm(u_mf[fixed_dofs]) < 1.0e-12


def test_lagrange_analyzer_matrix_free_state_matches_assembled_state():
    bm.set_backend("numpy")

    pde = HalfMBBBeamRight2d(
        domain=[0.0, 3.0, 0.0, 1.0],
        mesh_type="uniform_quad",
        P=-1.0,
        E=1.0,
        nu=0.3,
        plane_type="plane_stress",
    )
    mesh = pde.init_mesh(nx=3, ny=1)
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=pde.E,
        poisson_ratio=pde.nu,
        plane_type=pde.plane_type,
    )
    interpolation_scheme = MaterialInterpolationScheme(
        density_location="element",
        interpolation_method="simp",
        options={"penalty_factor": 1.0},
        enable_logging=False,
    )
    rho = bm.linspace(0.5, 1.0, mesh.number_of_cells())

    assembled_analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=pde,
        material=material,
        space_degree=1,
        integration_order=4,
        assembly_method="standard",
        solve_method="scipy",
        operator_backend="assembled",
        topopt_algorithm="density_based",
        interpolation_scheme=interpolation_scheme,
    )
    matrix_free_analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=pde,
        material=material,
        space_degree=1,
        integration_order=4,
        assembly_method="standard",
        solve_method="scipy",
        operator_backend="matrix_free",
        matrix_free_solver_options={"tol": 1.0e-12, "maxiter": 200},
        topopt_algorithm="density_based",
        interpolation_scheme=interpolation_scheme,
    )
    matrix_free_analyzer.assemble_stiff_matrix = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("matrix-free solve_state must not assemble K")
        )
    )
    matrix_free_analyzer._integrator._disable_action_assembly_fallback = True

    u_assembled = assembled_analyzer.solve_state(rho_val=rho)["displacement"]
    mf_state = matrix_free_analyzer.solve_state(rho_val=rho)
    u_matrix_free = mf_state["displacement"]

    assert mf_state["matrix_free_info"].converged
    rel_error = bm.linalg.norm(u_assembled - u_matrix_free) / bm.linalg.norm(u_assembled)
    assert rel_error < 1.0e-10
