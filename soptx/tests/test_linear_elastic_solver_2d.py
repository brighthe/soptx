"""
此文件用于测试在线弹性问题中, 针对不同 PDE 和不同 mesh 下, 位移计算结果的正确性
"""
from fealpy.backend import backend_manager as bm

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.solver import cg, spsolve
from fealpy.decorator import cartesian

from soptx.material import DensityBasedMaterialConfig, DensityBasedMaterialInstance
from soptx.solver import ElasticFEMSolver, AssemblyMethod

def test_elastic_with_interpolated_rhs_solver(pde, mesh, p, solver_type):
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, 2))
    print(f"CGDOF: {tensor_space_C.number_of_global_dofs()}")
    space_D = LagrangeFESpace(mesh=mesh, p=0, ctype='D')
    print(f"DGDOF: {space_D.number_of_global_dofs()}")

    material_config = DensityBasedMaterialConfig(
                            elastic_modulus=1,            
                            minimal_modulus=1e-9,         
                            poisson_ratio=0.3,            
                            plane_type="plane_stress",    
                            interpolation_model="SIMP",    
                            penalty_factor=3
                        )
    materials = DensityBasedMaterialInstance(config=material_config)

    solvers_top = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.STANDARD,
                    solver_type='direct',
                    solver_params={'solver_type': 'mumps'}, 
                )

    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x):
        val = bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)

    KK_top = solvers_top.get_base_local_stiffness_matrix()

    solvers_top.update_status(rho[:])
    solver_result = solvers_top.solve()
    uh = solver_result.displacement

    return uh

def test_elastic_with_interpolated_rhs(pde, mesh, p, solver_type):
    """
    该函数适用于右端项 F 可以通过直接插值得到的情况,
        通常用于问题没有解析真解的场景
    """
    from fealpy.fem.dirichlet_bc import DirichletBC
    from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 2))

    linear_elastic_material = LinearElasticMaterial(name='E1nu03', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_stress', 
                                                device=bm.get_device(mesh))

    integrator_K = LinearElasticIntegrator(
                        material=linear_elastic_material, 
                        q=tensor_space.p+3, 
                        method=None)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    K = bform.assembly(format='csr')

    F = tensor_space.interpolate(pde.force)

    # ? 什么时候函数，什么时候数组
    dbc = DirichletBC(space=tensor_space, 
                gd=pde.dirichlet, 
                threshold=pde.threshold(), 
                method='interp')
    K, F = dbc.apply(A=K, f=F[:], uh=None, gd=pde.dirichlet, check=True)

    uh = tensor_space.function()

    if solver_type == 'mumps':
        uh[:] = spsolve(K, F, solver='mumps')
    elif solver_type == 'cg':
        uh[:], info = cg(K, F[:], x0=None,
            batch_first=True, 
            atol=1e-12, rtol=1e-12, 
            maxit=5000, returninfo=True)
        
    return uh

def test_elastic_with_linearform_rhs(pde, mesh, p, solver_type):
    """
    该函数适用于右端项 F 需要通过 LinearForm 进行组装的情况, 
        通常用于问题有精确解, 可以进行误差分析的场景
    """
    from fealpy.fem.linear_form import LinearForm
    from fealpy.fem.dirichlet_bc import DirichletBC
    from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
    from fealpy.fem.vector_source_integrator import VectorSourceIntegrator

    from pathlib import Path
    current_file = Path(__file__)
    base_dir = current_file.parent.parent / 'vtu'
    base_dir = str(base_dir)
    maxit = 4
    errorType = ['$|| u  - u_h ||_{l2}$']
    errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
    NDof = bm.zeros(maxit, dtype=bm.int32)
    print(f"NN:{mesh.number_of_nodes()}")
    for i in range(maxit):
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
        NDof[i] = tensor_space.number_of_global_dofs()
        print(f"gdof:{NDof[i]}")

        linear_elastic_material = LinearElasticMaterial(name='E1nu03', 
                                                    elastic_modulus=1, poisson_ratio=0.3, 
                                                    hypo='plane_strain', 
                                                    device=bm.get_device(mesh))

        integrator_K = LinearElasticIntegrator(
                            material=linear_elastic_material, 
                            q=tensor_space.p+3, 
                            method=None)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
        integrator_F = VectorSourceIntegrator(
                            source=pde.source, 
                            q=tensor_space.p+3
                        )
        lform = LinearForm(tensor_space)    
        lform.add_integrator(integrator_F)
        F = lform.assembly()

        dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
        K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet, check=True)

        uh = tensor_space.function()

        if solver_type == 'mumps':
            uh[:] = spsolve(K, F, solver='mumps')
        elif solver_type == 'cg':
            uh[:], info = cg(K, F[:], x0=None,
                batch_first=True, 
                atol=1e-12, rtol=1e-12, 
                maxit=5000, returninfo=True)

        u_exact = tensor_space.interpolate(pde.solution)
        errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))

        if i < maxit-1:
            mesh.uniform_refine()

    print("errorMatrix:\n", errorType, "\n", errorMatrix)
    print("NDof:", NDof)
    print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))


if __name__ == "__main__":
    from soptx.pde import PolyDisp2dData, TriDisp2dData
    from fealpy.mesh import TriangleMesh, QuadrangleMesh
    p = 3
    # pde = PolyDisp2dData()
    pde_linearform_rhs = TriDisp2dData()
    domain_x, domain_y = pde_linearform_rhs.domain()[1], pde_linearform_rhs.domain()[3]
    nx, ny = 10, 5
    # mesh = UniformMesh2d(extent=[0, nx, 0, ny], h=[domain_x/nx, domain_y/ny], origin=[0.0, 0.0])
    # cip = mesh.cell_to_ipoint(p=p)
    # mesh = TriangleMesh.from_box(box=[0, domain_x, 0, domain_y], nx=nx, ny=ny)
    mesh = QuadrangleMesh.from_box(box=[0, domain_x, 0, domain_y], nx=nx, ny=ny)

    # test_elastic_with_linearform_rhs(pde_linearform_rhs, mesh, p, solver_type='mumps')

    from soptx.pde import HalfMBBBeam2dData1
    domain_x, domain_y = 60, 20
    nx, ny = 60, 20
    pde_interpolated_rhs = HalfMBBBeam2dData1(xmin=0, xmax=domain_x, ymin=0, ymax=domain_y, T=1)
    mesh = QuadrangleMesh.from_box(box=[0, domain_x, 0, domain_y], nx=nx, ny=ny)
    uh1 = test_elastic_with_interpolated_rhs(pde_interpolated_rhs, mesh, p, solver_type='mumps')
    uh2 = test_elastic_with_interpolated_rhs_solver(pde_interpolated_rhs, mesh, p, solver_type='mumps')
    print(f"error{bm.sum(uh1 - uh2)}")
