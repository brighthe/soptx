import matplotlib.pyplot as plt
from typing import Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike
# TODO 这里先导入网格再导入空间（这是由于网格的 init 中导入了 RadiusRatioSumObjective, 
# TODO 而 RadiusRatioSumObjective 里面调用了 solver）
from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from soptx.utils.show import showmultirate, show_error_table
from soptx.utils.base_logged import BaseLogged
from soptx.analysis.utils import project_solution_to_finer_mesh


class LagrangeFEMAnalyzerTest(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.pde = None
        self.mesh = None
        self.material = None
        self.space_degree = 1
        self.integrator_order = 4
        self.assembly_method = 'standard'
        self.solve_method = 'mumps'

        self.topopt_algorithm = None
        self.topopt_config = None


    def set_pde(self, pde):
        self.pde = pde

    def set_init_mesh(self, meshtype: str, **kwargs):
        self.pde.init_mesh.set(meshtype)
        self.mesh = self.pde.init_mesh(**kwargs)

    def set_material(self, material):
        self.material = material

    def set_space_degree(self, space_degree: int):
        self.space_degree = space_degree

    def set_integrator_order(self, integrator_order: int):
        self.integrator_order = integrator_order

    def set_assembly_method(self, method: str):
        self.assembly_method = method

    def set_solve_method(self, method: str):
        self.solve_method = method



    @variantmethod('lfa_exact_solution')
    def run(self, model_type: str = 'BoxPoly3d') -> TensorLike:

        if model_type == 'BoxTri2d':
            from soptx.model.linear_elasticity_2d import BoxTriLagrange2dData
            domain = [0, 1, 0, 1]
            E, nu = 1.0, 0.3
            pde = BoxTriLagrange2dData(domain=domain, E=E, nu=nu)
            nx, ny = 4, 4
            mesh_type = 'uniform_quad'
            pde.init_mesh.set(mesh_type)
            mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                youngs_modulus=pde.E, 
                                                poisson_ratio=pde.nu, 
                                                plane_type=pde.plane_type,
                                            )

        elif model_type == 'BoxPoly3d':
            from soptx.model.linear_elasticity_3d import BoxPolyLagrange3dData
            domain = [0, 1, 0, 1, 0, 1]
            lam, mu = 1.0, 1.0
            nx, ny, nz = 4, 4, 4
            pde = BoxPolyLagrange3dData(domain=domain, lam=lam, mu=mu)
            mesh_type = 'uniform_tet'
            pde.init_mesh.set(mesh_type)
            mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam,
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                            )

        space_degree = 1
        integration_order = space_degree + 3
        # 'standard', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        # s_space = LagrangeFESpace(mesh=mesh, p=space_degree, ctype='C')
        # GD = mesh.geo_dimension()
        # t_space = TensorFunctionSpace(scalar_space=s_space, shape=(GD, -1))
        # from soptx.analysis.integrators.linear_elastic_integrator import LinearElasticIntegrator
        # lei_standard = LinearElasticIntegrator(material=material, coef=None, q=integration_order, method='standard')
        # KE_standard = lei_standard.assembly(space=t_space)

        # lei_voigt = LinearElasticIntegrator(material=material, coef=None, q=integration_order, method='voigt')
        # KE_voigt = lei_voigt.assembly(space=t_space)
        

        # from fealpy.material.elastic_material import LinearElasticMaterial
        # material_fealpy = LinearElasticMaterial(name='test', elastic_modulus=E, poisson_ratio=nu, hypo=pde.plane_type)
        
        # from fealpy.fem.linear_elasticity_integrator import LinearElasticityIntegrator
        # lei_standard_fealpy = LinearElasticityIntegrator(material=material_fealpy, q=integration_order, method='standard')
        # KE_standard_fealpy = lei_standard_fealpy.assembly(space=t_space)

        # lei_voigt_fealpy = LinearElasticityIntegrator(material=material_fealpy, q=integration_order, method='voigt')
        # KE_voigt_fealpy = lei_voigt_fealpy.assembly(space=t_space)

        # error = bm.sum(bm.abs(KE_standard - KE_voigt))

        maxit = 5
        errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"第 {i+1}/{maxit} 次迭代...")

            lfa = LagrangeFEMAnalyzer(
                                    mesh=mesh,
                                    pde=pde, 
                                    material=material, 
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method='mumps')
                    
            uh = lfa.solve_displacement()

            mesh = lfa.mesh
            e0 = mesh.error(uh, pde.disp_solution)
            errorMatrix[0, i] = e0

            NDof[i] = lfa.tensor_space.number_of_global_dofs()
            
            initial_hx = mesh.meshdata.get('hx') if i == 0 else h[0]
            h[i] = initial_hx / (2 ** i)

            if i < maxit - 1:
                mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return self.uh
    

    @run.register('lfa_test_assembly')
    def run(self):
        if model_type == 'BoxTri2d':
            from soptx.model.linear_elasticity_2d import BoxTriLagrange2dData
            domain = [0, 1, 0, 1]
            E, nu = 1.0, 0.3
            pde = BoxTriLagrange2dData(domain=domain, E=E, nu=nu)
            nx, ny = 4, 4
            mesh_type = 'uniform_quad'
            pde.init_mesh.set(mesh_type)
            mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                youngs_modulus=pde.E, 
                                                poisson_ratio=pde.nu, 
                                                plane_type=pde.plane_type,
                                            )

        elif model_type == 'BoxPoly3d':
            from soptx.model.linear_elasticity_3d import BoxPolyLagrange3dData
            domain = [0, 1, 0, 1, 0, 1]
            lam, mu = 1.0, 1.0
            nx, ny, nz = 4, 4, 4
            pde = BoxPolyLagrange3dData(domain=domain, lam=lam, mu=mu)
            mesh_type = 'uniform_tet'
            pde.init_mesh.set(mesh_type)
            mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam,
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                            )

        space_degree = 1
        integration_order = space_degree + 3
        # 'standard', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        # s_space = LagrangeFESpace(mesh=mesh, p=space_degree, ctype='C')
        # GD = mesh.geo_dimension()
        # t_space = TensorFunctionSpace(scalar_space=s_space, shape=(GD, -1))
        # from soptx.analysis.integrators.linear_elastic_integrator import LinearElasticIntegrator
        # lei_standard = LinearElasticIntegrator(material=material, coef=None, q=integration_order, method='standard')
        # KE_standard = lei_standard.assembly(space=t_space)

        # lei_voigt = LinearElasticIntegrator(material=material, coef=None, q=integration_order, method='voigt')
        # KE_voigt = lei_voigt.assembly(space=t_space)
        

        # from fealpy.material.elastic_material import LinearElasticMaterial
        # material_fealpy = LinearElasticMaterial(name='test', elastic_modulus=E, poisson_ratio=nu, hypo=pde.plane_type)
        
        # from fealpy.fem.linear_elasticity_integrator import LinearElasticityIntegrator
        # lei_standard_fealpy = LinearElasticityIntegrator(material=material_fealpy, q=integration_order, method='standard')
        # KE_standard_fealpy = lei_standard_fealpy.assembly(space=t_space)

        # lei_voigt_fealpy = LinearElasticityIntegrator(material=material_fealpy, q=integration_order, method='voigt')
        # KE_voigt_fealpy = lei_voigt_fealpy.assembly(space=t_space)

        # error = bm.sum(bm.abs(KE_standard - KE_voigt))

    @run.register('lfa_analysis_exact_solution')
    def run(self) -> TensorLike:
        """使用 lfa 验证 HuZhang 算例的精确解"""

        from soptx.model.linear_elasticity_3d import BoxPolyHuZhangData3d, BoxPolyLagrange3dData
        # pde = BoxPolyHuZhangData3d(lam=1, mu=0.5)
        pde = BoxPolyLagrange3dData(lam=1, mu=1)
        pde.init_mesh.set('uniform_tet')
        nx, ny, nz = 2, 2, 2
        analysis_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)
        space_degree = 4

        integration_order = space_degree + 3

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        maxit = 2
        errorType = ['$|| \\boldsymbol{u}_h - \\boldsymbol{u} ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            N = 2**(i+1) 

            lfa = LagrangeFEMAnalyzer(
                                    mesh=analysis_mesh, 
                                    pde=pde, 
                                    material=material, 
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard', 
                                    solve_method='mumps',
                                    topopt_algorithm=None,
                                    interpolation_scheme=None
                                )

            NDof[i] = lfa.tensor_space.number_of_global_dofs()
            uh = lfa.solve_displacement(density_distribution=None)

            e0 = analysis_mesh.error(uh, pde.disp_solution)

            h[i] = 1/N

            errorMatrix[0, i] = e0

            analysis_mesh.uniform_refine()

        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table

        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()
        print('------------------')

        return uh
    
    @run.register('lfa_analysis_reference_solution')
    def run(self, maxit: int = 7, ref_level: int = 7) -> TensorLike:
        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, 6, 0, 2],
                            T=-1.0, 
                            E=1.0, nu=0.3,
                            enable_logging=False
                        )
        
        nx, ny = 6, 2
        pde.init_mesh.set('uniform_tri')
        mesh_fe = pde.init_mesh(nx=nx, ny=ny)

        mesh_fe.to_vtk(f'initial_mesh.vtu')

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        # 设置参考网格, 求解参考真解
        ref_mesh_fe = pde.init_mesh(nx=nx, ny=ny)
        self._log_info(f"Initial mesh with {ref_mesh_fe.number_of_cells()} cells.")

        total_refinement = maxit - 1 + ref_level
        for _ in range(total_refinement):
            ref_mesh_fe.bisect(isMarkedCell=None, options={'disp': False})
        self._log_info(f"Reference mesh_fe with {ref_mesh_fe.number_of_cells()} cells.")

        ref_mesh_fe.to_vtk(f'reference_mesh.vtu')

        lfa_ref = LagrangeFEMAnalyzer(
                            mesh=ref_mesh_fe, 
                            pde=pde, 
                            material=material, 
                            space_degree=self.space_degree,
                            integrator_order=self.integrator_order,
                            assembly_method=self.assembly_method, 
                            solve_method=self.solve_method,
                            topopt_algorithm=None,
                            interpolation_scheme=None,
                            enable_logging=False
                        )
        
        uh_ref = lfa_ref.solve_displacement(density_distribution=None)

        errorType = ['$|| \\boldsymbol{u}_h - \\boldsymbol{u}_{ref} ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"Solving on mesh level {i+1}/{maxit}...")

            mesh_i_fe = pde.init_mesh(nx=nx, ny=ny)            
            for _ in range(i):
                mesh_i_fe.bisect(isMarkedCell=None, options={'disp': False})

            mesh_i_fe.to_vtk(f'mesh_level_{i+1}.vtu')

            lfa_i = LagrangeFEMAnalyzer(
                                    mesh=mesh_i_fe, 
                                    pde=pde, 
                                    material=material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=None,
                                    interpolation_scheme=None,
                                    enable_logging=False
                                )

            uh_i = lfa_i.solve_displacement(density_distribution=None)

            uh_i_projected = project_solution_to_finer_mesh(
                                                pde=pde,
                                                nx=nx, ny=ny,
                                                uh=uh_i, 
                                                lfa=lfa_i, 
                                                source_refinement_level=i, 
                                                target_mesh=ref_mesh_fe
                                            )

            e0 = ref_mesh_fe.error(uh_i_projected, uh_ref)
            errorMatrix[0, i] = e0
             
            NDof[i] = lfa_i.tensor_space.number_of_global_dofs()
            
        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        order_l2 = bm.log2(errorMatrix[0, :-2] / errorMatrix[0, 2:])
        self._log_info(f"order_l2: {order_l2}")
        # show_error_table(h, errorType, errorMatrix)
        # showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        # plt.show()

        return uh_ref


    @run.register('test_bisect')
    def run(self, maxit: int = 6, ref_level: int = 4) -> TensorLike:
        """测试二分加密后的数值解充当真解的有效性"""
        # 设置 pde
        from soptx.model.linear_elasticity_2d import BoxTriLagrange2dData
        pde = BoxTriLagrange2dData(
                            domain=[0, 1, 0, 1], 
                            E=1.0, nu=0.3,
                            enable_logging=False
                        )
        nx, ny = 10, 10
        pde.init_mesh.set('uniform_tri')

        # mesh.to_vtk(f'initial_mesh.vtu')

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        base_material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

        # 设置拓扑优化配置
        interpolation_config = InterpolationConfig(
                                    method='msimp',
                                    penalty_factor=3,
                                    target_variables=['E'],
                                    void_youngs_modulus=1e-12
                                )
        density_based_config = DensityBasedConfig(
                                    density_location='element',
                                    initial_density=1.0,
                                    interpolation=interpolation_config
                                )
        self.set_topopt_config(
                        algorithm='density_based',
                        config=density_based_config
                    )

        # 设置参考网格, 求解参考真解
        ref_mesh = pde.init_mesh(nx=nx, ny=ny)
        self._log_info(f"Initial mesh with {ref_mesh.number_of_cells()} cells.")


        total_refinement = maxit - 1 + ref_level
        for _ in range(total_refinement):
            ref_mesh.bisect(isMarkedCell=None, options={'disp': False})
        self._log_info(f"Reference mesh with {ref_mesh.number_of_cells()} cells.")

        lfa_ref = LagrangeFEMAnalyzer(mesh=ref_mesh, 
                                    pde=pde, 
                                    material=base_material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=self.topopt_algorithm,
                                    topopt_config=self.topopt_config,
                                    enable_logging=True)
        uh_ref = lfa_ref.solve()

        errorType = [
            '$|| \\boldsymbol{u}_h - \\boldsymbol{u}_{ref} ||_{L^2}$',
            '$|| \\boldsymbol{u}_h - \\boldsymbol{u}_{ref} ||_{H^1}$'
        ]
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)

        for i in range(maxit):
            print(f"Solving on mesh level {i+1}/{maxit}...")

            mesh_i = pde.init_mesh(nx=nx, ny=ny)
            for _ in range(i):
                mesh_i.bisect(isMarkedCell=None, options={'disp': False})

            mesh_i.to_vtk(f'mesh_level_{i+1}.vtu')

            lfa_i = LagrangeFEMAnalyzer(
                                    mesh=mesh_i, 
                                    pde=pde, 
                                    material=base_material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=self.topopt_algorithm,
                                    topopt_config=self.topopt_config
                                )
            uh_i = lfa_i.solve()

            uh_i_projected = project_solution_to_finer_mesh(
                                                pde=pde,
                                                nx=nx, ny=ny,
                                                uh=uh_i, 
                                                lfa=lfa_i, 
                                                source_refinement_level=i, 
                                                target_mesh=ref_mesh
                                            )
            
            e0 = ref_mesh.error(uh_i_projected, uh_ref)
            e1 = ref_mesh.error(uh_i_projected, pde.disp_solution)
            errorMatrix[0, i] = e0
            errorMatrix[1, i] = e1
             
            NDof[i] = lfa_i.tensor_space.number_of_global_dofs()
            
        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        order_l2 = bm.log2(errorMatrix[0, :-2] / errorMatrix[0, 2:])
        self._log_info(f"order_l2: {order_l2}")
        order_l2_exact = bm.log2(errorMatrix[1, :-2] / errorMatrix[1, 2:])
        self._log_info(f"order_l2_exact: {order_l2_exact}")

        return uh_ref
    

    @run.register('topopt_analysis_reference_solution')
    def run(self, maxit: int = 5, ref_level: int = 5) -> TensorLike:

        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, 30, 0, 10],
                            T=-1.0, 
                            E=1.0, nu=0.3,
                            enable_logging=False
                        )
        
        nx, ny = 30, 10
        pde.init_mesh.set('uniform_tri')

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

        # 设置插值方案
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location='element_coscos',
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-12,
                                        'target_variables': ['E']
                                    },
                                )
        
        # 设置参考网格, 求解参考真解
        ref_mesh_fe = pde.init_mesh(nx=nx, ny=ny)
        self._log_info(f"Initial mesh with {ref_mesh_fe.number_of_cells()} cells.")

        total_refinement = maxit - 1 + ref_level
        for _ in range(total_refinement):
            ref_mesh_fe.bisect(isMarkedCell=None, options={'disp': False})
        self._log_info(f"Reference mesh_fe with {ref_mesh_fe.number_of_cells()} cells.")

        rho_ref = interpolation_scheme.setup_density_distribution(
                                mesh=ref_mesh_fe,
                                relative_density=0.5,
                            )

        lfa_ref = LagrangeFEMAnalyzer(
                                    mesh=ref_mesh_fe, 
                                    pde=pde, 
                                    material=material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                    enable_logging=False
                                )
        
        uh_ref = lfa_ref.solve(density_distribution=rho_ref)

        errorType = ['$|| \\boldsymbol{u}_h - \\boldsymbol{u}_{ref} ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"Solving on mesh level {i+1}/{maxit}...")

            mesh_i_fe = pde.init_mesh(nx=nx, ny=ny)            
            for _ in range(i):
                mesh_i_fe.bisect(isMarkedCell=None, options={'disp': False})

            mesh_i_fe.to_vtk(f'mesh_level_{i+1}.vtu')

            rho_i = interpolation_scheme.setup_density_distribution(
                                mesh=mesh_i_fe,
                                relative_density=0.5,
                                quadrature_order=4
                            )

            lfa_i = LagrangeFEMAnalyzer(
                                    mesh=mesh_i_fe, 
                                    pde=pde, 
                                    material=material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                    enable_logging=False
                                )

            uh_i = lfa_i.solve(density_distribution=rho_i)

            uh_i_projected = project_solution_to_finer_mesh(
                                                pde=pde,
                                                nx=nx, ny=ny,
                                                uh=uh_i, 
                                                lfa=lfa_i, 
                                                source_refinement_level=i, 
                                                target_mesh=ref_mesh_fe
                                            )

            e0 = ref_mesh_fe.error(uh_i_projected, uh_ref)
            errorMatrix[0, i] = e0
             
            NDof[i] = lfa_i.tensor_space.number_of_global_dofs()
            # h[i] = mesh_i.meshdata.get('hx')
            
        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        order_l2 = bm.log2(errorMatrix[0, :-2] / errorMatrix[0, 2:])
        self._log_info(f"order_l2: {order_l2}")
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return uh_ref


    @run.register('test_topopt_analysis_density_location')
    def run(self, 
            density_location: str = 'element'
        ) -> Function:
        """测试拓扑优化分析阶段 (不同密度分布下), 位移求解的正确性"""
        # 参数设置
        nx, ny = 120, 60
        init_relative_density = 0.5
        interpolation_order = 1
        
        # 设置 pde
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        domain_length = pde.domain[1] - pde.domain[0]
        domain_height = pde.domain[3] - pde.domain[2]
        pde.init_mesh.set('uniform_quad')

        # 设置过滤类型和半径
        filter_type = 'none'
        rmin = (0.04 * nx) / (domain_length / nx)

        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        # 设置插值方案
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-12,
                                        'target_variables': ['E']
                                    },
                                )
        
        # 设置密度
        opt_mesh = pde.init_mesh(nx=nx, ny=ny)
        rho = interpolation_scheme.setup_density_distribution(
                                        mesh=opt_mesh,
                                        relative_density=init_relative_density,
                                        integrator_order=self.integrator_order,
                                        interpolation_order=interpolation_order
                                    )
        
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=fe_mesh,
                                    pde=pde,
                                    material=material,
                                    interpolation_scheme=interpolation_scheme,
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                )
        uh = lagrange_fem_analyzer.solve_displacement(density_distribution=rho)
        
        print('-----------------------')


    @run.register('test_topopt_analysis_assembly_method')
    def run(self):
        domain = [0, 4, 0, 2]

        T = -1.0
        E, nu = 1000.0, 0.3

        nx, ny = 120, 60

        space_degree = 1
        integration_order = space_degree + 1
        
        density_location = 'lagrange_interpolation_point'  # 'lagrange_interpolation_point', 'element'
        interpolation_order = 1 
        relative_density = 0.5
        penalty_factor = 3.0

        assembly_method = 'standard'  # 'standard', 'voigt', 'sparse_optimized'

        from soptx.model.cantilever_2d import CantileverBeamMiddle2dData
        pde = CantileverBeamMiddle2dData(
                            domain=domain,
                            T=T, E=E, nu=nu,
                            enable_logging=False
                        )

        pde.init_mesh.set('uniform_quad')

        fe_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        opt_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=relative_density,
                                                interpolation_order=interpolation_order
                                            )

        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer_standrad = LagrangeFEMAnalyzer(
                                                mesh=fe_mesh,
                                                pde=pde,
                                                material=material,
                                                interpolation_scheme=interpolation_scheme,
                                                space_degree=space_degree,
                                                integration_order=integration_order,
                                                assembly_method='standard',
                                                solve_method='mumps',
                                                topopt_algorithm='density_based',
                                            )
        
        lagrange_fem_analyzer_standrad = LagrangeFEMAnalyzer(
                                        mesh=fe_mesh,
                                        pde=pde,
                                        material=material,
                                        interpolation_scheme=interpolation_scheme,
                                        space_degree=space_degree,
                                        integration_order=integration_order,
                                        assembly_method='voigt',
                                        solve_method='mumps',
                                        topopt_algorithm='density_based',
                                    )
        
        uh_standard = lagrange_fem_analyzer_standrad.solve_displacement(density_distribution=rho)
        uh_voigt = lagrange_fem_analyzer_standrad.solve_displacement(density_distribution=rho)
        print('-----------------------')

if __name__ == "__main__":
    test = LagrangeFEMAnalyzerTest(enable_logging=True)
    
    # p = 2
    # q = p+3
    # test.set_space_degree(p)
    # test.set_integrator_order(q)
    # test.set_assembly_method('standard')
    # test.set_solve_method('mumps')

    # test.run.set('topopt_analysis_reference_solution')
    # test.run.set('topopt_analysis_exact_solution')
    # test.run.set('lfa_analysis_reference_solution')
    
    # test.run.set('topopt_analysis')
    # test.run.set('test_bisect')

    # test.run.set('lfa_analysis_exact_solution')
    # uh = test.run()

    # test.run.set('test_topopt_analysis_density_location')
    # uh1 = test.run(density_location='element')

    test.run.set('lfa_exact_solution')
    test.run()
