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
from soptx.interpolation.config import DensityBasedConfig, LevelSetConfig, InterpolationConfig
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

    def set_topopt_config(self, algorithm: str, config: Union[DensityBasedConfig, None]) -> None:
        """设置拓扑优化配置"""
        self.topopt_algorithm = algorithm
        self.topopt_config = config
        self._log_info(f"Topology optimization set: {algorithm}, config: {type(config).__name__ if config else None}")


    @variantmethod('LFA_base_material')
    def run(self, maxit: int):
        if self.pde is None or self.material is None:
            raise ValueError("请先设置 PDE 和基本材料参数")
        
        self._log_info(f"LagrangeFEMAnalyzerTest configuration:\n"
                f"pde = {self.pde}, \n"
                f"mesh = {self.mesh}, \n"
                f"material = {self.material}, \n"
                f"p = {self.p}, assembly_method = '{self.assembly_method}', "
                f"solve_method = '{self.solve_method}'")

        errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"第 {i+1}/{maxit} 次迭代...")

            lfa = LagrangeFEMAnalyzer(pde=self.pde, material=self.material, space_degree=self.p,
                                assembly_method=self.assembly_method, 
                                solve_method=self.solve_method)
                    
            self.uh = lfa.solve()

            mesh = lfa.mesh
            e0 = self.mesh.error(self.uh, self.pde.disp_solution)
            errorMatrix[0, i] = e0

            NDof[i] = lfa.tensor_space.number_of_global_dofs()
            
            initial_hx = mesh.meshdata.get('hx') if i == 0 else h[0]
            h[i] = initial_hx / (2 ** i)

            if i < maxit - 1:
                mesh.uniform_refine()
                # NOTE 内部操作接受现有网格并设置到 PDE 中和拓扑优化材料中
                self.pde.mesh = mesh
                self.material.mesh = mesh

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return self.uh
    

    @run.register('lfa_analysis_exact_solution')
    def run(self, maxit: int = 5) -> TensorLike:
        
        # 设置 pde
        from soptx.model.linear_elasticity_2d import BoxTriLagrange2dData
        pde = BoxTriLagrange2dData(
                            domain=[0, 1, 0, 1], 
                            E=1.0, nu=0.3,
                            enable_logging=False
                        )
        nx, ny = 10, 10
        pde.init_mesh.set('uniform_tri')
        mesh = pde.init_mesh(nx=nx, ny=ny)

        mesh.to_vtk(f'initial_mesh.vtu')

        # 设置基础材料
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

        errorType = ['$|| \\boldsymbol{u}_h - \\boldsymbol{u} ||_{L^2}$', '$ || \\boldsymbol{u}_h - \\boldsymbol{u} ||_{H^1}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"Solving on mesh level {i+1}/{maxit}...")

            lfa = LagrangeFEMAnalyzer(
                                    mesh=mesh, 
                                    pde=pde, 
                                    material=material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=None,
                                    interpolation_scheme=None
                                )

            uh = lfa.solve(density_distribution=None)

            e0 = mesh.error(uh, pde.disp_solution)
            e1 = mesh.error(uh.grad_value, pde.disp_solution_gradient)
            errorMatrix[0, i] = e0
            errorMatrix[1, i] = e1

            NDof[i] = lfa.tensor_space.number_of_global_dofs()
            # h[i] = mesh.meshdata.get('hx')

            mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        order_l2 = bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:])
        self._log_info(f"order_l2: {order_l2}")
        order_h1 = bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:])
        self._log_info(f"order_h1: {order_h1}")
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

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
        
        uh_ref = lfa_ref.solve(density_distribution=None)

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

            uh_i = lfa_i.solve(density_distribution=None)

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


if __name__ == "__main__":
    test = LagrangeFEMAnalyzerTest(enable_logging=True)
    
    p = 3
    q = p+3
    test.set_space_degree(p)
    test.set_integrator_order(q)
    test.set_assembly_method('standard')
    test.set_solve_method('mumps')

    # test.run.set('topopt_analysis_reference_solution')
    # test.run.set('topopt_analysis_exact_solution')
    test.run.set('lfa_analysis_reference_solution')
    # test.run.set('topopt_analysis')
    # test.run.set('test_bisect')

    uh1 = test.run()
