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
    
    @run.register('LFA_top_material')
    def run(self):
        if self.pde is None or self.material is None:
            raise ValueError("请先设置 PDE 和拓扑材料参数")
        
        self._log_info(f"LagrangeFEMAnalyzerTest configuration:\n"
                f"pde = {self.pde}, \n"
                f"mesh = {self.mesh}, \n"
                f"material = {self.material}, \n"
                f"relative_density = {self.material.relative_density}, "
                f"interpolation_method = '{self.material.interpolation_method}', "
                f"density_location = '{self.material.density_location}', \n"
                f"p = {self.p}, assembly_method = '{self.assembly_method}', "
                f"solve_method = '{self.solve_method}'")
        
        lfa = LagrangeFEMAnalyzer(pde=self.pde, material=self.material, space_degree=self.p,
                    assembly_method=self.assembly_method, 
                    solve_method=self.solve_method)
                    
        self.uh = lfa.solve()

        return self.uh
    
    @run.register('LFA_top_material_analysis_true_solution')
    def run(self, maxit: int) -> TensorLike:
        if self.pde is None or self.material is None:
            raise ValueError("请先设置 PDE 和拓扑材料参数")
        
        if not isinstance(self.material, TopologyOptimizationMaterial):
            error_msg = (f"Expected TopologyOptimizationMaterial, got {type(self.material).__name__}. "
                        "This analysis method is specifically for topology optimization materials.")
            self._log_error(error_msg)
            raise TypeError(error_msg)
        
        top_material = self.material
        mesh = self.mesh

        top_material.setup_density_distribution(mesh=mesh)
        interpolation_scheme = top_material.interpolation_scheme
        
        self._log_info(f"LagrangeFEMAnalyzerTest configuration:\n"
                f"pde = {self.pde}, \n"
                f"mesh = {self.mesh}, \n"
                f"material = {top_material}, \n"
                f"relative_density = {top_material.relative_density}, "
                f"interpolation_method = '{interpolation_scheme.interpolation_method}', "
                f"density_location = '{top_material.density_location}', "
                f"p = {self.p}, assembly_method = '{self.assembly_method}', "
                f"solve_method = '{self.solve_method}'")
        
        errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"第 {i+1}/{maxit} 次迭代...")

            lfa = LagrangeFEMAnalyzer(mesh=mesh, pde=self.pde, 
                        material=top_material, space_degree=self.p,
                        assembly_method=self.assembly_method, 
                        solve_method=self.solve_method)
            
            self.uh = lfa.solve()

            e0 = mesh.error(self.uh, self.pde.disp_solution)
            errorMatrix[0, i] = e0

            NDof[i] = lfa.tensor_space.number_of_global_dofs()

            initial_hx = mesh.meshdata.get('hx') if i == 0 else h[0]
            h[i] = initial_hx / (2 ** i)

            if i < maxit - 1:
                mesh.uniform_refine()
                top_material.setup_density_distribution(mesh=mesh)

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return self.uh
    

    @run.register('topopt_analysis_exact_solution')
    def run(self, maxit: int = 5) -> TensorLike:
        
        # 设置 pde
        from soptx.model.linear_elasticity_2d import BoxTriLagrangeData2d
        pde = BoxTriLagrangeData2d(
                            domain=[0, 1, 0, 1], 
                            E=1.0, nu=0.3,
                            enable_logging=False
                        )
        pde.init_mesh.set('uniform_tri')
        mesh = pde.init_mesh(nx=10, ny=10)

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
        
        mesh = self.mesh

        errorType = ['$|| P_h(\\boldsymbol{u}_h) - \\boldsymbol{u}_{ref} ||_{L^2}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"Solving on mesh level {i+1}/{maxit}...")

            lfa = LagrangeFEMAnalyzer(
                                    mesh=mesh, 
                                    pde=pde, 
                                    material=base_material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=self.topopt_algorithm,
                                    topopt_config=self.topopt_config
                                )

            uh = lfa.solve()

            e0 = mesh.error(uh, pde.disp_solution)
            errorMatrix[0, i] = e0

            NDof[i] = lfa.tensor_space.number_of_global_dofs()
            h[i] = mesh.meshdata.get('hx')

            
            if i < maxit - 1:
                mesh.uniform_refine()
                # mesh.bisect(isMarkedCell=None, options={'disp': False})

                mesh.to_vtk(f'mesh_level_{i+1}.vtu')

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return uh

    @run.register('topopt_analysis_reference_solution')
    def run(self, 
            nx: int = 10, ny: int = 10, 
            maxit: int = 6, ref_level: int = 7
        ) -> TensorLike:
        
        # 设置拓扑优化配置
        interpolation_config = InterpolationConfig(
                                    method='msimp',
                                    penalty_factor=3,
                                    target_variables=['E'],
                                    void_youngs_modulus=1e-12
                                )
        density_based_config = DensityBasedConfig(
                                    density_location='gauss_integration_point',
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

        ref_mesh.to_vtk(f'initial_mesh.vtu')

        total_refinement = maxit - 1 + ref_level
        for _ in range(total_refinement):
            ref_mesh.bisect(isMarkedCell=None, options={'disp': False})
        self._log_info(f"Reference mesh with {ref_mesh.number_of_cells()} cells.")

        ref_mesh.to_vtk(f'ref_mesh.vtu')

        lfa_ref = LagrangeFEMAnalyzer(mesh=ref_mesh, 
                                    pde=pde, 
                                    material=base_material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=self.topopt_algorithm,
                                    topopt_config=self.topopt_config)
        uh_ref = lfa_ref.solve()

        errorType = ['$|| P_h(\\boldsymbol{u}_h) - \\boldsymbol{u}_{ref} ||_{L^2}$']
        errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            print(f"Solving on mesh level {i+1}/{maxit}...")

            mesh_i = pde.init_mesh(nx=nx, ny=ny)
            for _ in range(i):
                mesh_i.bisect(isMarkedCell=None, options={'disp': False})

            lfa_i = LagrangeFEMAnalyzer(mesh=mesh_i, 
                                    pde=pde, 
                                    material=base_material, 
                                    space_degree=self.space_degree,
                                    integrator_order=self.integrator_order,
                                    assembly_method=self.assembly_method, 
                                    solve_method=self.solve_method,
                                    topopt_algorithm=self.topopt_algorithm,
                                    topopt_config=self.topopt_config)
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
            e1 = ref_mesh.error(uh_ref, pde.disp_solution)
            errorMatrix[0, i] = e0
            errorMatrix[1, i] = e1
             
            NDof[i] = lfa_i.tensor_space.number_of_global_dofs()
            h[i] = mesh_i.meshdata.get('hx')
            
        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return uh_ref


if __name__ == "__main__":
    test1 = LagrangeFEMAnalyzerTest(enable_logging=True)

    # 一、基础线弹性材料求解
    # ## 1.1 创建 pde
    # from soptx.model.linear_elasticity_2d import BoxTriLagrangeData2d
    # pde = BoxTriLagrangeData2d(
    #                     domain=[0, 1, 0, 1], 
    #                     E=1.0, nu=0.3,
    #                     enable_logging=False
    #                 )
    # ## 1.2 创建基础材料
    # from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
    # base_material = IsotropicLinearElasticMaterial(
    #                                     youngs_modulus=pde.E, 
    #                                     poisson_ratio=pde.nu, 
    #                                     plane_type=pde.plane_type,
    #                                     enable_logging=False
    #                                 )

    # test1.set_pde(pde)
    # test1.set_init_mesh('uniform_quad', nx=5, ny=5)
    # test1.set_material(base_material)
    # test1.set_space_degree(3)
    # test1.set_assembly_method('fast')
    # test1.set_solve_method('mumps')

    # uh1 = test1.run(maxit=4)

    # 二、拓扑优化材料求解
    ## 2.1 创建 pde 并设置网格
    # from soptx.model.mbb_beam_2d import HalfMBBBeam2dData1
    # pde = HalfMBBBeam2dData1(
    #                     domain=[0, 60, 0, 20],
    #                     T=-1.0, E=1.0, nu=0.3,
    #                     enable_logging=False
    #                 )
    # pde.init_mesh.set('uniform_quad')
    # mesh = pde.init_mesh(nx=60, ny=20)

    # ## 2.2 创建基础材料
    # from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
    # base_material = IsotropicLinearElasticMaterial(
    #                                     youngs_modulus=pde.E, 
    #                                     poisson_ratio=pde.nu, 
    #                                     plane_type=pde.plane_type,
    #                                     enable_logging=False
    #                                 )

    # # 2.3 设置材料插值方案
    # from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
    # interpolation_scheme = MaterialInterpolationScheme(
    #                             penalty_factor=3.0, 
    #                             void_youngs_modulus=1e-12, 
    #                             interpolation_method='simp', 
    #                             enable_logging=False
    #                         )
    
    # # 2.4. 创建拓扑优化材料
    # from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial
    # top_material = TopologyOptimizationMaterial(
    #                         mesh=mesh,
    #                         base_material=base_material,
    #                         interpolation_scheme=interpolation_scheme,
    #                         relative_density=1.0,
    #                         density_location='element',
    #                         quadrature_order=3,
    #                         enable_logging=True
    #                     )
    
    # test1.set_pde(pde)
    # test1.set_material(top_material)
    # test1.set_space_degree(1)
    # test1.set_assembly_method('standard')
    # test1.set_solve_method('mumps')
    # test1.run.set('LFA_top_material_analysis')

    # uh1 = test1.run(maxit=4)

    # 三、验证
    ## 3.1 创建 pde
    from soptx.model.linear_elasticity_2d import BoxTriLagrangeData2d
    pde = BoxTriLagrangeData2d(
                        domain=[0, 1, 0, 1], 
                        E=1.0, nu=0.3,
                        enable_logging=False
                    )

    ## 3.2 创建基础材料
    from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
    base_material = IsotropicLinearElasticMaterial(
                                        youngs_modulus=pde.E, 
                                        poisson_ratio=pde.nu, 
                                        plane_type=pde.plane_type,
                                        enable_logging=False
                                    )

    # # 3.3 设置材料插值方案
    # from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
    # interpolation_scheme = MaterialInterpolationScheme(
    #                             penalty_factor=3.0, 
    #                             void_youngs_modulus=1e-12, 
    #                             interpolation_method='simp', 
    #                             enable_logging=False
    #                         )
    
    # # 2.4. 创建拓扑优化材料
    # from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial
    # top_material = TopologyOptimizationMaterial(
    #                         base_material=base_material,
    #                         interpolation_scheme=interpolation_scheme,
    #                         relative_density=1.0,
    #                         density_location='element_gauss_integrate_point',
    #                         quadrature_order=3,
    #                         enable_logging=False
    #                     )
    
    # test1.set_pde(pde)
    # test1.set_init_mesh('uniform_tri', nx=20, ny=20)
    # test1.set_init_mesh('tri')
    # test1.set_material(base_material)
    test1.set_space_degree(1)
    test1.set_integrator_order(4)
    test1.set_assembly_method('standard')
    test1.set_solve_method('mumps')

    # test1.run.set('topopt_analysis_reference_solution')
    test1.run.set('topopt_analysis_exact_solution')

    uh1 = test1.run()
