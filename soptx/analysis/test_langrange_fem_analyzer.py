import matplotlib.pyplot as plt
from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.utils.show import showmultirate, show_error_table
from soptx.utils.base_logged import BaseLogged

class LagrangeFEMAnalyzerTest(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.pde = None
        self.mesh = None
        self.material = None
        self.p = 1
        self.assembly_method = 'standard'
        self.solve_method = 'mumps'

        self._log_info(f"LagrangeFEMAnalyzerTest initialized: "
                f"pde = {self.pde}, "
                f"mesh = {self.mesh}, "
                f"material = {self.material}, "
                f"p = {self.p}, assembly_method = '{self.assembly_method}', "
                f"solve_method = '{self.solve_method}'")

    def set_pde(self, pde):
        self.pde = pde

    def set_init_mesh(self, meshtype: str, **kwargs):
        self.pde.init_mesh.set(meshtype)
        self.mesh = self.pde.init_mesh(**kwargs)

    def set_material(self, material):
        self.material = material

    def set_space_degree(self, p):
        self.p = p

    def set_assembly_method(self, method: str):
        self.assembly_method = method

    def set_solve_method(self, method: str):
        self.solve_method = method

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
                # NOTE 内部操作接受现有网格并设置到 PDE 中
                self.pde._mesh = mesh

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


if __name__ == "__main__":
    test1 = LagrangeFEMAnalyzerTest(enable_logging=True)

    # 一、基础线弹性材料求解
    ## 1.1 创建 pde
    from soptx.model.linear_elasticity_2d import BoxTriLagrangeData2d
    pde = BoxTriLagrangeData2d(
                        domain=[0, 1, 0, 1], 
                        E=1.0, nu=0.3,
                        enable_logging=False
                    )
    ## 1.2 创建基础材料
    from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
    base_material = IsotropicLinearElasticMaterial(
                                        youngs_modulus=pde.E, 
                                        poisson_ratio=pde.nu, 
                                        plane_type=pde.plane_type,
                                        enable_logging=False
                                    )

    test1.set_pde(pde)
    test1.set_init_mesh('uniform_quad', nx=5, ny=5)
    test1.set_material(base_material)
    test1.set_space_degree(3)
    test1.set_assembly_method('fast')
    test1.set_solve_method('mumps')

    uh1 = test1.run(maxit=4)

    # 二、拓扑优化材料求解
    # ## 2.1 创建 pde 并设置网格
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
    #                         density_location='element_gauss_integrate_point',
    #                         quadrature_order=3,
    #                         enable_logging=False
    #                     )
    
    # test1.set_pde(pde)
    # test1.set_material(top_material)
    # test1.set_space_degree(1)
    # test1.set_assembly_method('standard')
    # test1.set_solve_method('mumps')