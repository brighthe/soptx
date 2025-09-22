from typing import Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from soptx.utils.show import showmultirate, show_error_table
from soptx.analysis.utils import project_solution_to_finer_mesh


class HuZhangMFEMAnalyzerTest(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test')
    def run(self, test_demo: str = '2d_simplex') -> None:

        if test_demo == '2d_simplex':
            from soptx.model.linear_elasticity_2d import BoxTriHuZhangData2d
            pde = BoxTriHuZhangData2d(lam=1, mu=0.5)
            # TODO 支持四边形网格
            pde.init_mesh.set('uniform_tri')
            nx, ny = 4, 4
            analysis_mesh = pde.init_mesh(nx=nx, ny=ny)
            # TODO 支持 3 次以下
            space_degree = 2

            integration_order = space_degree**2 + 2

            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam, 
                                                shear_modulus=pde.mu, 
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )
            maxit = 5
            errorType = [
                        '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                        # '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\Omega,1}$',
                        '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$'
                        ]
            errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
            NDof = bm.zeros(maxit, dtype=bm.int32)
            h = bm.zeros(maxit, dtype=bm.float64)

            for i in range(maxit):
                N = 2**(i+1) 

                huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                                        mesh=analysis_mesh,
                                                        pde=pde,
                                                        material=material,
                                                        space_degree=space_degree,
                                                        integration_order=integration_order,
                                                        solve_method='mumps',
                                                        topopt_algorithm=None,
                                                        interpolation_scheme=None,
                                                    )
                
                uh_dof = huzhang_mfem_analyzer._tensor_space.number_of_global_dofs()
                sigma_dof = huzhang_mfem_analyzer._huzhang_space.number_of_global_dofs()
                NDof[i] = uh_dof + sigma_dof

                sigmah, uh = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)
                
                e0 = analysis_mesh.error(uh, 
                                        pde.disp_solution,
                                        q=integration_order) # 位移 L2 范数误差
                # e1 = analysis_mesh.error(uh.grad_value, 
                #                         pde.grad_disp_solution, 
                #                         q=integration_order) # 位移 H1 半范数误差
                e1 = analysis_mesh.error(sigmah, 
                                        pde.stress_solution, 
                                        q=integration_order) # 应力 L2 范数误差

                h[i] = 1/N
                errorMatrix[0, i] = e0
                errorMatrix[1, i] = e1 
                # errorMatrix[2, i] = e2

                if i < maxit - 1:
                    analysis_mesh.uniform_refine()

            print("errorMatrix:\n", errorType, "\n", errorMatrix)   
            print("NDof:", NDof)
            print("order_uh_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
            # print("order_uh_h1:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
            print("order_sigma_l2:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))


            import matplotlib.pyplot as plt
            from soptx.utils.show import showmultirate, show_error_table

            show_error_table(h, errorType, errorMatrix)
            showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
            plt.show()
            print('------------------')

    @run.register('test_jump_penalty_integrator')
    def run(self):
        from soptx.model.linear_elasticity_2d import BoxTriHuZhangData2d
        pde = BoxTriHuZhangData2d(lam=1, mu=0.5)
        # TODO 支持四边形网格
        pde.init_mesh.set('uniform_tri')
        nx, ny = 2, 2
        mesh = pde.init_mesh(nx=nx, ny=ny)
        GD = mesh.geo_dimension()
        # TODO 支持 3 次以下
        p = 2
        q = p**2 + 2

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

        scalar_space = LagrangeFESpace(mesh, p=p-1, ctype='D')
        tensor_space = TensorFunctionSpace(scalar_space, shape=(GD, -1))
        gdof = tensor_space.number_of_global_dofs()
        from soptx.analysis.integrators.jump_penalty_integrator import JumpPenaltyIntegrator
        # jpi = JumpPenaltyIntegrator(q=q)
        # test = jpi.to_global_dof(tensor_space)
        # KE_jump = jpi.assembly(tensor_space)
        from fealpy.fem import BilinearForm
        bform = BilinearForm(tensor_space)
        bform.add_integrator(JumpPenaltyIntegrator(q=q))
        K = bform.assembly()
        print("--------------")


if __name__ == "__main__":
    huzhang_analyzer = HuZhangMFEMAnalyzerTest(enable_logging=True)

    huzhang_analyzer.run.set('test')
    # huzhang_analyzer.run.set('test_jump_penalty')
    huzhang_analyzer.run()
