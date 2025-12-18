from typing import Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod

from soptx.utils.base_logged import BaseLogged

class HuZhangMFEMAnalyzerTest(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

    @variantmethod('test_exact_solution_hzmfem')
    def run(self, model) -> None:
        """基于有真解的算例验证胡张混合有限元的正确性"""
        if model == 'tri_sol_dir_huzhang':
            from soptx.model.linear_elasticity_2d import TriSolHomoDirHuZhang2d
            lam, mu = 1.0, 0.5
            pde = TriSolHomoDirHuZhang2d(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_crisscross_tri')
            nx, ny = 2, 2
            analysis_mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam, 
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )


        elif model == 'poly_sol_pure_homo_dir_huzhang_3d':
            # 三维纯齐次 Dirichlet
            from soptx.model.linear_elasticity_3d import PolySolPureHomoDirHuZhang3d
            lam, mu = 1.0, 0.5
            pde = PolySolPureHomoDirHuZhang3d(domain=[0, 1, 0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_tet')
            nx, ny, nz = 2, 2, 2
            analysis_mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam, 
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )


        space_degree = 2
        integration_order = space_degree*2 + 2
        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                          f"空间次数={space_degree}, 积分阶数={integration_order}")

        maxit = 5
        errorType = [
                    '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega, 0}$',
                    '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega, 0}$',
                    '$|| \\boldsymbol{\\div\\sigma} - \\boldsymbol{\\div\\sigma}_h||_{\\Omega, 0}$',
                    '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega, H(div)}$'
                    ]
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            N = 2**(i+1) 
            from soptx.analysis.huzhang_mfem_analyzer_old import HuZhangMFEMAnalyzer
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

            sigmah, uh = huzhang_mfem_analyzer.solve_displacement(rho_val=None)
            sigma = pde.stress_solution(analysis_mesh.node)

            e_uh_l2 = analysis_mesh.error(u=uh, 
                                    v=pde.disp_solution,
                                    q=integration_order) # 位移 L2 范数误差
            e_sigmah_l2 = analysis_mesh.error(u=sigmah, 
                                            v=pde.stress_solution, 
                                            q=integration_order) # 应力 L2 范数误差
            e_div_sigmah_l2 = analysis_mesh.error(u=sigmah.div_value, 
                                                v=pde.div_stress_solution, 
                                                q=integration_order) # 应力散度 L2 范数误差
            e_sigmah_hdiv = bm.sqrt(e_sigmah_l2**2 + e_div_sigmah_l2**2) # 应力 H(div) 范数误差

            h[i] = 1/N
            errorMatrix[0, i] = e_uh_l2
            errorMatrix[1, i] = e_sigmah_l2
            errorMatrix[2, i] = e_div_sigmah_l2
            errorMatrix[3, i] = e_sigmah_hdiv

            if i < maxit - 1:
                analysis_mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)   
        print("NDof:", NDof)
        print("order_uh_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        print("order_sigmah_l2:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
        print("order_div_sigmah_l2:\n", bm.log2(errorMatrix[2, :-1] / errorMatrix[2, 1:]))
        print("order_sigmah_hdiv:\n", bm.log2(errorMatrix[3, :-1] / errorMatrix[3, 1:]))


        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table

        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()
        print('------------------')



if __name__ == "__main__":

    huzhang_analyzer = HuZhangMFEMAnalyzerTest(enable_logging=True)

    huzhang_analyzer.run.set('test_exact_solution_hzmfem')
    huzhang_analyzer.run(model='tri_sol_dir_huzhang')