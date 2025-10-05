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


class LagrangeFEMAnalyzerTest(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)


    @variantmethod('lfa_exact_solution')
    def run(self, model_type: str = 'BoxTriMixed2d') -> TensorLike:

        if model_type == 'BoxTrDirichleti2d':
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
        
        elif model_type == 'BoxTriMixed2d':
            from soptx.model.linear_elasticity_2d import BoxTriMixedLagrange2dData
            domain = [0, 1, 0, 1]
            E, nu = 1.0, 0.3
            pde = BoxTriMixedLagrange2dData(domain=domain, E=E, nu=nu)
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

        space_degree = 2
        integration_order = space_degree + 3
        # 'standard', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard'

        maxit = 4
        errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{\\Omega, 0}$', 
                     '$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{\\Omega, 1}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        self._log_info(f"模型: {type(pde).__name__}, 网格: {type(mesh).__name__}, ")

        for i in range(maxit):
            N = 2**(i+1)

            lfa = LagrangeFEMAnalyzer(
                                    mesh=mesh,
                                    pde=pde, 
                                    material=material, 
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method='mumps',
                                    topopt_algorithm=None,
                                    interpolation_scheme=None
                                )
                    
            uh = lfa.solve_displacement()
            NDof[i] = lfa.tensor_space.number_of_global_dofs()

            e_l2 = mesh.error(uh, pde.disp_solution)
            e_h1 = mesh.error(uh.grad_value, pde.disp_solution_gradient)

            h[i] = 1/N
            errorMatrix[0, i] = e_l2
            errorMatrix[1, i] = e_h1

            if i < maxit - 1:
                mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        print("order_h1:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))

        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return uh

if __name__ == "__main__":
    test = LagrangeFEMAnalyzerTest(enable_logging=True)
    
    test.run.set('lfa_exact_solution')
    test.run()
