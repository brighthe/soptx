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

    @variantmethod('test_exact_solution')
    def run(self, test_demo: str = 'box_tri_huzhang') -> None:
        """基于有真解的算例验证胡张混合有限元的正确性"""
        if test_demo == 'box_tri_huzhang':
            from soptx.model.linear_elasticity_2d import BoxTriHuZhangData2d
            pde = BoxTriHuZhangData2d(domain=[0, 1, 0, 1], lam=1, mu=0.5)
            # TODO 支持四边形网格
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
            analysis_mesh = pde.init_mesh(nx=nx, ny=ny)
            # TODO 支持 3 次以下
            space_degree = 2

            # 单纯形网格
            integration_order = space_degree + 4

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
                        '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                        '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,H(div)}$'
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
                errorMatrix[2, i] = e_sigmah_hdiv

                if i < maxit - 1:
                    analysis_mesh.uniform_refine()

            print("errorMatrix:\n", errorType, "\n", errorMatrix)   
            print("NDof:", NDof)
            print("order_uh_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
            print("order_sigmah_l2:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
            print("order_sigmah_hdiv:\n", bm.log2(errorMatrix[2, :-1] / errorMatrix[2, 1:]))

            import matplotlib.pyplot as plt
            from soptx.utils.show import showmultirate, show_error_table

            show_error_table(h, errorType, errorMatrix)
            showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
            plt.show()
            print('------------------')

    @run.register('test_huzhang')
    def run(self, test_demo: str = 'bridge_2d_double_load') -> None:
        """对于无真解的算例, 基于位移有限元的结果验证胡张混合有限元的结果的正确性"""
        if test_demo == 'bridge_2d_double_load':

            E = 1.0
            nu = 0.35
            from soptx.model.bridge_2d import Bridge2dDoubleLoadData
            pde = Bridge2dDoubleLoadData(domain=[0, 1, 0, 1], 
                                        T1 = -2.0, T2=-2.0,
                                        E=E, nu=nu)
            # TODO 支持四边形网格
            pde.init_mesh.set('uniform_tri')
            nx, ny = 2, 2
            displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                youngs_modulus=pde.E,
                                                poisson_ratio=pde.nu,
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )
            
            ## 位移 Lagrange 有限元
            space_degree = 3
            # 单纯形网格
            integration_order = space_degree + 4
            from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
            lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=space_degree,
                                        integration_order=integration_order,
                                        assembly_method='standard',
                                        solve_method='mumps',
                                        topopt_algorithm=None,
                                        interpolation_scheme=None,
                                    )
            uh = lagrange_fem_analyzer.solve_displacement(density_distribution=None)

            ## 位移应力混合 HuZhang 有限元
            huzhang_space_degree = 3
            integration_order = huzhang_space_degree + 4
            from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
            huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                        mesh=displacement_mesh,
                                        pde=pde,
                                        material=material,
                                        space_degree=huzhang_space_degree,
                                        integration_order=integration_order,
                                        solve_method='mumps',
                                        topopt_algorithm=None,
                                        interpolation_scheme=None,
                                    )
            sigmah, uh = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)
            
            print('------------------')

        else:
            raise NotImplementedError(f"The test_demo '{test_demo}' has not been implemented yet.")


    @run.register('test_none_exact_solution')
    def run(self, model_type) -> TensorLike:
        if model_type == 'bearing_device_2d':
            E = 100.0
            nu = 0.4   # 可压缩
            plane_type = 'plane_strain'  # 'plane_stress' or 'plane_strain'
            
            from soptx.model.bearing_device_2d import HalfBearingDevice2D
            pde = HalfBearingDevice2D(
                                domain=[0, 0.6, 0, 0.4],
                                t=-1.8,
                                E=E, nu=nu,
                                plane_type=plane_type,
                            )
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 60, 40

        elif model_type == 'clamped_beam_2d':
            E = 30.0
            nu = 0.4  # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'

            from soptx.model.clamped_beam_2d import HalfClampedBeam2D
            pde = HalfClampedBeam2D(
                    domain=[0, 80, 0, 20],
                    p=-1.5,
                    E=E, nu=nu,
                    plane_type=plane_type,
                )
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 80, 20

        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)
        NN = displacement_mesh.number_of_nodes()
        NE = displacement_mesh.number_of_edges()
        NC = displacement_mesh.number_of_cells()

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E,
                                            poisson_ratio=pde.nu,
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        ## 位移应力混合 HuZhang 有限元
        huzhang_space_degree = 3
        integration_order = huzhang_space_degree + 4
        from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
        huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=huzhang_space_degree,
                                    integration_order=integration_order,
                                    solve_method='mumps',
                                    topopt_algorithm=None,
                                    interpolation_scheme=None,
                                )
        
        space_sigmah = huzhang_mfem_analyzer.huzhang_space
        space_uh = huzhang_mfem_analyzer.tensor_space

        SGDOF_uh = space_uh.scalar_space.number_of_global_dofs()
        TGDOF_uh = space_uh.number_of_global_dofs()
        TGDOF_sigmah = space_sigmah.number_of_global_dofs()
        TLDOF_sigmah_n = space_sigmah.dof.number_of_internal_local_dofs('node')
        TLDOF_sigmah_e = space_sigmah.dof.number_of_internal_local_dofs('edge')
        TLDOF_sigmah_c = space_sigmah.dof.number_of_internal_local_dofs('cell')
        TGDOF_sigmah_n = TLDOF_sigmah_n * NN
        TGDOF_sigmah_e = TLDOF_sigmah_e * NE
        TGDOF_sigmah_c = TLDOF_sigmah_c * NC

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                       f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 总自由度={TGDOF_uh}, "
                       f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, "
                        f"总自由度={TGDOF_sigmah}, 节点自由度={TGDOF_sigmah_n}, 边自由度={TGDOF_sigmah_e}, 单元自由度={TGDOF_sigmah_c}")

        sigmah, uh = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)


        from soptx.analysis.utils import _get_val_tensor_to_component, _get_val_component_to_tensor
        uh_component = _get_val_tensor_to_component(val=uh, space=huzhang_mfem_analyzer.tensor_space) # (NN, GD)
        # sigmah_component = _get_val_tensor_to_component(val=sigmah, space=huzhang_mfem_analyzer.huzhang_space) # (NN, 3)
        displacement_mesh.nodedata['uh'] = uh_component
        # displacement_mesh.nodedata['stress'] = sigmah_component

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        displacement_mesh.to_vtk(f"{base_dir}/uh_hzmfem.vtu")

        return uh

    @run.register('test_jump_penalty_integrator')
    def run(self):
        """测试稳定化项积分子 JumpPenaltyIntegrator 的正确性"""
        from soptx.model.linear_elasticity_2d import BoxTriHuZhangData2d
        pde = BoxTriHuZhangData2d(lam=1, mu=0.5)
        pde.init_mesh.set('uniform_aligned_tri')
        nx, ny = 2, 2
        mesh = pde.init_mesh(nx=nx, ny=ny)
        GD = mesh.geo_dimension()

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # mesh.add_plot(axes)
        # mesh.find_node(axes, showindex=True, color='g', markersize=12, fontsize=16, fontcolor='g')
        # mesh.find_face(axes, showindex=True, color='r', markersize=14, fontsize=18, fontcolor='r')
        # mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
        # plt.show()

        # TODO 支持 3 次以下
        p = 2
        q = p + 4

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )

        scalar_space = LagrangeFESpace(mesh, p=p-1, ctype='D')
        tensor_space = TensorFunctionSpace(scalar_space, shape=(-1, GD))
        ldof = tensor_space.number_of_local_dofs()
        gdof = tensor_space.number_of_global_dofs()
        from soptx.analysis.integrators.jump_penalty_integrator import JumpPenaltyIntegrator
        # from soptx.analysis.integrators.jump_penalty_integrator_2 import JumpPenaltyIntergrator2
        JPI = JumpPenaltyIntegrator(q=q, method='matrix_jump')
        # JP12 = JumpPenaltyIntergrator2(q=q)

        index, is_internal_flag = JPI.make_index(space=tensor_space)
        test = JPI.to_global_dof(tensor_space)
        KE_jump = JPI.assembly(tensor_space)

        KE_jump2 = JP12.assembly(tensor_space)
        from fealpy.fem import BilinearForm
        bform = BilinearForm(tensor_space)
        bform.add_integrator(JPI)
        K = bform.assembly()
        print("--------------")


if __name__ == "__main__":
    huzhang_analyzer = HuZhangMFEMAnalyzerTest(enable_logging=True)

    # huzhang_analyzer.run.set('test_exact_solution')
    # huzhang_analyzer.run.set('test_huzhang')
    # huzhang_analyzer.run.set('test_none_exact_solution')
    # huzhang_analyzer.run(model_type='bearing_device_2d')
    huzhang_analyzer.run.set('test_jump_penalty_integrator')

    huzhang_analyzer.run()
