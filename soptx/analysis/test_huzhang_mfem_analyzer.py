from typing import Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from fealpy.mesh import TriangleMesh, HexahedronMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function


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
            from soptx.model.linear_elasticity_2d import TriSolDirHuZhangData
            lam, mu = 1.0, 0.5
            pde = TriSolDirHuZhangData(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2

        elif model == 'poly_sol_pure_homo_dir_huzhang_2d':
            # 二维纯齐次 Dirichlet
            from soptx.model.linear_elasticity_2d import PolySolPureHomoDirHuZhang2d
            lam, mu = 0.3, 0.35
            pde = PolySolPureHomoDirHuZhang2d(domain=[-1, 1, -1, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
            analysis_mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam, 
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )

        elif model == 'tri_sol_mix_homo_dir_huzhang':
            # 齐次 Dirichlet + 非齐次 Neumann
            from soptx.model.linear_elasticity_2d import TriSolMixHomoDirHuZhang
            lam, mu = 1.0, 0.5
            pde = TriSolMixHomoDirHuZhang(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2 

        elif model == 'tri_sol_mix_nhomo_dir_huzhang':
            # 非齐次 Dirichlet + 非齐次 Neumann
            from soptx.model.linear_elasticity_2d import TriSolMixNoneHomoDirHuZhang            
            lam, mu = 1.0, 0.5
            pde = TriSolMixNoneHomoDirHuZhang(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2

        elif model == 'tri_sol_pure_homo_neu_huzhang':
            # 纯齐次 Neumann
            from soptx.model.linear_elasticity_2d import TriSolPureHomoNeuHuZhang
            lam, mu = 1.0, 0.5
            pde = TriSolPureHomoNeuHuZhang(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
        
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

        # n = analysis_mesh.edge_unit_normal()
        # t = analysis_mesh.edge_tangent(unit=True)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # analysis_mesh.add_plot(axes)
        # analysis_mesh.find_node(axes, showindex=True, color='g', markersize=12, fontsize=16, fontcolor='g')
        # analysis_mesh.find_edge(axes, showindex=True, color='r', markersize=14, fontsize=18, fontcolor='r')
        # analysis_mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
        # plt.show()

        space_degree = 1
        integration_order = space_degree + 4

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                          f"空间次数={space_degree}, 积分阶数={integration_order}")
        

        maxit = 4
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


    @run.register('test_exact_solution_lfem_hzmfem')
    def run(self, model: str) -> None:
        """对于有真解的算例, 分别采用位移法和混合元方法求解"""
        if model == 'tri_sol_mix_huzhang':
            lam = 1.0
            mu = 0.5
            from soptx.model.linear_elasticity_2d import TriSolMixHuZhangData
            pde = TriSolMixHuZhangData(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 5, 5
            displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        elif model == 'tri_sol_dir_huzhang':
            lam = 1.0
            mu = 0.5
            from soptx.model.linear_elasticity_2d import TriSolDirHuZhangData
            pde = TriSolDirHuZhangData(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 128, 128
            displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            lame_lambda=pde.lam, 
                                            shear_modulus=pde.mu,
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
        ## 位移 Lagrange 有限元
        space_degree = 1
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
        space = lagrange_fem_analyzer.tensor_space
        TGDOF_uh = space.number_of_global_dofs()
        self._log_info(f"分析阶段参数, "
                    f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                    f"离散方法={lagrange_fem_analyzer.__class__.__name__}, "
                    f"空间={space.__class__.__name__}, 次数={space.p}, 总自由度={TGDOF_uh}")
        
        uh = lagrange_fem_analyzer.solve_displacement(density_distribution=None)

        e_uh_l2 = displacement_mesh.error(u=uh, 
                                        v=pde.disp_solution,
                                        q=integration_order) # 位移 L2 范数误差
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # displacement_mesh.add_plot(axes)
        # displacement_mesh.find_node(axes, showindex=True, color='g', markersize=12, fontsize=16, fontcolor='g')
        # displacement_mesh.find_edge(axes, showindex=True, color='r', markersize=14, fontsize=18, fontcolor='r')
        # displacement_mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
        # plt.show()

        ## 位移应力混合 HuZhang 有限元
        huzhang_space_degree = 2
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

        isBdDof = space_sigmah.is_boundary_dof(threshold=None, method='barycenter')


        space_uh = huzhang_mfem_analyzer.tensor_space
        TGDOF_uh = space_uh.number_of_global_dofs()
        TLDOF_uh = space_uh.number_of_local_dofs()
        TGDOF_sigmah = space_sigmah.number_of_global_dofs()
        TLDOF_sigmah_n = space_sigmah.dof.number_of_internal_local_dofs('node')
        TLDOF_sigmah_e = space_sigmah.dof.number_of_internal_local_dofs('edge')
        TLDOF_sigmah_c = space_sigmah.dof.number_of_internal_local_dofs('cell')
        NN = displacement_mesh.number_of_nodes()
        NE = displacement_mesh.number_of_edges()
        NC = displacement_mesh.number_of_cells()
        TGDOF_sigmah_n = TLDOF_sigmah_n * NN
        TGDOF_sigmah_e = TLDOF_sigmah_e * NE
        TGDOF_sigmah_c = TLDOF_sigmah_c * NC
        self._log_info(f"分析阶段参数, "
                    f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                    f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 位移总自由度={TGDOF_uh}, "
                    f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, "
                    f"应力总自由度={TGDOF_sigmah}, 节点自由度={TGDOF_sigmah_n}, 边自由度={TGDOF_sigmah_e}, 单元自由度={TGDOF_sigmah_c}")
        
        sigmah_hz, uh_hz = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)

        e_uh_hz_l2 = displacement_mesh.error(u=uh_hz, 
                                v=pde.disp_solution,
                                q=integration_order) # 位移 L2 范数误差
        
        print('------------------')


    @run.register('test_none_exact_solution_hzmfem')
    def run(self, model) -> TensorLike:
        """基于无真解的算例验证胡张混合有限元的正确性"""
        if model == 'bearing_device_2d':
            E = 100.0
            nu = 0.4   # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'
            
            from soptx.model.bearing_device_2d import HalfBearingDevice2D
            pde = HalfBearingDevice2D(
                                domain=[0, 0.6, 0, 0.4],
                                t=-1.8,
                                E=E, nu=nu,
                                plane_type=plane_type,
                            )
            pde.init_mesh.set('uniform_aligned_tri')
            # nx, ny = 60, 40
            nx, ny = 6, 4

        elif model == 'clamped_beam_2d':
            E = 30.0
            nu = 0.4  # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'

            from soptx.model.clamped_beam_2d import HalfClampedBeam2D
            # domain = [0, 80, 0, 20]
            domain = [0, 8, 0, 2]
            pde = HalfClampedBeam2D(
                    domain=domain,
                    p=-1.5,
                    E=E, nu=nu,
                    plane_type=plane_type,
                )
            pde.init_mesh.set('uniform_aligned_tri')
            # nx, ny = 80, 20
            nx, ny = 8, 2

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
        huzhang_space_degree = 1
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

        TGDOF_uh = space_uh.number_of_global_dofs()
        TLDOF_uh = space_uh.number_of_local_dofs()
        TGDOF_sigmah = space_sigmah.number_of_global_dofs()
        TLDOF_sigmah_n = space_sigmah.dof.number_of_internal_local_dofs('node')
        TLDOF_sigmah_e = space_sigmah.dof.number_of_internal_local_dofs('edge')
        TLDOF_sigmah_c = space_sigmah.dof.number_of_internal_local_dofs('cell')
        TGDOF_sigmah_n = TLDOF_sigmah_n * NN
        TGDOF_sigmah_e = TLDOF_sigmah_e * NE
        TGDOF_sigmah_c = TLDOF_sigmah_c * NC

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                       f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 位移总自由度={TGDOF_uh}, "
                       f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, "
                       f"应力总自由度={TGDOF_sigmah}, 节点自由度={TGDOF_sigmah_n}, 边自由度={TGDOF_sigmah_e}, 单元自由度={TGDOF_sigmah_c}")

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # displacement_mesh.add_plot(axes)
        # displacement_mesh.find_node(axes, showindex=True, color='g', markersize=12, fontsize=16, fontcolor='g')
        # displacement_mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
        # plt.show()
        
        # uh.shape = (NC*LDOF, );
            # p = 0: (NC*2, )
            # p = 1: (NC*6, )  
        # sigmah.shape
            # p = 1: (NN*3, )
            # p = 2: (NN*3 + NE*2 + NC*3, )
        sigmah, uh = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)

        uh_component = uh.reshape(NC, TLDOF_uh) 
        sigmah_component = sigmah.reshape(NN, TLDOF_sigmah_n)

        displacement_mesh.celldata['uh'] = uh_component
        displacement_mesh.nodedata['stress'] = sigmah_component

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        displacement_mesh.to_vtk(f"{base_dir}/uh_hzmfem.vtu")

        return uh
    

    @run.register('test_none_exact_solution_lfem_hzmfem')
    def run(self, model) -> None:
        """对于无真解的算例, 分别采用位移法和混合元方法求解"""
        if model == 'bearing_device_2d':
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
            nx, ny = 6, 4

        elif model == 'clamped_beam_2d':
            E = 30.0
            nu = 0.4  # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'

            from soptx.model.clamped_beam_2d import HalfClampedBeam2D
            domain = [0, 80, 0, 20]
            # domain = [0, 8, 0, 2]
            pde = HalfClampedBeam2D(
                    domain=domain,
                    p=-1.5,
                    E=E, nu=nu,
                    plane_type=plane_type,
                )
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 80, 20
            # nx, ny = 8, 2

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
        
        ## 位移 Lagrange 有限元
        space_degree = 1
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
        space = lagrange_fem_analyzer.tensor_space
        TGDOF_uh = space.number_of_global_dofs()

        self._log_info(f"分析阶段参数, "
                f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                f"离散方法={lagrange_fem_analyzer.__class__.__name__}, "
                f"空间={space.__class__.__name__}, 次数={space.p}, 总自由度={TGDOF_uh}")

        uh = lagrange_fem_analyzer.solve_displacement(density_distribution=None)

        ## 位移应力混合 HuZhang 有限元
        huzhang_space_degree = 1
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

        isBdDof = space_sigmah.is_boundary_dof(threshold=pde.is_neumann_boundary(), method='barycenter')


        TGDOF_uh = space_uh.number_of_global_dofs()
        TLDOF_uh = space_uh.number_of_local_dofs()
        TGDOF_sigmah = space_sigmah.number_of_global_dofs()
        TLDOF_sigmah_n = space_sigmah.dof.number_of_internal_local_dofs('node')
        TLDOF_sigmah_e = space_sigmah.dof.number_of_internal_local_dofs('edge')
        TLDOF_sigmah_c = space_sigmah.dof.number_of_internal_local_dofs('cell')
        TGDOF_sigmah_n = TLDOF_sigmah_n * NN
        TGDOF_sigmah_e = TLDOF_sigmah_e * NE
        TGDOF_sigmah_c = TLDOF_sigmah_c * NC

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                       f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 位移总自由度={TGDOF_uh}, "
                       f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, "
                       f"应力总自由度={TGDOF_sigmah}, 节点自由度={TGDOF_sigmah_n}, 边自由度={TGDOF_sigmah_e}, 单元自由度={TGDOF_sigmah_c}")

        # ipoints_uh = space_uh.interpolation_points()
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # displacement_mesh.add_plot(axes)
        # displacement_mesh.find_node(axes, node=ipoints_uh, showindex=True, color='g', markersize=12, fontsize=16, fontcolor='g')
        # # displacement_mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
        # plt.show()
        
        # uh.shape = (NC*LDOF, );
            # p = 0: (NC*2, )
            # p = 1: (NC*6, )  
        # sigmah.shape
            # p = 1: (NN*3, )
            # p = 2: (NN*3 + NE*2 + NC*3, )
        sigmah_hz, uh_hz = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        displacement_mesh.to_vtk(f"{base_dir}/uh_hzmfem.vtu")


    @run.register('test_jump_penalty_integrator')
    def run(self):
        """测试稳定化项积分子 JumpPenaltyIntegrator 的正确性"""
        from soptx.model.linear_elasticity_2d import TriSolMixHuZhangData
        pde = TriSolMixHuZhangData(lam=1, mu=0.5)
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
        p = 1
        q = p + 4

        scalar_space = LagrangeFESpace(mesh, p=p-1, ctype='D')
        tensor_space = TensorFunctionSpace(scalar_space, shape=(-1, GD))

        from soptx.analysis.integrators.jump_penalty_integrator import JumpPenaltyIntegrator
        JPI_vector = JumpPenaltyIntegrator(q=q, method='vector_jump')
        JPI_matrix = JumpPenaltyIntegrator(q=q, method='matrix_jump')

        _, vector_jump, _, _ = JPI_vector.fetch_vector_jump(tensor_space)
        _, matrix_jump, _, _ = JPI_matrix.fetch_matrix_jump(tensor_space)

        print("--------------")



if __name__ == "__main__":

    huzhang_analyzer = HuZhangMFEMAnalyzerTest(enable_logging=True)

    huzhang_analyzer.run.set('test_exact_solution_hzmfem')
    huzhang_analyzer.run(model='poly_sol_pure_homo_dir_huzhang_3d')

    # huzhang_analyzer.run.set('test_exact_solution_lfem_hzmfem')
    # huzhang_analyzer.run(model='tri_sol_mix_huzhang')

    # huzhang_analyzer.run.set('test_none_exact_solution_hzmfem')
    # huzhang_analyzer.run(model='clamped_beam_2d')

    # huzhang_analyzer.run.set('test_none_exact_solution_lfem_hzmfem')
    # huzhang_analyzer.run(model='bearing_device_2d')

    # huzhang_analyzer.run.set('test_jump_penalty_integrator')
    # huzhang_analyzer.run()
