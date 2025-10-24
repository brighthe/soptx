from typing import Optional, Union
from pathlib import Path
from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.compliant_mechanism_objective import CompliantMechanismObjective


class CompliantMechanismObjectiveTester(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.compliant_mechanism_objective = None

    @variantmethod('test_compliant_mechanism_none_exact_solution_lfem_hzmfem')
    def run(self, model: str) -> None:
        """对于无真解的算例, 分别采用位移法和混合元方法的结果计算目标函数"""

        if model == 'displacement_inverter_2d':
            E = 1.0
            nu = 0.3  
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'

            from soptx.model.displacement_inverter_2d import DisplacementInverter2d
            pde = DisplacementInverter2d(
                        domain=[0, 40, 0, 20],
                        f_in=1.0,
                        f_out=-1.0,
                        k_in=0.1,
                        k_out=0.1,
                        E=E, nu=nu,
                        plane_type=plane_type,
                    )
            nx, ny = 40, 20
            pde.init_mesh.set('uniform_quad')
        
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)
        NN = displacement_mesh.number_of_nodes()
        NE = displacement_mesh.number_of_edges()
        NC = displacement_mesh.number_of_cells()
        GD = displacement_mesh.geo_dimension()

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # displacement_mesh.add_plot(axes)
        # displacement_mesh.find_node(axes, showindex=True, color='g', markersize=12, fontsize=16, fontcolor='g')
        # displacement_mesh.find_edge(axes, showindex=True, color='r', markersize=14, fontsize=18, fontcolor='r')
        # displacement_mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
        # plt.show()

    
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        # 'element', 'node'
        density_location = 'element'
        penalty_factor = 3.0
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='simp',
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
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
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
        space = lagrange_fem_analyzer.tensor_space
        TGDOF_uh = space.number_of_global_dofs()
        self._log_info(f"分析阶段参数, "
                    f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                    f"离散方法={lagrange_fem_analyzer.__class__.__name__}, "
                    f"空间={space.__class__.__name__}, 次数={space.p}, 总自由度={TGDOF_uh}")

        # uh = lagrange_fem_analyzer.solve_displacement(density_distribution=None, adjoint=True)

        relative_density = 0.3
        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                ) 

        state_variable = 'u'
        cmo_lfem = CompliantMechanismObjective(analyzer=lagrange_fem_analyzer)
        c_lfem = cmo_lfem.fun(density=rho)

        dc_lfem = cmo_lfem.jac(density=rho, diff_mode='manual')

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)

        uh_component = uh.reshape(GD, NN).T
        displacement_mesh.nodedata['uh'] = uh_component

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
        self._log_info(f"分析阶段参数, "
                    f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                    f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 位移总自由度={TGDOF_uh}, "
                    f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, "
                    f"应力总自由度={TGDOF_sigmah}, 节点自由度={TGDOF_sigmah_n}, 边自由度={TGDOF_sigmah_e}, 单元自由度={TGDOF_sigmah_c}")
        
        sigmah_hz, uh_hz = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)

        state_variable = 'u'
        co_hzmfem = ComplianceObjective(analyzer=huzhang_mfem_analyzer, state_variable=state_variable)
        c_hzmfem = co_hzmfem.fun(density=None)

        uh_hz_component = uh_hz.reshape(TLDOF_uh, NC).T 
        displacement_mesh.celldata['uh_hzmfem'] = uh_hz_component
        displacement_mesh.to_vtk(f"{base_dir}/test_uh_none_exact_solution_concentrated_lfem_hzmfem.vtu")

        print(f"--------------")

    @run.register('test_compliance_exact_solution_lfem_hzmfem')
    def run(self, model: str) -> None:
        """对于有真解的算例, 分别采用位移法和混合元方法的结果计算目标函数"""
        if model == 'tri_sol_mix_huzhang':
            lam = 1.0
            mu = 0.5
            from soptx.model.linear_elasticity_2d import TriSolMixHuZhangData
            pde = TriSolMixHuZhangData(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
            displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        elif model == 'tri_sol_dir_huzhang':
            lam = 1.0
            mu = 0.5
            from soptx.model.linear_elasticity_2d import TriSolDirHuZhangData
            pde = TriSolDirHuZhangData(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 64, 64
            # nx, ny = 128, 128
            displacement_mesh = pde.init_mesh(nx=nx, ny=ny)

        NN = displacement_mesh.number_of_nodes()
        NE = displacement_mesh.number_of_edges()
        NC = displacement_mesh.number_of_cells()
        GD = displacement_mesh.geo_dimension()

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
        
        state_variable = 'u'
        co_lfem = ComplianceObjective(analyzer=lagrange_fem_analyzer, state_variable=state_variable)
        c_lfem = co_lfem.fun(density=None)

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)

        uh_component = uh.reshape(GD, NN).T
        displacement_mesh.nodedata['uh'] = uh_component

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
        self._log_info(f"分析阶段参数, "
                    f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                    f"位移空间={space_uh.__class__.__name__}, 次数={space_uh.p}, 位移总自由度={TGDOF_uh}, "
                    f"应力空间={space_sigmah.__class__.__name__}, 次数={space_sigmah.p}, "
                    f"应力总自由度={TGDOF_sigmah}, 节点自由度={TGDOF_sigmah_n}, 边自由度={TGDOF_sigmah_e}, 单元自由度={TGDOF_sigmah_c}")
        
        sigmah_hz, uh_hz = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)
        e_uh_hz_l2 = displacement_mesh.error(u=uh_hz, 
                                v=pde.disp_solution,
                                q=integration_order) # 位移 L2 范数误差
        state_variable = 'u'
        co_hzmfem = ComplianceObjective(analyzer=huzhang_mfem_analyzer, state_variable=state_variable)
        c_hzmfem = co_hzmfem.fun(density=None)

        uh_hz_component = uh_hz.reshape(TLDOF_uh, NC).T 
        displacement_mesh.celldata['uh_hzmfem'] = uh_hz_component
        displacement_mesh.to_vtk(f"{base_dir}/test_uh_exact_solution_dir_lfem_hzmfem.vtu")

        print('------------------')


if __name__ == '__main__':

    compliant_mechanism_objective = CompliantMechanismObjectiveTester(enable_logging=True)

    compliant_mechanism_objective.run.set('test_compliance_none_exact_solution_lfem_hzmfem')
    compliant_mechanism_objective.run(model='displacement_inverter_2d')

    # compliant_mechanism_objective.run.set('test_compliance_exact_solution_lfem_hzmfem')
    # compliant_mechanism_objective.run(model='tri_sol_dir_huzhang')