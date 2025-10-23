from typing import Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.interpolation.linear_elastic_material import LinearElasticMaterial


class LagrangeFEMAnalyzerTest(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)


    @variantmethod('test_exact_solution_lfem')
    def run(self, model: str) -> TensorLike:
        """基于有真解的算例验证拉格朗日位移有限元的正确性"""
        if model == 'tri_sol_dir_huzhang':
            from soptx.model.linear_elasticity_2d import TriSolDirHuZhangData
            lam, mu = 1.0, 0.5
            pde = TriSolDirHuZhangData(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
            mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                    lame_lambda=pde.lam, 
                                    shear_modulus=pde.mu,
                                    plane_type=pde.plane_type,
                                    enable_logging=False
                                )
            
        elif model == 'tri_sol_mix_homo_dir_huzhang':
            # 齐次 Dirichlet + 非齐次 Neumann
            lam, mu = 1.0, 0.5
            from soptx.model.linear_elasticity_2d import TriSolMixHomoDirHuZhang
            pde = TriSolMixHomoDirHuZhang(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
            mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam, 
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )
        
        elif model == 'tri_sol_mix_nhomo_dir_huzhang':
            # 非齐次 Dirichlet + 非齐次 Neumann
            lam, mu = 1.0, 0.5
            from soptx.model.linear_elasticity_2d import TriSolMixNHomoDirHuZhang
            pde = TriSolMixNHomoDirHuZhang(domain=[0, 1, 0, 1], lam=lam, mu=mu)
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 2, 2
            mesh = pde.init_mesh(nx=nx, ny=ny)
            from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
            material = IsotropicLinearElasticMaterial(
                                                lame_lambda=pde.lam, 
                                                shear_modulus=pde.mu,
                                                plane_type=pde.plane_type,
                                                enable_logging=False
                                            )
        
        elif model == 'BoxTrDirichleti2d':
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
        
        elif model == 'BoxTriMixed2d':
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

        elif model == 'BoxPoly3d':
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
        integration_order = space_degree + 4

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, "
                    f"空间次数={space_degree}, 积分阶数={integration_order}")

        maxit = 5
        errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{\\Omega, 0}$', 
                     '$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{\\Omega, 1}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            N = 2**(i+1)

            lfa = LagrangeFEMAnalyzer(
                                    mesh=mesh,
                                    pde=pde, 
                                    material=material, 
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method='standard',
                                    solve_method='mumps',
                                    topopt_algorithm=None,
                                    interpolation_scheme=None
                                )
                    
            uh = lfa.solve_displacement(rho_val=None)

            NDof[i] = lfa.tensor_space.number_of_global_dofs()

            e_l2 = mesh.error(uh, pde.disp_solution)
            e_h1 = mesh.error(uh.grad_value, pde.grad_disp_solution)

            h[i] = 1/N
            errorMatrix[0, i] = e_l2
            errorMatrix[1, i] = e_h1

            if i < maxit - 1:
                mesh.uniform_refine()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
        print("order_h1:\n", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))

        GD = mesh.geo_dimension()
        NN = mesh.number_of_nodes()
        uh_component = uh.reshape(GD, NN).T  # (NN, GD)
        mesh.nodedata['uh'] = uh_component

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        mesh.to_vtk(f"{base_dir}/{model}_uh_lfem.vtu") 

        import matplotlib.pyplot as plt
        from soptx.utils.show import showmultirate, show_error_table
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

        return uh


    @run.register('test_none_exact_solution_lfem')
    def run(self, model) -> TensorLike:
        if model == 'bearing_device_2d':
            E = 100.0
            nu = 0.4   # 可压缩
            plane_type = 'plane_strain'  # 'plane_stress' or 'plane_strain'
            
            from soptx.model.bearing_device_2d import HalfBearingDevice2D
            pde = HalfBearingDevice2D(
                                domain=[0, 60, 0, 40],
                                t=-1.8,
                                E=E, nu=nu,
                                plane_type=plane_type,
                            )
            pde.init_mesh.set('uniform_aligned_tri')
            nx, ny = 60, 40

        elif model == 'clamped_beam_2d':
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

        elif model == 'disp_inverter_2d':
            E = 1.0
            nu = 0.3  # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'

            from soptx.model.displacement_inverter_2d import DisplacementInverter2d
            pde = DisplacementInverter2d(
                        domain=[0, 40, 0, 20],
                        f_in=1.0,
                        f_out=-1.0,
                        k_in=1.0,
                        k_out=1.0,
                        E=E, nu=nu,
                        plane_type=plane_type,
                    )
            pde.init_mesh.set('uniform_quad')
            nx, ny = 4, 2

        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)
        NN = displacement_mesh.number_of_nodes()
        GD = displacement_mesh.geo_dimension()

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.gca()
        # displacement_mesh.add_plot(axes)
        # displacement_mesh.find_node(axes, showindex=True, markersize=10, fontsize=12, fontcolor='r')
        # displacement_mesh.find_edge(axes, showindex=True, markersize=14, fontsize=16, fontcolor='g')
        # displacement_mesh.find_cell(axes, showindex=True, markersize=16, fontsize=20, fontcolor='b')
        # plt.show()

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E,
                                            poisson_ratio=pde.nu,
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        
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
        
        s_space = lagrange_fem_analyzer.scalar_space
        SGDOF = s_space.number_of_global_dofs()
        t_space = lagrange_fem_analyzer.tensor_space
        TGDOF = t_space.number_of_global_dofs()

        self._log_info(f"模型名称={pde.__class__.__name__}, 平面类型={pde.plane_type}, 外载荷类型={pde.load_type}, 边界类型={pde.boundary_type}, \n"
                       f"网格={displacement_mesh.__class__.__name__}, 节点数={NN}, "
                       f"空间={t_space.__class__.__name__}, 次数={t_space.p}, 标量自由度={SGDOF}, 总自由度={TGDOF}")

        uh = lagrange_fem_analyzer.solve_displacement(rho_val=None, adjoint=False)

        uh_component = uh.reshape(GD, NN).T  # (NN, GD)
        displacement_mesh.nodedata['uh'] = uh_component

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        displacement_mesh.to_vtk(f"{base_dir}/{model}_uh_lfem.vtu") 

        # 计算应变和应力
        q = lagrange_fem_analyzer._integration_order
        qf = displacement_mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = s_space.grad_basis(bc=bcs, variable='x')
        cell2dof = t_space.cell_to_dof()
        cuh = uh[cell2dof]

        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        B = material.strain_displacement_matrix(dof_priority=t_space.dof_priority, 
                                                            gphi=gphi)
        D = material.elastic_matrix()
        strain_quadrature = bm.einsum('cqil, cl -> cqi', B, cuh) # (NC, NQ, 3)
        stress_quadrature = bm.einsum('cqij, cqjl, cl -> cqi', D, B, cuh) # (NC, NQ, 3)

        # 计算单元加权平均应变和应力
        cell_measure = displacement_mesh.entity_measure('cell')  # (NC,)
        strain_cell = bm.mean(strain_quadrature, axis=1) # (NC, 3)
        stress_cell = bm.mean(stress_quadrature, axis=1) # (NC, 3)
        weighted_strain_cell = strain_cell * cell_measure[:, None] # (NC, 3)
        weighted_stress_cell = stress_cell * cell_measure[:, None] # (NC, 3)

        # 计算每个节点相邻的单元数
        cell2dof = s_space.cell_to_dof()  # (NC, LDOF)
        node_weight = bm.zeros(SGDOF, dtype=bm.float64)
        node_weight[:] = bm.add_at(node_weight, cell2dof.reshape(-1), bm.repeat(cell_measure, cell2dof.shape[1]))

        # 计算节点应变分量和节点应力分量
        strain_node = bm.zeros((SGDOF, 3), dtype=bm.float64)
        stress_node = bm.zeros((SGDOF, 3), dtype=bm.float64)
        for i in range(3):
            strain_node[:, i] = bm.add_at(strain_node[..., i], cell2dof.reshape(-1), bm.repeat(weighted_strain_cell[:, i], cell2dof.shape[1]))
            stress_node[:, i] = bm.add_at(stress_node[..., i], cell2dof.reshape(-1), bm.repeat(weighted_stress_cell[:, i], cell2dof.shape[1]))

        strain_node /= node_weight[:, None]
        stress_node /= node_weight[:, None]

        from soptx.analysis.utils import _get_val_tensor_to_component
        uh_component = _get_val_tensor_to_component(val=uh, space=t_space) # (SGDOF, GD)
        displacement_mesh.nodedata['uh'] = uh_component
        displacement_mesh.nodedata['stress'] = stress_node

        from pathlib import Path
        current_file = Path(__file__)
        base_dir = current_file.parent.parent / 'vtu'
        base_dir = str(base_dir)
        displacement_mesh.to_vtk(f"{base_dir}/uh_lfem.vtu")

        return uh


if __name__ == "__main__":
    test = LagrangeFEMAnalyzerTest(enable_logging=True)
    
    test.run.set('test_exact_solution_lfem')
    test.run(model='tri_sol_mix_homo_dir_huzhang')

    # test.run.set('test_none_exact_solution_lfem')
    # test.run(model='bearing_device_2d')