

# 二、拓扑优化材料求解
## 2.1 创建 pde 并设置网格
from soptx.model.mbb_beam_2d import HalfMBBBeam2dData1
pde = HalfMBBBeam2dData1(
                    domain=[0, 60, 0, 20],
                    T=-1.0, E=1.0, nu=0.3,
                    enable_logging=False
                )
pde.init_mesh.set('uniform_quad')
mesh = pde.init_mesh(nx=1, ny=1)

## 2.2 创建基础材料
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
base_material = IsotropicLinearElasticMaterial(
                                    youngs_modulus=pde.E, 
                                    poisson_ratio=pde.nu, 
                                    plane_type=pde.plane_type,
                                    enable_logging=False
                                )

# 2.3 设置材料插值方案
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
interpolation_scheme = MaterialInterpolationScheme(
                            penalty_factor=3.0, 
                            void_youngs_modulus=1e-12, 
                            interpolation_method='simp', 
                            enable_logging=False
                        )

# 2.4. 创建拓扑优化材料
from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial
top_material = TopologyOptimizationMaterial(
                        mesh=mesh,
                        base_material=base_material,
                        interpolation_scheme=interpolation_scheme,
                        relative_density=1.0,
                        density_location='element_gauss_integrate_point',
                        quadrature_order=3,
                        enable_logging=True
                        )

density_distribution = top_material.density_distribution

# 2.5 创建有限元分析器
from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
lagrange_fem = LagrangeFEMAnalyzer(
                        pde=pde,
                        material=top_material,
                        space_degree=1,
                        assembly_method='standard',
                        solve_method='mumps',
                        enable_logging=True
                    )
mesh = lagrange_fem.mesh
ip1 = mesh.interpolation_points(p=3)




scalar_space = lagrange_fem.scalar_space



# 2.6 创建柔顺度目标函数
from soptx.optimization.compliance_objective import ComplianceObjective
compliance_obj = ComplianceObjective(
                        analyzer=lagrange_fem,
                        enable_logging=True
                    )
c = compliance_obj.fun(density_distribution=top_material.density_distribution)

dc = compliance_obj.jac(density_distribution=top_material.density_distribution, 
                        diff_mode='manual')



lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                        mesh=mesh,
                                        pde=pde,
                                        material=material,
                                        topopt_algorithm='density_based',
                                        topopt_config=DensityBasedConfig(
                                                        density_location='element',
                                                        initial_density=0.5,
                                                        interpolation=InterpolationConfig(
                                                                            method='simp',
                                                                            penalty_factor=3.0,
                                                                            target_variables=['E'],
                                                                            void_youngs_modulus=1e-9),
                                                                        )
                                                        )
uh = lagrange_fem_analyzer.solve_displacement(rho=rho)
filter_regularization = Filter(
                            mesh=mesh,
                            filter_type='sensitivity',
                            rmin=6.0,
                        )
compliance_objective = ComplianceObjective(analyzer=lagrange_fem_analyzer)
c = compliance_objective.fun(rho=rho, u=uh)
dc = compliance_objective.jac(rho=rho, u=uh)
volume_constraint = VolumeConstraint(volume_fraction=0.4)
v = volume_constraint.fun(rho=rho)
dv = volume_constraint.jac(rho=rho)

oc_optimizer = OCOptimizer(
                    objective=compliance_objective,
                    constraint=volume_constraint,
                    regularization=filter_regularization, 
                    options={
                        'max_iterations': 100,
                        'tolerance': 1e-5,
                        'initial_lambda': 1.0
                    }
                )

rho_new, history = oc_optimizer.optimize(rho=rho)
save_optimization_history(mesh, history, str(save_path))
plot_optimization_history(history, save_path=str(save_path))