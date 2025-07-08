

# 二、拓扑优化材料求解
## 2.1 创建 pde 并设置网格
from soptx.model.mbb_beam_2d import HalfMBBBeam2dData1
pde = HalfMBBBeam2dData1(
                    domain=[0, 60, 0, 20],
                    T=-1.0, E=1.0, nu=0.3,
                    enable_logging=False
                )
pde.init_mesh.set('uniform_quad')
mesh = pde.init_mesh(nx=60, ny=20)

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
                        enable_logging=False
                        )

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

# 2.6 创建柔顺度目标函数
from soptx.optimization.compliance_objective import ComplianceObjective
compliance_obj = ComplianceObjective(
                        analyzer=lagrange_fem,
                        enable_logging=True
                    )
c = compliance_obj.fun(density_distribution=top_material.density_distribution)