from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer

# 一、基础线弹性材料求解
## 1.1 创建 pde
from soptx.pde.linear_elasticity_2d import BoxTriLagrangeData2d

pde = BoxTriLagrangeData2d(domain=[0, 1, 0, 1], 
                           E=1.0, nu=0.3,
                           enable_logging=True,
                           mesh_type='uniform_tri')
pde.init_mesh(nx=5, ny=5)
mesh = pde.mesh
E = pde.E
nu = pde.nu
plane_type = pde.plane_type
## 1.2 创建基础材料
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
base_material = IsotropicLinearElasticMaterial(
                                    youngs_modulus=E, 
                                    poisson_ratio=nu, 
                                    plane_type=plane_type,
                                    enable_logging=False
                                )


## 1.2 创建分析器
lfa = LagrangeFEMAnalyzer(pde=pde, 
                        material=base_material, 
                        assembly_method='standard',
                        solve_method='mumps'
                    )
uh = lfa.solve()
lfa.solve.set('cg')     # 设置变体 (返回 None)
uh2 = lfa.solve(maxiter=5000, atol=1e-12, rtol=1e-12)


# 2. 设置材料插值方案
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
interpolation_scheme = MaterialInterpolationScheme(
                                penalty_factor=3.0, 
                                void_youngs_modulus=1e-12, 
                                interpolation_method='simp', 
                                enable_logging=False
                            )

# 3. 创建简单网格
from soptx.pde.mbb_beam_2d import HalfMBBBeam2dData1
pde = HalfMBBBeam2dData1(domain=[0, 2, 0, 1], T=-1.0, E=1.0, nu=0.3)
pde.init_mesh.set('uniform_quad')
mesh = pde.init_mesh(nx=4, ny=2)
print(f"网格信息: {mesh.number_of_cells()} 个单元")

# 4. 创建拓扑优化材料
from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial
topm = TopologyOptimizationMaterial(
                        mesh=mesh,
                        base_material=base_material,
                        interpolation_scheme=interpolation_scheme,
                        relative_density=0.5,
                        density_location='element',
                        enable_logging=False
                    )

lfa = LagrangeFEMAnalyzer(pde=pde, material=topm, solve_method='mumps')