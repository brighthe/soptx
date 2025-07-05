from fealpy.backend import backend_manager as bm



from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial




# E = im.youngs_modulus
# nu = im.poisson_ratio
# lam = im.lame_lambda
# mu = im.shear_modulus
# K = im.bulk_modulus
# plane_type = im.plane_type

# im.set_material_parameters(youngs_modulus=2.0, poisson_ratio=0.6)
# im.display_material_params()
# im.set_plane_type(plane_type='plane_strain')
# E = im.youngs_modulus
# nu = im.poisson_ratio
# lam = im.lame_lambda
# mu = im.shear_modulus
# K = im.bulk_modulus
# plane_type = im.plane_type

# D0 = im.elastic_matrix()

# rrho = bm.ones(10) * 1/2


# p = simpi.penalty_factor

# simpi.set_penalty_factor(penalty_factor=2.0)

# p = simpi.penalty_factor

# # 方法 1
# D0_1 = simpi.interpolate(im, rrho)
# rrho = simpi.relative_density

# # 方法 2
# simpi.set_relative_density(rrho*0.8)
# D0_2 = simpi.interpolate(im)
# rrho = simpi.relative_density

# tom.set_relative_density(rrho*0.8)
# tom.set_penalty_factor(2.5)

# rd1 = tom.relative_density
# pf1 = tom.penalty_factor

# D0_3 = tom.elastic_matrix()

# tom.display_material_info()

# 设置基础线弹性材料
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
ilem = IsotropicLinearElasticMaterial(youngs_modulus=1.0, 
                                    poisson_ratio=0.3, 
                                    plane_type='plane_stress',
                                    enable_logging=False)
ilem.display_material_params()

# 设置材料插值方案
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
simpi = MaterialInterpolationScheme(penalty_factor=3.0, 
                                    void_youngs_modulus=1e-12, 
                                    interpolation_method='simp', 
                                    enable_logging=True)
p = simpi.penalty_factor
Emin = simpi.void_youngs_modulus
im = simpi.interpolation_method


# 设置拓扑优化材料
topm = TopologyOptimizationMaterial(base_material=ilem, 
                                interpolation_scheme=simpi,
                                enable_logging=True)
rd = topm.relative_density
pf = topm.penalty_factor


# 设置模型
from soptx.pde.mbb_beam_2d import HalfMBBBeam2dData1
hmb1 = HalfMBBBeam2dData1(domain=[0, 60, 0, 20],
                        mesh_method='uniform_quad',
                        T=-1.0, E=1.0, nu=0.3)
mesh = hmb1.init_mesh()
nx, ny = mesh.meshdata['nx'], mesh.meshdata['ny']
hmb1.init_mesh.set('uniform_tri')
mesh2 = hmb1.init_mesh(nx=30, ny=10)
nx1, ny1 = mesh2.meshdata['nx'], mesh2.meshdata['ny']

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
LFA = LagrangeFEMAnalyzer(pde=hmb1, 
                            material=tom, 
                            space_degree=1, 
                            integration_method='standard', 
                            solve_method='mumps')


print("--------------")

# im.set_material_parameters(youngs_modulus=2.0, poissons_ratio=0.6       )
# im.set_plane_tyepe(plane_type='plane_stress')






# tm = TopologyOptimizationMaterial(base_material=im, 
#                                   interpolation_scheme='simp_single', 
#                                   relative_density=0.5)