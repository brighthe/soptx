from fealpy.backend import backend_manager as bm

from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
from soptx.interpolation.interpolation_scheme import SIMPInterpolationSingle
from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial

im = IsotropicLinearElasticMaterial(youngs_modulus=1.0, 
                                    poisson_ratio=0.3, 
                                    plane_type='plane_stress',
                                    enable_logging=False)
im.display_material_params()

E = im.youngs_modulus
nu = im.poisson_ratio
lam = im.lame_lambda
mu = im.shear_modulus
K = im.bulk_modulus
plane_type = im.plane_type

im.set_material_parameters(youngs_modulus=2.0, poisson_ratio=0.6)
im.display_material_params()
im.set_plane_type(plane_type='plane_strain')
E = im.youngs_modulus
nu = im.poisson_ratio
lam = im.lame_lambda
mu = im.shear_modulus
K = im.bulk_modulus
plane_type = im.plane_type

D0 = im.elastic_matrix()

rrho = bm.ones(10) * 1/2

simpi = SIMPInterpolationSingle(penalty_factor=3.0, enable_logging=True)
p = simpi.penalty_factor

simpi.set_penalty_factor(penalty_factor=2.0)

p = simpi.penalty_factor

# 方法 1
D0_1 = simpi.interpolate(im, rrho)
rrho = simpi.relative_density

# 方法 2
simpi.set_relative_density(rrho*0.8)
D0_2 = simpi.interpolate(im)
rrho = simpi.relative_density

tom = TopologyOptimizationMaterial(base_material=im, 
                                interpolation_scheme=simpi,
                                enable_logging=True)
rd = tom.relative_density
pf = tom.penalty_factor

tom.set_relative_density(rrho*0.8)
tom.set_penalty_factor(2.5)

rd1 = tom.relative_density
pf1 = tom.penalty_factor

D0_3 = tom.elastic_matrix()

tom.display_material_info()


print("--------------")

# im.set_material_parameters(youngs_modulus=2.0, poissons_ratio=0.6       )
# im.set_plane_tyepe(plane_type='plane_stress')






# tm = TopologyOptimizationMaterial(base_material=im, 
#                                   interpolation_scheme='simp_single', 
#                                   relative_density=0.5)