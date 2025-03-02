from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.pde import Bridge2dData1
from soptx.material import (
                            LevelSetMaterialConfig,
                            LevelSetMaterialInstance,
                        )
from soptx.solver import ElasticFEMSolver, AssemblyMethod
from soptx.opt import ComplianceObjective, VolumeConstraint

pde = Bridge2dData1(xmin=0, xmax=60,
                    ymin=0, ymax=30,
                    T=1)
mesh = UniformMesh2d(extent=[0, 60, 0, 30], h=[1, 1], origin=[0, 0],
                    ipoints_ordering='yx', device='cpu')
GD = mesh.geo_dimension()
p = 1
space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
print(f"CGDOF: {tensor_space_C.number_of_global_dofs()}")
space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
print(f"DGDOF: {space_D.number_of_global_dofs()}")

material_config = LevelSetMaterialConfig(
                        elastic_modulus=1,
                        minimal_modulus=1e-5,                 
                        poisson_ratio=0.3,            
                        plane_assumption="plane_stress",    
                    )
materials = LevelSetMaterialInstance(config=material_config)
node = mesh.entity('node')
kwargs = bm.context(node)
@cartesian
def density_func(x: TensorLike):
    val = bm.ones(x.shape[0], **kwargs)
    return val
rho = space_D.interpolate(u=density_func)

solver = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=AssemblyMethod.FAST,
                solver_type='direct',
                solver_params={'solver_type': 'mumps'}, 
            )

objective = ComplianceObjective(solver=solver)
obj_value = objective.fun(rho=rho[:], u=None)
volreq = 0.3
constraint = VolumeConstraint(solver=solver, 
                            volume_fraction=volreq
                            )
volfrac = constraint.get_volume_fraction(rho=rho[:])

print("--------------------------")