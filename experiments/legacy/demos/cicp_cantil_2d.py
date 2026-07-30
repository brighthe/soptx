from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (DensityBasedMaterialConfig, DensityBasedMaterialInstance)
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter_ import SensitivityBasicFilter
from soptx.opt import (ComplianceObjective, ComplianceConfig,
                       VolumeConstraint, VolumeConfig)
from soptx.opt import OCOptimizer

from soptx.pde import Cantilever2dData1

pde = Cantilever2dData1(xmin=0, xmax=160,
                        ymin=0, ymax=100,
                        T = -1)

mesh = UniformMesh(extent=[0, 160, 0, 100], h=[1, 1], origin=[0, 0])

space_C = LagrangeFESpace(mesh=mesh, p=1, ctype='C')
tensor_space_C = TensorFunctionSpace(scalar_space=space_C, shape=(-1, 2))
space_D = LagrangeFESpace(mesh=mesh, p=0, ctype='D')

material_config = DensityBasedMaterialConfig(
                            elastic_modulus=1.0,            
                            minimal_modulus=1e-9,         
                            poisson_ratio=0.3,            
                            plane_assumption="plane_stress",    
                            interpolation_model="SIMP",    
                            penalty_factor=3.0)
materials = DensityBasedMaterialInstance(config=material_config)

solver = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=AssemblyMethod.STANDARD,
                solver_type='direct',
                solver_params={'solver_type': 'mumps'})
sens_filter = SensitivityBasicFilter(mesh=mesh, rmin=6.0)

objective = ComplianceObjective(solver=solver)
constraint = VolumeConstraint(solver=solver, volume_fraction=0.4)

optimizer = OCOptimizer(objective=objective,
                        constraint=constraint,
                        filter=sens_filter,
                        options={'max_iterations': 200, 'tolerance': 0.01})

if __name__ == "__main__":
    @cartesian
    def density_func(x):
        # val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        val = bm.ones(x.shape[0])
        return val
    rho = space_D.interpolate(u=density_func)
    rho_opt, history = optimizer.optimize(rho=rho[:])
    
    from soptx.opt import save_optimization_history, plot_optimization_history
    save_optimization_history(mesh, history)
    plot_optimization_history(history)