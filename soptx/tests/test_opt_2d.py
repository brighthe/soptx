"""测试 opt 模块"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            ElasticMaterialConfig,
                            ElasticMaterialInstance,
                        )

from soptx.pde import Cantilever2dData1, Cantilever2dData2
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter import (Filter, FilterConfig)
from soptx.opt import ComplianceObjective, VolumeConstraint

from soptx.opt import OCOptimizer, save_optimization_history
from soptx.opt import MMAOptimizer

from soptx.utils import timer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']
    pde_type: Literal['cantilever_2d_1', 'cantilever_2d_2']

    elastic_modulus: float
    poisson_ratio: float
    minimal_modulus: float

    domain_length : float
    domain_width : float

    load : float

    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['uniform_mesh_2d', 'triangle_mesh']
    nx: int
    ny: int
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any]

    diff_mode: Literal['auto', 'manual'] 

    # filter_type: Literal['sensitivity', 'density', 'heaviside']
    # filter_radius: float

    # optimizer_type: Literal['oc', 'mma']
    # max_iterations: int = 100
    # tolerance: float = 0.01

def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')

    if config.pde_type == 'cantilever_2d_2':
        pde = Cantilever2dData2(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    T = config.load
                )
        if config.mesh_type == 'triangle_mesh':
            mesh = TriangleMesh.from_box(box=pde.domain(), nx=config.nx, ny=config.ny)
    elif config.pde_type == 'cantilever_2d_1':
        pde = Cantilever2dData1(
                    xmin=0, xmax=extent[1] * h[0],
                    ymin=0, ymax=extent[3] * h[1]
                )
        if config.mesh_type == 'uniform_mesh_2d':
            extent = [0, config.nx, 0, config.ny]
            h = [1.0, 1.0]
            origin = [0.0, 0.0]
            mesh = UniformMesh2d(
                        extent=extent, h=h, origin=origin,
                        ipoints_ordering='yx', flip_direction=None,
                        device='cpu'
                    )

    GD = mesh.geo_dimension()
    
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    material_config = ElasticMaterialConfig(
                            elastic_modulus=config.elastic_modulus,            
                            minimal_modulus=config.minimal_modulus,         
                            poisson_ratio=config.poisson_ratio,            
                            plane_assumption="plane_stress",    
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = ElasticMaterialInstance(config=material_config)

    solver = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=config.assembly_method,
                solver_type=config.solver_type,
                solver_params=config.solver_params 
            )
    
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return solver, rho

def run_compliane_test(config: TestConfig) -> Dict[str, Any]:
    """Run topology optimization test with given configuration."""
    solver, rho = create_base_components(config)

    objective = ComplianceObjective(solver=solver)

    obj_value = objective.fun(rho=rho[:], u=None)
    print(f"Objective function value: {obj_value:.6e}")
    
    ce = objective.get_element_compliance()
    print(f"\nElement compliance information:")
    print(f"- Shape: {ce.shape}:\n {ce}")
    print(f"- Min: {bm.min(ce):.6e}")
    print(f"- Max: {bm.max(ce):.6e}")
    print(f"- Mean: {bm.mean(ce):.6e}")
    
    dce = objective.jac(rho=rho[:], u=None, diff_mode=config.diff_mode)
    print(f"\nElement compliance_diff information:")
    print(f"- Shape: {dce.shape}:\n, {dce}")
    print(f"- Min: {bm.min(dce):.6e}")
    print(f"- Max: {bm.max(dce):.6e}")
    print(f"- Mean: {bm.mean(dce):.6e}")

def run_diff_test(config: TestConfig):
    """测试不同灵敏度计算方法."""
    solver, rho = create_base_components(config)

    objective = ComplianceObjective(solver=solver)

    t = timer(f"dce Timing")
    next(t) 
    
    dce_auto = objective.jac(rho=rho[:], u=None, diff_mode='auto')
    t.send('auto time')
    dce_manual = objective.jac(rho=rho[:], u=None, diff_mode='manual')
    t.send('manual time')
    t.send(None)

    diff = bm.max(bm.abs(dce_auto - dce_manual))
    print(f"Difference between auto and manual differentiation: {diff:.6e}")


if __name__ == "__main__":
    config = TestConfig(
                    backend='numpy',
                    pde_type='cantilever_2d_2',
                    elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                    domain_length=3.0, domain_width=1.0,
                    load=2000,
                    volume_fraction=0.5,
                    penalty_factor=3.0,
                    mesh_type='triangle_mesh', nx=300, ny=100,
                    assembly_method=AssemblyMethod.STANDARD,
                    solver_type='direct', solver_params={'solver_type': 'mumps'},
                    diff_mode='manual',
                )

    config_diff = TestConfig(
                backend='pytorch',
                pde_type='cantilever_2d_2',
                elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                domain_length=3.0, domain_width=1.0,
                load=2000,
                volume_fraction=0.5,
                penalty_factor=3.0,
                mesh_type='triangle_mesh', nx=300, ny=100,
                assembly_method=AssemblyMethod.STANDARD,
                solver_type='direct', solver_params={'solver_type': 'mumps'},
                diff_mode=None,
            )

    result = run_diff_test(config_diff)
    