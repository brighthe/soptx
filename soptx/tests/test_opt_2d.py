"""测试 opt 模块中的 oc 和 mma 类以及 filter 模块."""

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

    optimizer_type: Literal['oc', 'mma']
    max_iterations: int
    tolerance: float

    save_dir: Union[str, Path]

    # filter_type: Literal['sensitivity', 'density', 'heaviside']
    # filter_radius: float
    # projection_beta: Optional[float]

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
        extent = [0, config.nx, 0, config.ny]
        h = [1.0, 1.0]
        origin = [0.0, 0.0]
        pde = Cantilever2dData1(
                    xmin=0, xmax=extent[1] * h[0],
                    ymin=0, ymax=extent[3] * h[1]
                )
        if config.mesh_type == 'uniform_mesh_2d':
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

    objective = ComplianceObjective(solver=solver)
    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction)
    
    return rho, objective, constraint

def run_optimization_test(config: TestConfig) -> Dict[str, Any]:
    """Run topology optimization test with given configuration."""
    rho, objective, constraint = create_base_components(config)
    mesh = objective.solver.tensor_space.mesh

    if config.optimizer_type == 'oc':
        optimizer = OCOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=None,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                        }
                    )
        # 设置高级参数 (可选)
        optimizer.options.set_advanced_options(
                            move_limit=0.2,
                            damping_coef=0.5,
                            initial_lambda=1e9,
                            bisection_tol=1e-3)
    elif config.optimizer_type == 'mma':
        NC = mesh.number_of_cells()
        optimizer = MMAOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=None,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                            'm': 1,
                            'n': NC,
                            'xmin': bm.zeros(NC, dtype=bm.float64).reshape(-1, 1),
                            'xmax': bm.ones(NC, dtype=bm.float64).reshape(-1, 1),
                            "a0": 1,
                            "a": bm.zeros(1, dtype=bm.float64).reshape(-1, 1),
                            'c': 1e4 * bm.ones(1, dtype=bm.float64).reshape(-1, 1),
                            'd': bm.zeros(1, dtype=bm.float64).reshape(-1,),
                        }
                    )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    # Prepare optimization parameters
    opt_params = {}
    # if config.filter_type == 'heaviside' and config.projection_beta is not None:
    #     opt_params['beta'] = config.projection_beta
    
    # Run optimization
    rho_opt, history = optimizer.optimize(rho=rho[:], **opt_params)
    
    # Save results
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_optimization_history(mesh, history, str(save_path))
    
    return {
        'optimal_density': rho_opt,
        'history': history,
        'mesh': mesh
    }

if __name__ == "__main__":
    base_dir = '/home/heliang/FEALPy_Development/soptx/soptx/vtu'
    pde_type = 'cantilever_2d_2'
    filter_type = 'None'
    optimizer_type = 'oc'
    config = TestConfig(
                backend='numpy',
                pde_type=pde_type,
                elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                domain_length=3.0, domain_width=1.0,
                load=2000,
                volume_fraction=0.5,
                penalty_factor=3.0,
                mesh_type='triangle_mesh', nx=300, ny=100,
                assembly_method=AssemblyMethod.STANDARD,
                solver_type='direct', solver_params={'solver_type': 'mumps'},
                diff_mode='manual',
                optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
                save_dir=f'{base_dir}/{pde_type}_{optimizer_type}_{filter_type}',
            )
    
    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    OC 优化方法, 灵敏度滤波器
    '''
    pde_type = 'cantilever_2d_1'
    config1 = TestConfig(
                backend='numpy',
                pde_type=pde_type,
                elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                domain_length=300, domain_width=100,
                load=-1,
                volume_fraction=0.4,
                penalty_factor=3.0,
                mesh_type='uniform_mesh_2d', nx=300, ny=100,
                assembly_method=AssemblyMethod.STANDARD,
                solver_type='direct', solver_params={'solver_type': 'mumps'},
                diff_mode='manual',
                optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
                save_dir=f'{base_dir}/{pde_type}_{optimizer_type}_{filter_type}',
            )
    
    result = run_optimization_test(config1)
    