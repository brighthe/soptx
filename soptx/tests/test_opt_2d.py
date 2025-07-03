"""测试 opt 模块中的 oc 和 mma 类以及 filter 模块."""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            ElasticMaterialConfig,
                            ElasticMaterialInstance,
                        )
from soptx.pde import (MBBBeam2dData1, 
                       Cantilever2dData1, 
                       Cantilever2dData2)
from soptx.analysis import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter import (SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.opt import OCOptimizer, MMAOptimizer, save_optimization_history
from soptx.utils import timer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']
    pde_type: Literal['cantilever_2d_1', 
                      'cantilever_2d_2', 
                      'mbb_beam_2d_1']

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
    hx: float
    hy: float
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any]

    diff_mode: Literal['auto', 'manual'] 

    optimizer_type: Literal['oc', 'mma']
    max_iterations: int
    tolerance: float

    filter_type: Literal['None', 'sensitivity', 'density', 'heaviside']
    filter_radius: float

    save_dir: Union[str, Path]

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
    elif config.pde_type == 'mbb_beam_2d_1':
        pde = MBBBeam2dData1(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    T = config.load
                )
        if config.mesh_type == 'uniform_mesh_2d':
            extent = [0, config.nx, 0, config.ny]
            origin = [0.0, 0.0]
            mesh = UniformMesh2d(
                        extent=extent, h=[config.hx, config.hy], origin=origin,
                        ipoints_ordering='yx', flip_direction='y',
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
    
    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)

    objective = ComplianceObjective(solver=solver)
    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction)
    
    return rho, objective, constraint

def run_filters_time_test(config: TestConfig):
    t = timer(f"Filter-{config.filter_type}")
    next(t) 
    rho, objective, constraint = create_base_components(config)
    mesh = objective.solver.tensor_space.mesh
    tensor_kwargs = bm.context(rho)
    rho_phys = bm.zeros_like(rho, **tensor_kwargs)
    t.send('prepare')
    filter_dens = DensityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    t.send('density_filter_init')
    filter_dens.filter_variables(x=rho[:], xPhys=rho_phys)
    t.send('density_variables')
    filter_sens = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    t.send('sensitivity_filter_init')
    filter_sens.filter_variables(x=rho[:], xPhys=rho_phys)
    t.send('sensitivity_variables')
    t.send(None)
    print("--------------------------")


def run_oc_filters_time_test(config: TestConfig) -> Dict[str, Any]:
    """Run topology optimization test with given configuration."""
    rho, objective, constraint = create_base_components(config)
    mesh = objective.solver.tensor_space.mesh

    if config.filter_type == 'None':
        filter = None
    elif config.filter_type == 'sensitivity':
        filter = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius) 
    elif config.filter_type == 'density':
        filter = DensityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    elif config.filter_type == 'heaviside':
        filter = HeavisideProjectionBasicFilter(mesh=mesh, rmin=config.filter_radius,
                                        beta=1, max_beta=512, continuation_iter=50) 
    optimizer = OCOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter,
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

    rho_opt, history = optimizer.optimize(rho=rho[:])
    
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
    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    pde_type = 'mbb_beam_2d_1'
    optimizer_type = 'oc'
    filter_type = 'sensitivity'
    nx = 150
    ny = 50
    hx = 1
    hy = 1
    config_sens_filter = TestConfig(
                            backend='numpy',
                            pde_type=pde_type,
                            elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                            domain_length=nx, domain_width=ny,
                            load=-1,
                            volume_fraction=0.5,
                            penalty_factor=3.0,
                            mesh_type='uniform_mesh_2d', nx=nx, ny=ny, hx=hy, hy=hy,
                            assembly_method=AssemblyMethod.FAST_STRESS_UNIFORM,
                            solver_type='direct', solver_params={'solver_type': 'mumps'},
                            diff_mode='manual',
                            optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
                            filter_type=filter_type, filter_radius=nx*0.04,
                            save_dir=f'{base_dir}/{pde_type}_{optimizer_type}_{filter_type}_{nx*ny}',
                        )
    filter_type = 'density'
    config_dens_filter = TestConfig(
                            backend='numpy',
                            pde_type=pde_type,
                            elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                            domain_length=nx, domain_width=ny,
                            load=-1,
                            volume_fraction=0.5,
                            penalty_factor=3.0,
                            mesh_type='uniform_mesh_2d', nx=nx, ny=ny, hx=hy, hy=hy,
                            assembly_method=AssemblyMethod.FAST_STRESS_UNIFORM,
                            solver_type='direct', solver_params={'solver_type': 'mumps'},
                            diff_mode='manual',
                            optimizer_type=optimizer_type, max_iterations=400, tolerance=0.01,
                            filter_type=filter_type, filter_radius=nx*0.04,
                            save_dir=f'{base_dir}/{pde_type}_{optimizer_type}_{filter_type}_{nx*ny}',
                        )

    # pde_type = 'cantilever_2d_2'
    # filter_type = 'None'
    # optimizer_type = 'oc'
    # config = TestConfig(
    #             backend='numpy',
    #             pde_type=pde_type,
    #             elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
    #             domain_length=3.0, domain_width=1.0,
    #             load=2000,
    #             volume_fraction=0.5,
    #             penalty_factor=3.0,
    #             mesh_type='triangle_mesh', nx=300, ny=100,
    #             assembly_method=AssemblyMethod.STANDARD,
    #             solver_type='direct', solver_params={'solver_type': 'mumps'},
    #             diff_mode='manual',
    #             optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
    #             save_dir=f'{base_dir}/{pde_type}_{optimizer_type}_{filter_type}',
    #         )
    
    # result1 = run_oc_filters_time_test(config_dens_filter)
    result2 = run_oc_filters_time_test(config_sens_filter)
    # result2 = run_filters_time_test(config_dens_filter)
    