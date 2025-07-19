"""测试 Hu-Zhang 元在拓扑优化下的应用"""
from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            DensityBasedMaterialConfig,
                            DensityBasedMaterialInstance,
                        )
from soptx.pde import HalfSinglePointLoadBridge2D
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter import (SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)
from soptx.opt import (ComplianceObjective, ComplianceConfig,
                       VolumeConstraint, VolumeConfig)
from soptx.opt import OCOptimizer, MMAOptimizer, save_optimization_history, plot_optimization_history

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch', 'jax']
    device: Literal['cpu', 'cuda']
    pde_type: Literal['half_single_point_load_bridge_2d']

    init_volume_fraction: float
    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['triangle_mesh', 'quadrangle_mesh']
    nx: int
    ny: int
    hx: float
    hy: float

    p: int
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
    elif config.backend == 'jax':
        bm.set_backend('jax')

    if config.pde_type == 'half_single_point_load_bridge_2d':
        pde = HalfSinglePointLoadBridge2D()

    if config.mesh_type == 'quadrangle_mesh':
        mesh = QuadrangleMesh.from_box(
                                    box=pde.domain(),
                                    nx=config.nx, ny=config.ny,
                                    device=config.device
                                )
    elif config.mesh_type == 'triangle_mesh':
        mesh = TriangleMesh.from_box(
                                box=pde.domain(), 
                                nx=config.nx, ny=config.ny,
                                device=config.device
                            )

    GD = mesh.geo_dimension()
    
    p = config.p
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    #! dof_priority-(GD, -1) 的效率比 gd_prioirty-(-1, GD) 的效率要高
    # tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    tensor_space_C = TensorFunctionSpace(space_C, (GD, -1))
    CGDOF = tensor_space_C.number_of_global_dofs()
    print(f"CGDOF: {CGDOF}")
    space_D = LagrangeFESpace(mesh=mesh, p=0, ctype='D')
    
    return pde, mesh, tensor_space_C, space_D

def test_elastic(config: TestConfig):
    pde, mesh, tensor_space_C, space_D = create_base_components(config)
    material_config = DensityBasedMaterialConfig(
                            elastic_modulus=pde.E,            
                            minimal_modulus=1e-9,         
                            poisson_ratio=pde.nu,            
                            plane_type=pde.plane_type,    
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    materials = DensityBasedMaterialInstance(config=material_config)

    solvers = ElasticFEMSolver(
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
    def density_func(x):
        val = bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)

    KK = solvers.get_base_local_stiffness_matrix()

    solvers.update_status(rho[:])
    solver_result = solvers.solve()
    uh = solver_result.displacement
    print("---------------------")

def run(config: TestConfig) -> Dict[str, Any]:
    pde, mesh, tensor_space_C, space_D = create_base_components(config)

    material_config = DensityBasedMaterialConfig(
                            elastic_modulus=pde.E,            
                            minimal_modulus=1e-9,         
                            poisson_ratio=pde.nu,            
                            plane_type=pde.plane_type,
                            device=config.device,      
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = DensityBasedMaterialInstance(config=material_config)

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
        val = config.init_volume_fraction * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)

    obj_config = ComplianceConfig(diff_mode=config.diff_mode)
    objective = ComplianceObjective(solver=solver, config=obj_config)
    cons_config = VolumeConfig(diff_mode=config.diff_mode)
    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction,
                                config=cons_config)

    if config.filter_type == 'None':
        filter = None
    elif config.filter_type == 'sensitivity':
        filter = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius, domain=pde.domain()) 
    elif config.filter_type == 'density':
        filter = DensityBasicFilter(mesh=mesh, rmin=config.filter_radius, domain=pde.domain())
    elif config.filter_type == 'heaviside':
        filter = HeavisideProjectionBasicFilter(mesh=mesh, rmin=config.filter_radius, domain=pde.domain())  


    if config.optimizer_type == 'oc':
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
                                bisection_tol=1e-3
                            )
    elif config.optimizer_type == 'mma':
        NC = mesh.number_of_cells()
        optimizer = MMAOptimizer(
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
                                m=1,
                                n=NC,
                                xmin=bm.zeros((NC, 1)),
                                xmax=bm.ones((NC, 1)),
                                a0=1,
                                a=bm.zeros((1, 1)),
                                c=1e4 * bm.ones((1, 1)),
                                d=bm.zeros((1, 1)),
                            )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    rho_opt, history = optimizer.optimize(rho=rho[:])
    
    # Save results
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_optimization_history(mesh, history, str(save_path))
    plot_optimization_history(history, save_path=str(save_path))
    
    return {
        'optimal_density': rho_opt,
        'history': history,
        'mesh': mesh
    }

if __name__ == "__main__":
    current_file = Path(__file__)
    base_dir = current_file.parent.parent / 'vtu'
    base_dir = str(base_dir)

    backend = 'numpy'
    # backend = 'pytorch'
    # backend = 'jax'
    device = 'cpu'

    '''参数来源论文: Topology optimization of incompressible media using mixed finite elements'''
    pde_type = 'half_single_point_load_bridge_2d'
    init_volume_fraction = 0.35
    volume_fraction = 0.35
    mesh_type = 'quadrangle_mesh'
    nx, ny = 32, 32
    optimizer_type = 'oc'
    # optimizer_type = 'mma'
    # filter_type = None
    # filter_type = 'sensitivity'
    filter_type = 'density'
    filter_radius = nx * 0.04
    fem_p = 1
    config_basic_filter = TestConfig(
        backend=backend,
        device=device,
        pde_type=pde_type,
        init_volume_fraction=init_volume_fraction,
        volume_fraction=volume_fraction,
        penalty_factor=3.0,
        mesh_type=mesh_type, nx=nx, ny=ny, hx=1, hy=1,
        p = fem_p,
        assembly_method=AssemblyMethod.FAST,
        solver_type='direct', solver_params={'solver_type': 'mumps'},
        # solver_type='cg', solver_params={'maxiter': 2000, 'atol': 1e-12, 'rtol': 1e-12},
        diff_mode='manual',
        # diff_mode='auto',
        optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
        filter_type=filter_type, filter_radius=filter_radius,
        save_dir=f'{base_dir}/{pde_type}_{mesh_type}_{optimizer_type}_{filter_type}_p{fem_p}',
        )
    result = run(config_basic_filter)   

    # result = test_elastic(config_basic_filter)