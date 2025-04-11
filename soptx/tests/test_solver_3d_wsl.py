"""3D 情况下测试 solver 模块."""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh3d, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            DensityBasedMaterialConfig,
                            DensityBasedMaterialInstance,
                        )
from soptx.pde import Cantilever3dData1
from soptx.solver import ElasticFEMSolver, AssemblyMethod
from soptx.utils import timer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch', 'jax']
    device: Literal['cpu', 'cuda']
    pde_type: Literal['cantilever_3d_1']

    elastic_modulus: float
    poisson_ratio: float
    minimal_modulus: float

    domain_length : float
    domain_width : float
    domain_height : float

    load : float

    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['uniform_mesh_3d', 'tetrahedron_mesh']
    nx: int
    ny: int
    nz: int
    hx: float
    hy: float
    hz: float

    p: int
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any]


def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')
    elif config.backend == 'jax':
        bm.set_backend('jax')
    
    if config.pde_type == 'cantilever_3d_1':
        pde = Cantilever3dData1(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    zmin=0, zmax=config.domain_height,
                    T = config.load
                )
        # 创建网格时需要指定设备
        if config.mesh_type == 'uniform_mesh_3d':
            extent = [0, config.nx, 0, config.ny, 0, config.nz]
            origin = [0.0, 0.0, 0.0]
            mesh = UniformMesh3d(
                        extent=extent, h=[config.hx, config.hy, config.hz], origin=origin,
                        ipoints_ordering='zyx', device=config.device
                    )
        elif config.mesh_type == 'tetrahedron_mesh':
            mesh = TetrahedronMesh.from_box(
                        box=[0, config.domain_length, 
                             0, config.domain_width, 
                             0, config.domain_height], 
                        nx=config.nx, ny=config.ny, nz=config.nz,
                        device=config.device
                    )

    GD = mesh.geo_dimension()
    
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    print(f"CGDOF: {tensor_space_C.number_of_global_dofs()}")
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    print(f"DGDOF: {space_D.number_of_global_dofs()}")
    
    # 创建材料时需要指定设备
    material_config = DensityBasedMaterialConfig(
                            elastic_modulus=config.elastic_modulus,            
                            minimal_modulus=config.minimal_modulus,         
                            poisson_ratio=config.poisson_ratio,            
                            plane_assumption="3D",
                            device=config.device,       
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = DensityBasedMaterialInstance(config=material_config)

    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)
    
    return materials, tensor_space_C, pde, rho

def run_solve_uh_time_test(config: TestConfig):
    """测试求解位移的效率"""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    for i in range(4):
        solver.update_status(rho[:])
        solver_result = solver.solve()
        uh = solver_result.displacement

if __name__ == "__main__":
    '''
    参数来源论文: An efficient 3D topology optimization code written in Matlab
    '''
    # backend = 'numpy'
    backend = 'pytorch'
    # backend = 'jax'
    device = 'cpu'
    # device = 'cuda'
    pde_type = 'cantilever_3d_1'
    # mesh_type = 'tetrahedron_mesh'
    mesh_type = 'uniform_mesh_3d'
    optimizer_type = 'oc'
    filter_type = 'sensitivity'
    # nx, ny, nz = 60, 20, 4
    # filter_radius = 1.5
    nx, ny, nz = 120, 40, 8
    filter_radius = 3.0
    config_uh_time = TestConfig(
                    backend=backend,
                    device=device,
                    pde_type=pde_type,
                    elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                    domain_length=nx, domain_width=ny, domain_height=nz,
                    load=-1,
                    volume_fraction=0.3,
                    penalty_factor=3.0,
                    mesh_type=mesh_type, nx=nx, ny=ny, nz=nz, hx=1, hy=1, hz=1,
                    p=1,
                    assembly_method=AssemblyMethod.FAST,
                    # assembly_method=AssemblyMethod.STANDARD,
                    # solver_type='direct', solver_params={'solver_type': 'mumps'},
                    # solver_type='direct', solver_params={'solver_type': 'cupy'},
                    solver_type='cg', solver_params={'maxiter': 5000, 'atol': 1e-12, 'rtol': 1e-12},
                )
    result = run_solve_uh_time_test(config=config_uh_time)




    # mesh_type = 'tetrahedron_mesh'
    # nx, ny, nz = 3, 2, 2
    # config_assmeble_exact = TestConfig(
    #                     backend='numpy',
    #                     pde_type='cantilever_3d_1',
    #                     elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
    #                     domain_length=nx, domain_width=ny, domain_height=nz,
    #                     load=-1,
    #                     volume_fraction=0.3,
    #                     penalty_factor=3.0,
    #                     mesh_type=mesh_type, nx=nx, ny=ny, nz=nz, hx=1, hy=1, hz=1,
    #                     p=1,
    #                     assembly_method=None,
    #                     solver_type='direct', 
    #                     solver_params={'solver_type': 'mumps'},
    #                     )
    # nx, ny, nz = 60, 20, 4
    # hx, hy, hz = 1, 1, 1
    # mesh_type = 'uniform_mesh_3d'
    # config_solver_time = TestConfig(
    #                         backend='numpy',
    #                         pde_type='cantilever_3d_1',
    #                         elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
    #                         domain_length=nx, domain_width=ny, domain_height=nz,
    #                         load=-1,
    #                         volume_fraction=0.3,
    #                         penalty_factor=3.0,
    #                         mesh_type=mesh_type, nx=nx, ny=ny, nz=nz, hx=hy, hy=hy, hz=hz,
    #                         p=1,
    #                         assembly_method=AssemblyMethod.FAST,
    #                         solver_type='direct', solver_params={'solver_type': 'mumps'},
    #                     )
    # # result = run_solve_uh_exact_test(config_solve_uh_exact)
    # # result2 = run_assmeble_exact_test(config_assmeble_exact)
    # # result3 = run_assemble_time_test(config=config_assemble_time)
    # result = run_solve_uh_time_test(config=config_solver_time)