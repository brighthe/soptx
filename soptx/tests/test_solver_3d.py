"""测试 solver 模块求解位移的正确性."""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh3d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            ElasticMaterialConfig,
                            ElasticMaterialInstance,
                        )
from soptx.pde import Cantilever3dData1
from soptx.solver import ElasticFEMSolver, AssemblyMethod

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']
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
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any]


def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')
    
    if config.pde_type == 'cantilever_3d_1':
        pde = Cantilever3dData1(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    zmin=0, zmax=config.domain_height,
                    T = config.load
                )
        if config.mesh_type == 'uniform_mesh_3d':
            extent = [0, config.nx, 0, config.ny, 0, config.nz]
            origin = [0.0, 0.0, 0.0]
            mesh = UniformMesh3d(
                        extent=extent, h=[config.hx, config.hy, config.hz], origin=origin,
                        ipoints_ordering='zyx', flip_direction=None,
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
                            plane_assumption="3D",    
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = ElasticMaterialInstance(config=material_config)

    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)
    
    return materials, tensor_space_C, pde, rho

def run_assmeble_exact_test(config: TestConfig):
    """测试 SOPTX 中不同的 assembly_method 的正确性."""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver_s = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.STANDARD,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    solver_f3u = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.FAST_3D_UNIFORM,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    solver_vu = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.VOIGT_UNIFORM,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    
    solver_v = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.VOIGT,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    solver_s.update_status(rho[:])
    K_s = solver_s._assemble_global_stiffness_matrix()
    K_s_full = K_s.toarray()

    solver_f3u.update_status(rho[:])
    K_f3u = solver_f3u._assemble_global_stiffness_matrix()
    K_f3u_full = K_f3u.toarray()

    solver_vu.update_status(rho[:])
    K_vu = solver_vu._assemble_global_stiffness_matrix()
    K_vu_full = K_vu.toarray()

    solver_v.update_status(rho[:])
    K_v = solver_v._assemble_global_stiffness_matrix()
    K_v_full = K_v.toarray()

    print(f"diff_K1: {bm.sum(bm.abs(K_s_full - K_f3u_full))}")
    print(f"diff_K1: {bm.sum(bm.abs(K_f3u_full - K_vu_full))}")
    print(f"diff_K2: {bm.sum(bm.abs(K_vu_full - K_v_full))}")
    print(f"-------------------------------")

def run_solve_uh_exact_test(config: TestConfig):
    """测试位移求解是否正确."""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver_cg = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type='cg',
                    solver_params={'maxiter': 5000, 'atol': 1e-12, 'rtol': 1e-12}, 
                )
    solver_cg.update_status(rho[:])
    solver_result_cg = solver_cg.solve()
    uh_cg = solver_result_cg.displacement
    print(f"uh_cg: {bm.mean(uh_cg):.10f}")
    solver_mumps = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type='direct',
                    solver_params={'solver_type': 'mumps'}, 
                )
    solver_mumps.update_status(rho[:])
    K = solver_mumps._assemble_global_stiffness_matrix()
    K_full = K.toarray()
    print(f"K_full: {bm.sum(bm.abs(K_full)):.10f}")
    solver_result_mumps = solver_mumps.solve()
    uh_mumps = solver_result_mumps.displacement
    print(f"uh_mumps: {bm.mean(uh_mumps):.10f}")
    diff = bm.max(bm.abs(uh_cg - uh_mumps))
    print(f"Difference between CG and MUMPS : {diff:.6e}")


if __name__ == "__main__":
    '''
    参数来源论文: An efficient 3D topology optimization code written in Matlab
    '''
    pde_type = 'cantilever_3d_1'
    optimizer_type = 'oc'
    filter_type = 'sensitivity'
    nx, ny, nz = 60, 20, 4
    hx, hy, hz = 1, 1, 1
    config_solve_uh_exact = TestConfig(
            backend='numpy',
            pde_type=pde_type,
            elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
            domain_length=nx, domain_width=ny, domain_height=nz,
            load=-1,
            volume_fraction=0.3,
            penalty_factor=3.0,
            mesh_type='uniform_mesh_3d', nx=nx, ny=ny, nz=nz, hx=hy, hy=hy, hz=hz,
            assembly_method=AssemblyMethod.FAST_3D_UNIFORM,
            solver_type='direct', solver_params={'solver_type': 'mumps'},
        )
    config_assmeble_exact = TestConfig(
                        backend='numpy',
                        pde_type='cantilever_3d_1',
                        elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                        domain_length=nx, domain_width=ny, domain_height=nz,
                        load=-1,
                        volume_fraction=0.3,
                        penalty_factor=3.0,
                        mesh_type='uniform_mesh_3d', nx=nx, ny=ny, nz=nz, hx=hy, hy=hy, hz=hz,
                        assembly_method=None,
                        solver_type='direct', 
                        solver_params={'solver_type': 'mumps'},
                        )
    result = run_solve_uh_exact_test(config_solve_uh_exact)
    # result2 = run_assmeble_exact_test(config_assmeble_exact)