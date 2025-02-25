"""测试 filter 模块中的 basic_filter."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, UniformMesh3d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.filter import (SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']

    mesh_type: Literal['uniform_mesh_2d', 'uniform_mesh_3d']
    nx: int
    ny: int
    nz: int
    hx: float
    hy: float
    hz: float

    filter_radius: float

def run_filter_H_test(config: TestConfig):
    """测试不同的滤波器"""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')

    extent = [0, config.nx, 0, config.ny]
    origin = [0.0, 0.0]
    mesh = UniformMesh2d(
                extent=extent, h=[config.hx, config.hy], origin=origin,
                ipoints_ordering='yx', flip_direction=None,
                device='cpu'
            )

    SF = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    H_2d, _ = SF._compute_filter_2d_1(nx=config.nx, ny=config.ny,
                                 hx=config.hx, hy=config.hy,
                                 rmin=config.filter_radius)
    H_2d_Kd_tree, _ = SF._compute_filter_2d(nx=config.nx, ny=config.ny,
                                            hx=config.hx, hy=config.hy,
                                            rmin=config.filter_radius)
    diff_H_2d = bm.max(bm.abs(H_2d - H_2d_Kd_tree))
    print(f"difference between H_2d and H_2d_Kd_tree: {diff_H_2d}")

def run_filter_new_2d_test(config: TestConfig):
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')

    extent = [0, config.nx, 0, config.ny]
    origin = [0.0, 0.0]
    mesh = UniformMesh2d(
                extent=extent, h=[config.hx, config.hy], origin=origin,
                ipoints_ordering='yx', device='cpu'
                )

    SF = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    H, HS = SF._compute_filter_matrix()
    H1 = H.toarray()

    cell_centers = mesh.entity_barycenter('cell')
    domain = [0, config.nx*config.hx, 0, config.ny*config.hy]
    cell_self, neighbors = bm.query_point(x=cell_centers, y=cell_centers, h=config.filter_radius, 
                                        box_size=domain, mask_self=True, periodic=[False, False, False])
    
    H_test, Hs_test = SF._compute_filter_general(cell_centers=cell_centers, 
                                                rmin=config.filter_radius, 
                                                domain=domain)
    H1_test = H_test.toarray()
    
    diff_H = bm.max(bm.abs(H1 - H1_test))
    print(f"difference between H and H_test: {diff_H}")

    mesh_tri = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=10, ny=10)
    SF_tri = SensitivityBasicFilter(mesh=mesh_tri, rmin=config.filter_radius, domain=domain)
    H_tri, HS_tri = SF_tri._compute_filter_matrix()
    H_tri_1 = H_tri.toarray()
    print("--------------------")

def run_filter_new_3d_test(config: TestConfig):
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')

    extent = [0, config.nx, 0, config.ny, 0, config.nz]
    origin = [0.0, 0.0, 0.0]
    mesh = UniformMesh3d(
                extent=extent, h=[config.hx, config.hy, config.hz], origin=origin,
                ipoints_ordering='zyx', device='cpu'
                )

    SF = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    H, HS = SF._compute_filter_2d(nx=config.nx, ny=config.ny,
                                        hx=config.hx, hy=config.hy,
                                        rmin=config.filter_radius)
    H1 = H.toarray()
    

    node = mesh.entity('node')
    node_bc = mesh.entity_barycenter('cell')
    domain = [0, config.nx*config.hx, 0, config.ny*config.hy, 0, config.nz*config.hz]
    node_self, neighbors = bm.query_point(x=node_bc, y=node_bc, h=config.filter_radius, 
                                        box_size=domain, mask_self=True, periodic=[False, False, False])
    
    H_test, Hs_test = SF.compute_filter_general(cell_centers=node_bc, 
                                                rmin=config.filter_radius, 
                                                domain=domain)
    H1_test = H_test.toarray()
    
    diff_H = bm.max(bm.abs(H1 - H1_test))
    print(f"difference between H and H_test: {diff_H}")
    print("--------------------")
    

if __name__ == "__main__":
    nx, ny = 150, 100
    hx, hy = 1, 1
    backend = 'numpy'
    mesh_type = 'uniform_mesh_2d'
    config_2d = TestConfig(
        backend=backend,
        mesh_type=mesh_type,
        nx=nx, ny=ny, nz=None, hx=1, hy=1, hz=None,
        filter_radius=nx*0.04,
    )
    nx, ny, nz = 40, 30, 20
    hx, hy, hz = 1, 1, 1
    config_3d = TestConfig(
        backend=backend,
        mesh_type=mesh_type,
        nx=nx, ny=ny, nz=nz, hx=1, hy=1, hz=1,
        filter_radius=nx*0.04,
    )
    run_filter_new_2d_test(config_2d)
    # run_filter_new_3d_test(config_3d)
