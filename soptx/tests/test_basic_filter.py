"""测试 filter 模块中的 basic_filter."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.filter import (SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']

    mesh_type: Literal['uniform_mesh_2d']
    nx: int
    ny: int
    hx: float
    hy: float

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
                ipoints_ordering='yx', flip_direction='y',
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

if __name__ == "__main__":
    nx, ny = 300, 100
    config = TestConfig(
        backend='numpy',
        mesh_type='uniform_mesh_2d',
        nx=nx, ny=ny, hx=1, hy=1,
        filter_radius=nx*0.04,
    )
    run_filter_H_test(config)
