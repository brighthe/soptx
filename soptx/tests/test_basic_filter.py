"""测试 filter 模块."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.filter import (SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)
from soptx.filter import (FilterMatrix, Filter, FilterConfig)

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

    H1, Hs1 = FilterMatrix._compute_filter_2d(nx=config.nx, ny=config.ny, 
                                        rmin=config.filter_radius)
    SF = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    H2, Hs2 = SF.H, SF.Hs

    diffH = bm.sum(bm.abs(H1.toarray() - H2.toarray()))
    diffHs = bm.sum(bm.abs(Hs1 - Hs2))
    print(f"Diff H: {diffH}, Diff Hs: {diffHs}")

if __name__ == "__main__":
    config = TestConfig(
        backend='numpy',
        mesh_type='uniform_mesh_2d',
        nx=160, ny=100, hx=1, hy=1,
        filter_radius=6.0,
    )
    run_filter_H_test(config)
