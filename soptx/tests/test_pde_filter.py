"""测试 filter 模块中的 pde_filter."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.filter import (SensitivityPDEBasedFilter, 
                          DensityPDEBasedFilter)

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
                ipoints_ordering='yx', flip_direction=None,
                device='cpu'
            )

    SPF = SensitivityPDEBasedFilter(mesh=mesh, rmin=config.filter_radius)
    SPF._build_filter_matrix()

if __name__ == "__main__":
    config = TestConfig(
        backend='numpy',
        mesh_type='uniform_mesh_2d',
        nx=60, ny=20, hx=1, hy=1,
        filter_radius=2.4,
    )
    run_filter_H_test(config)
