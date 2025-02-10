"""测试 filter 模块."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.filter import (Filter, FilterConfig)

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']
    pde_type: Literal['cantilever_2d_1', 'cantilever_2d_2']

    filter_radius: float
    filter_type: Literal['sensitivity', 'density', 'heaviside']

def run_filter_test(config: TestConfig):
    """测试不同的滤波器"""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')

    filter_config = FilterConfig(
                        filter_type=config.filter_type,
                        filter_radius=config.filter_radius
                    )
    filter_instance = Filter(filter_config)
