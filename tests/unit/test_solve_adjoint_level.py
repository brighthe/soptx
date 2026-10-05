"""``LagrangeFEMAnalyzer.solve_adjoint`` 装配层级前提的回归测试.

``solve_adjoint`` 按行列施加齐次 Dirichlet 条件, 只适用于显式矩阵 (``'fa'``); 矩阵自由
层级下曾在 ``_apply_matrix`` 中报含义不明的错误, 现改为明确的 NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import QuadrangleMesh
from soptx.problems import HalfMBBBeamRight2d

DOMAIN = (0.0, 6.0, 0.0, 2.0)


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _analyzer(operator_level: str, solve_method: str) -> LagrangeFEMAnalyzer:
    """6x2 四边形网格上的半 MBB 梁分析器."""
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, hypothesis="plane_stress", enable_logging=False
    )
    return LagrangeFEMAnalyzer(
        disp_mesh=QuadrangleMesh.from_box(list(DOMAIN), nx=6, ny=2),
        pde=HalfMBBBeamRight2d(domain=DOMAIN),
        material=material,
        space_degree=1,
        integration_order=3,
        operator_level=operator_level,
        solve_method=solve_method,
        topopt_algorithm=None,
        enable_logging=False,
    )


def test_matrix_free_level_is_rejected():
    """矩阵自由层级调用 solve_adjoint 明确报错."""
    analyzer = _analyzer("ea", "cg")
    rhs = bm.zeros(analyzer.tensor_space.number_of_global_dofs(), dtype=bm.float64)
    with pytest.raises(NotImplementedError, match="operator_level='fa'"):
        analyzer.solve_adjoint(rhs=rhs)


def test_full_assembly_level_solves():
    """'fa' 层级照常求解: 解满足 K lambda = rhs, 且 Dirichlet 自由度为零."""
    analyzer = _analyzer("fa", "scipy")
    gdof = analyzer.tensor_space.number_of_global_dofs()
    rhs = bm.tensor(np.random.default_rng(0).standard_normal(gdof))
    lam = bm.to_numpy(analyzer.solve_adjoint(rhs=rhs))

    _, is_bd = analyzer.tensor_space.boundary_interpolate(
        gd=analyzer.pde.dirichlet_bc, threshold=analyzer.pde.is_dirichlet_boundary(), method="interp"
    )
    is_bd = bm.to_numpy(is_bd)
    K = analyzer.assemble_stiff_matrix().to_scipy().toarray()
    free = ~is_bd
    np.testing.assert_array_equal(lam[is_bd], 0.0)
    np.testing.assert_allclose(K[np.ix_(free, free)] @ lam[free], bm.to_numpy(rhs)[free], rtol=0, atol=1e-10)
