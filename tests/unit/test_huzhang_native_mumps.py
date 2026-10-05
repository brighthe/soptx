"""Hu--Zhang 原生格式经 MUMPS 求解的回归测试.

k = 3, n_x = 4 是论文 5.1 节制造解的最粗一级: 零对角块鞍点系统在此规模下
MUMPS 默认 ICNTL(14) 的工作空间估计偏小 (INFOG(1) = -9), 须由
``DirectSolver`` 放大 ICNTL(14) 重试才能分解.
"""

from __future__ import annotations

import pytest

from soptx.backend import backend_manager as bm
from soptx.fem import HuZhangMFEMAnalyzer
from soptx.mesh import create_huzhang_checkerboard_mesh
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import MixedBoundarySinusoidalElasticity2D

pytest.importorskip("mumps", reason="PyMUMPS 未安装")

# 论文表 5.1 中 k = 3, n_x = 4 的位移 L2 误差 (旧基准 summary.json 的完整精度值)
DISP_L2_ERROR_K3_NX4 = 3.0644170382857568e-3


def test_native_k3_coarsest_level_solves_with_mumps() -> None:
    bm.set_backend("numpy")
    problem = MixedBoundarySinusoidalElasticity2D(lame_lambda=1.0, shear_modulus=0.5)
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    mesh = create_huzhang_checkerboard_mesh(box=problem.domain, nx=4, ny=4)
    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=3,
        integration_order=8,
        use_relaxation=True,
        solve_method="mumps",
        topopt_algorithm=None,
        stabilization="none",
    )
    state = analyzer.solve_state(rho_val=None)

    assert analyzer.relative_state_residual() < 1e-12
    error = mesh.error(state["displacement"], problem.disp_solution, q=8)
    error = float(bm.to_numpy(error).reshape(-1)[0])
    assert error == pytest.approx(DISP_L2_ERROR_K3_NX4, rel=1e-8)
