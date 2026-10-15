# -*- coding: utf-8 -*-
"""整边选取的局部载荷区与网格对齐检查.

未注入 ``traction`` 的局部载荷区按整边选取施加牵引, 端点不是网格节点时合力静默
偏离 P (悬臂梁 8 x 4 网格上直接为零). ``check_traction_patch_alignment`` 把它变成
构造时的错误.
"""

import pytest

from soptx.backend import backend_manager as bm
from soptx.fem import (
    HuZhangMFEMAnalyzer,
    LagrangeFEMAnalyzer,
    check_traction_patch_alignment,
    project_patch_traction_to_p1_trace,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TriangleMesh
from soptx.problems import (
    CantileverMiddlePatch2d,
    FixedFixedBeamPatch2d,
    HalfFixedFixedBeamPatch2d,
    PortalFramePatch2d,
)
from soptx.topology.interpolation import MaterialInterpolationScheme

DOMAIN = (0.0, 80.0, 0.0, 40.0)


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _mesh(nx, ny):
    return TriangleMesh.from_checkerboard_box(box=DOMAIN, nx=nx, ny=ny)


def test_misaligned_patch_is_rejected():
    """8 x 4 网格的节点 y = 10, 20, 30 都不是载荷区端点 17 与 23."""
    with pytest.raises(ValueError, match="不是网格节点"):
        check_traction_patch_alignment(_mesh(8, 4), CantileverMiddlePatch2d(domain=DOMAIN))


def test_aligned_patch_passes():
    check_traction_patch_alignment(_mesh(80, 40), CantileverMiddlePatch2d(domain=DOMAIN))


def test_injected_traction_skips_check():
    """注入 P1 迹投影后合力与网格无关, 不要求对齐."""
    problem = CantileverMiddlePatch2d(domain=DOMAIN)
    traction = project_patch_traction_to_p1_trace(
        line=(0.0, 40.0), n_cells=4, level=80.0, patch=problem.traction_patch,
        intensity=problem.traction_intensity, axis=1)
    problem = CantileverMiddlePatch2d(domain=DOMAIN, traction=traction)
    assert problem.uses_step_traction is False
    check_traction_patch_alignment(_mesh(8, 4), problem)


@pytest.mark.parametrize("problem_type", [
    CantileverMiddlePatch2d, FixedFixedBeamPatch2d, HalfFixedFixedBeamPatch2d, PortalFramePatch2d,
])
def test_patch_problems_declare_step_traction(problem_type):
    """带局部载荷区的问题都声明 uses_step_traction, 未注入 traction 时为 True."""
    assert problem_type().uses_step_traction is True


@pytest.mark.parametrize("analyzer_type, extra", [
    (LagrangeFEMAnalyzer, {"assembly_method": "standard"}),
    (HuZhangMFEMAnalyzer, {"use_relaxation": True}),
])
def test_analyzers_reject_misaligned_patch(analyzer_type, extra):
    """两个分析器都在构造时调用对齐检查."""
    problem = CantileverMiddlePatch2d(domain=DOMAIN, E=1.0)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.25,
                                              hypothesis="plane_stress")
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp",
        options={"penalty_factor": 3.0, "void_youngs_modulus": 1e-9, "target_variables": ["E"]})
    with pytest.raises(ValueError, match="不是网格节点"):
        analyzer_type(disp_mesh=_mesh(8, 4), pde=problem, material=material, space_degree=2,
                      integration_order=6, solve_method="mumps", topopt_algorithm="density_based",
                      interpolation_scheme=interpolation, **extra)
