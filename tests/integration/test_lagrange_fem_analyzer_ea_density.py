"""``operator_level='ea'`` 在密度拓扑优化下的两种单元装配形式.

单元密度 (NC, ) 下 K_e = s_e K_e^0, 分析器只常驻 K_e^0 (与敏度共用的那一份) 与 s_e,
即 N_k = NC 的 ``SharedReferenceElementAssembly``; 泊松比插值等系数不是单元标量时,
退回逐单元积分的标准 EA (``ElementAssembly``). 这里验证:

1. 单元密度选逐单元参考形式, K_e^0 与分析器缓存是同一个对象, 换密度时原地只换 s_e;
2. 泊松比插值与非拓扑优化走标准 EA;
3. 两种形式的 ``@`` 都与 'fa' 装配的全局矩阵一致到舍入;
4. 系数形式在两次装配间改变时重建层级, 而不是错用旧形式原地更新.
"""

from __future__ import annotations

import numpy as np
import pytest
from soptx.backend import backend_manager as bm

from soptx.fem import LagrangeFEMAnalyzer, create_huzhang_checkerboard_mesh
from soptx.fem.levels import ElementAssembly, SharedReferenceElementAssembly
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import BearingDevice2d
from soptx.topology.interpolation import MaterialInterpolationScheme

NX, NY = 12, 4
RTOL = 1.0e-12


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def make_analyzer(operator_level: str,
                  nu: float = 0.3,
                  target_variables=("E", ),
                  topopt: bool = True,
                  p: int = 2) -> LagrangeFEMAnalyzer:
    problem = BearingDevice2d(t=-8.0e-2, E=1.0, nu=nu, plane_type="plane_strain")
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    kwargs = dict(
        disp_mesh=create_huzhang_checkerboard_mesh(box=problem.domain, nx=NX, ny=NY),
        pde=problem,
        material=material,
        space_degree=p,
        integration_order=2 * p + 2,
        assembly_method="standard",
        operator_level=operator_level,
        solve_method="scipy" if operator_level == "fa" else "cg",
    )
    if topopt:
        kwargs.update(
            topopt_algorithm="density_based",
            interpolation_scheme=MaterialInterpolationScheme(
                density_location="element",
                interpolation_method="msimp",
                options={
                    "penalty_factor": 3.0,
                    "void_youngs_modulus": 1.0e-9,
                    "target_variables": list(target_variables),
                    "nu_penalty_factor": 1.0,
                    "void_poisson_ratio": 0.3,
                },
                enable_logging=False,
            ),
        )
    else:
        kwargs.update(topopt_algorithm=None)

    return LagrangeFEMAnalyzer(**kwargs)


def random_density(analyzer: LagrangeFEMAnalyzer, seed: int):
    NC = analyzer.disp_mesh.number_of_cells()
    rng = np.random.default_rng(seed)

    return bm.tensor(0.3 + 0.6 * rng.random(NC), dtype=bm.float64)


def random_vector(analyzer: LagrangeFEMAnalyzer, seed: int = 2026):
    gdof = analyzer.tensor_space.number_of_global_dofs()

    return bm.asarray(np.random.default_rng(seed).standard_normal(gdof))


def relative_difference(a, b) -> float:
    a, b = bm.to_numpy(a), bm.to_numpy(b)

    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def assert_matches_fa(ea: LagrangeFEMAnalyzer, fa: LagrangeFEMAnalyzer, rho) -> None:
    operator = ea.assemble_stiff_matrix(rho_val=rho)
    matrix = fa.assemble_stiff_matrix(rho_val=rho)
    x = random_vector(fa)

    assert relative_difference(operator @ x, matrix.matmul(x)) < RTOL


@pytest.mark.parametrize("p", [1, 2])
def test_element_density_uses_per_element_reference(p: int) -> None:
    """单元密度: 常驻 K_e^0 与 s_e, K_e^0 即分析器缓存, 换密度只换 s_e."""
    ea, fa = make_analyzer("ea", p=p), make_analyzer("fa", p=p)
    rho = random_density(ea, seed=7)

    assert_matches_fa(ea, fa, rho)
    level = ea._level
    assert isinstance(level, SharedReferenceElementAssembly)
    assert level.num_classes == ea.disp_mesh.number_of_cells()
    assert level.reference_matrices is ea._solid_stiffness_matrix()

    assert_matches_fa(ea, fa, random_density(ea, seed=11))
    assert ea._level is level
    assert level.reference_matrices is ea._solid_stiffness_matrix()


def test_poisson_interpolation_uses_standard_ea() -> None:
    """泊松比插值下 K_e 不是 K_e^0 的标量倍, 走逐单元积分的标准 EA."""
    ea = make_analyzer("ea", nu=0.4999, target_variables=("E", "nu"))
    fa = make_analyzer("fa", nu=0.4999, target_variables=("E", "nu"))
    rho = random_density(ea, seed=7)

    assert_matches_fa(ea, fa, rho)
    assert ea.poisson_ratio_interpolated
    assert type(ea._level) is ElementAssembly

    level = ea._level
    assert_matches_fa(ea, fa, random_density(ea, seed=11))
    assert ea._level is level


def test_without_topopt_uses_standard_ea() -> None:
    """非拓扑优化没有密度, 走标准 EA."""
    ea = make_analyzer("ea", topopt=False)
    fa = make_analyzer("fa", topopt=False)

    assert_matches_fa(ea, fa, None)
    assert type(ea._level) is ElementAssembly


def test_level_is_rebuilt_when_the_coefficient_form_changes() -> None:
    """同一分析器上系数形式改变时重建层级, 不以旧形式原地更新."""
    ea = make_analyzer("ea")
    fa = make_analyzer("fa")
    rho = random_density(ea, seed=7)

    assert_matches_fa(ea, fa, rho)
    assert isinstance(ea._level, SharedReferenceElementAssembly)

    # 单元标量系数可原地复用; 逐积分点形状 (NC, NQ) 的系数不是单元标量, 应拒绝复用
    coef = ea._integrator.coef
    assert ea._level_reusable(coef)
    assert not ea._level_reusable(bm.stack([coef, coef], axis=1))
