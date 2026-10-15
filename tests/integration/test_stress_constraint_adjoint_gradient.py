# -*- coding: utf-8 -*-
"""应力约束 AL 目标的解析梯度与中心差分对照.

覆盖两个适配器的完整灵敏度链路: 显式偏导, 伴随载荷, 伴随求解与隐式项. 公式层的
偏导另由 ``tests/unit/test_stress_constraint_formulations.py`` 核对.

网格取 16 x 8, 载荷宽度 10, 使牵引贴片 [15, 25] 恰好覆盖整条边界边 (贴片不覆盖
任何整条边时载荷会静默为零).

Hu--Zhang 另有两组不做角点松弛的算例, 覆盖强施加边界在伴随方程中的固定集合: 悬臂梁
右端两角是两条牵引边的交点 (按 Hu & Ma 2021 式 (4.1) 联立, 三个顶点自由度全部固定);
半域固支梁的对称面只强加切向牵引, 且与底边载荷区交于直角 (秩 2, 固定两个). 伴随方程的
固定集合漏掉其中任何一类, 相邻单元的梯度都会与差分不符. 许用应力按实际应力取值, 使应力比为 O(1), 否则约束
值几乎与应力无关, 伴随链路得不到检验. 密度取在 [0.3, 0.9], 避开 Hu--Zhang 隐式项
在 m_E <= 1e-4 处的有意截断; 靠近 AL 激活集切换点的单元不做比较.
"""

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TriangleMesh
from soptx.problems import CantileverMiddlePatch2d, HalfFixedFixedBeamPatch2d
from soptx.topology.constraints import (
    EpsilonRelaxedStressFormulation,
    HuZhangStressConstraint,
    LagrangeStressConstraint,
)
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import AugmentedLagrangianObjective, VolumeObjective
from soptx.topology.optimizers import ALMMMAOptions

STEP = 1.0e-4        # 中心差分步长; 更小的步长会被求解器舍入噪声淹没
RTOL = 1.0e-4
N_CELLS_CHECKED = 6


def _problem_and_mesh(case):
    """返回算例的问题与网格; 载荷区端点均落在网格节点上."""
    if case == "cantilever":
        problem = CantileverMiddlePatch2d(domain=(0.0, 80.0, 0.0, 40.0), P=-400.0, load_width=10.0,
                                     E=1.0, nu=0.25, plane_type="plane_stress")
        return problem, TriangleMesh.from_checkerboard_box(box=problem.domain, nx=16, ny=8)
    # 载荷区 [75, 80] 紧贴对称面 x = 80, 与之交于直角
    problem = HalfFixedFixedBeamPatch2d(domain=(0.0, 80.0, 0.0, 20.0), P=-3.0, load_width=10.0,
                                        E=1.0, nu=0.25, plane_type="plane_stress")
    return problem, TriangleMesh.from_checkerboard_box(box=problem.domain, nx=16, ny=4)


def _build(method, stress_limit, case="cantilever", use_relaxation=True):
    """组装分析链与 AL 目标, 返回 (analyzer, constraint, density, al)."""
    problem, mesh = _problem_and_mesh(case)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.25,
                                              hypothesis="plane_stress")
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp",
        options={"penalty_factor": 3.5, "void_youngs_modulus": 1e-9, "target_variables": ["E"]})
    common = dict(disp_mesh=mesh, pde=problem, material=material, space_degree=2,
                  integration_order=6, solve_method="mumps", topopt_algorithm="density_based",
                  interpolation_scheme=interpolation)
    formulation = EpsilonRelaxedStressFormulation(epsilon=1e-3)
    if method == "lfem":
        analyzer = LagrangeFEMAnalyzer(assembly_method="standard", **common)
        constraint = LagrangeStressConstraint(analyzer=analyzer, stress_limit=stress_limit,
                                              formulation=formulation)
    else:
        analyzer = HuZhangMFEMAnalyzer(use_relaxation=use_relaxation, **common)
        constraint = HuZhangStressConstraint(analyzer=analyzer, stress_limit=stress_limit,
                                             formulation=formulation)
    _, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh, displacement_mesh=mesh, relative_density=0.5)
    al = AugmentedLagrangianObjective(volume_objective=VolumeObjective(analyzer=analyzer),
                                      stress_constraint=constraint,
                                      options=ALMMMAOptions(mu_0=10.0), enable_logging=False)
    return analyzer, constraint, density, al


@pytest.mark.parametrize("method, case, use_relaxation", [
    ("lfem", "cantilever", True),
    ("huzhang", "cantilever", True),
    ("huzhang", "cantilever", False),
    ("huzhang", "half_fixed_fixed", False),
])
def test_al_gradient_matches_central_difference(method, case, use_relaxation):
    bm.set_backend("numpy")
    rng = np.random.default_rng(0)
    build = lambda stress_limit: _build(method, stress_limit, case, use_relaxation)

    # 先按单位许用应力求一次应力, 再取中位数的两倍作许用应力, 使应力比为 O(1)
    analyzer, constraint, density, _ = build(stress_limit=1.0)
    nc = density.shape[0]
    base = rng.uniform(0.3, 0.9, nc)
    density[:] = base
    state = analyzer.solve_state(rho_val=density)
    constraint.fun(density, state)  # 填充 state["von_mises"]
    von_mises = np.asarray(state["von_mises"]).reshape(-1)
    # 载荷为零时 von Mises 全部停在 calculate_von_mises_stress 的下限 1e-6
    assert von_mises.max() > 1e-3, "应力全为零: 牵引贴片没有覆盖任何整条边界边"
    stress_limit = 2.0 * float(np.median(von_mises))

    analyzer, constraint, density, al = build(stress_limit=stress_limit)
    al.lamb = bm.asarray(rng.uniform(2.0, 5.0, (nc, 1)))
    al.mu = 10.0

    def objective(values):
        density[:] = values
        state = analyzer.solve_state(rho_val=density)
        return float(al.fun(density, state)), state

    _, state = objective(base)
    gradient = np.asarray(al.jac(density, state)).reshape(-1)
    g = np.asarray(al._cache_g).reshape(-1)
    lamb = np.asarray(al.lamb).reshape(-1)
    ratio = np.asarray(state["von_mises"]).reshape(-1) / stress_limit
    assert 0.1 < np.median(ratio) < 2.0, "应力比不在 O(1), 伴随链路检验无效"

    margin = np.abs(g + lamb / al.mu)
    candidates = np.flatnonzero(margin > 0.05)
    # 一半取右端一列单元 (悬臂梁的牵引角点 / 半域梁的对称面所在), 强施加边界的伴随固定集合
    # 出错时首先体现在这里; 另一半随机
    center_x = np.asarray(bm.to_numpy(analyzer.disp_mesh.entity_barycenter("cell")))[:, 0]
    near_right = np.intersect1d(candidates, np.flatnonzero(center_x > 75.0))
    cells = rng.choice(near_right, N_CELLS_CHECKED // 2, replace=False)
    rest = np.setdiff1d(candidates, cells)
    cells = np.concatenate([cells, rng.choice(rest, N_CELLS_CHECKED - cells.size, replace=False)])
    for cell in cells:
        plus, minus = base.copy(), base.copy()
        plus[cell] += STEP
        minus[cell] -= STEP
        finite = (objective(plus)[0] - objective(minus)[0]) / (2.0 * STEP)
        assert gradient[cell] == pytest.approx(finite, rel=RTOL), f"单元 {cell}"
