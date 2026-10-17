from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm

from soptx.fem import interpolate_patch_traction_to_p1_trace, project_patch_traction_to_p1_trace
from soptx.problems import CantileverMiddlePatch2d, FixedFixedBeamPatch2d


def _projected(n_cells: int, problem: FixedFixedBeamPatch2d):
    return project_patch_traction_to_p1_trace(
        line=(problem.domain[0], problem.domain[1]),
        n_cells=n_cells,
        level=problem.traction_level,
        patch=problem.traction_patch,
        intensity=problem.traction_intensity,
        axis=problem.traction_axis,
    )


@pytest.mark.parametrize("n_cells", [16, 40, 160, 161])
def test_projection_preserves_the_resultant(n_cells: int) -> None:
    """常数在 P1 迹空间内, 因此 L2 投影必须精确保持合力, 与载荷区是否对齐无关."""
    bm.set_backend("numpy")
    problem = FixedFixedBeamPatch2d()

    load = _projected(n_cells, problem)

    assert load.resultant() == pytest.approx(problem.P, rel=1.0e-12)


def test_projection_preserves_the_first_moment() -> None:
    """一次多项式同样落在 P1 迹空间内, 因此载荷的一阶矩也被精确保持."""
    bm.set_backend("numpy")
    problem = FixedFixedBeamPatch2d()
    load = _projected(160, problem)

    nodes = load.start + load.h * np.arange(load.coefficients.shape[0])
    coefficients = bm.to_numpy(load.coefficients)
    # 对 P1 x P1 的乘积用单元两点梯形公式不精确, 这里直接用 x*phi 的解析单元积分
    moment = 0.0
    for cell in range(load.coefficients.shape[0] - 1):
        x0, x1 = nodes[cell], nodes[cell + 1]
        c0, c1 = coefficients[cell], coefficients[cell + 1]
        moment += load.h * ((2.0 * x0 + x1) * c0 + (x0 + 2.0 * x1) * c1) / 6.0

    x_mid = (problem.domain[0] + problem.domain[1]) / 2.0
    assert moment == pytest.approx(x_mid * problem.P, rel=1.0e-10)


def test_injected_traction_replaces_only_the_load() -> None:
    """注入后几何、材料与边界标记全部不变, 只有牵引函数被替换."""
    bm.set_backend("numpy")
    baseline = FixedFixedBeamPatch2d()
    problem = FixedFixedBeamPatch2d(traction=_projected(160, baseline))

    assert problem.domain == baseline.domain
    assert problem.P == baseline.P
    points = bm.array([[80.0, 10.0], [0.0, 10.0], [80.0, 20.0], [80.0, 0.0]])
    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_displacement_boundary(points)),
        bm.to_numpy(baseline.is_displacement_boundary(points)),
    )
    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_traction_boundary(points)),
        bm.to_numpy(baseline.is_traction_boundary(points)),
    )

    load = problem.loads()[0]
    values = bm.to_numpy(load.traction(points))
    # 顶边与内部保持零牵引, 底边载荷区中心的值被投影抬高, 不再等于 P / load_width
    np.testing.assert_allclose(values[:3], np.zeros((3, 2)), atol=1.0e-14)
    assert values[3, 0] == 0.0
    assert values[3, 1] < problem.traction_intensity
    assert not hasattr(problem, "traction_bc")
    assert not hasattr(problem, "neumann_bc")


def test_projection_rejects_inconsistent_geometry() -> None:
    bm.set_backend("numpy")
    with pytest.raises(ValueError):
        project_patch_traction_to_p1_trace(
            line=(0.0, 160.0), n_cells=0, level=0.0, patch=(79.5, 80.5), intensity=-3.0
        )
    with pytest.raises(ValueError):
        project_patch_traction_to_p1_trace(
            line=(0.0, 160.0), n_cells=16, level=0.0, patch=(159.0, 161.0), intensity=-3.0
        )


# ---------------------------------------------------------------- 节点插值


def _interpolated(n_cells: int, problem: CantileverMiddlePatch2d):
    _, _, ymin, ymax = problem.domain
    return interpolate_patch_traction_to_p1_trace(
        line=(ymin, ymax),
        n_cells=n_cells,
        level=problem.traction_level,
        patch=problem.traction_patch,
        intensity=problem.traction_intensity,
        axis=problem.traction_axis,
    )


def _first_moment(load) -> float:
    """P1 载荷沿直线的一阶矩, 按 x * phi 的解析单元积分."""
    nodes = load.start + load.h * np.arange(load.coefficients.shape[0])
    coefficients = bm.to_numpy(load.coefficients)
    moment = 0.0
    for cell in range(coefficients.shape[0] - 1):
        x0, x1 = nodes[cell], nodes[cell + 1]
        c0, c1 = coefficients[cell], coefficients[cell + 1]
        moment += load.h * ((2.0 * x0 + x1) * c0 + (x0 + 2.0 * x1) * c1) / 6.0
    return moment


@pytest.mark.parametrize("n_cells", [40, 80, 160])
def test_interpolation_nodal_values_and_compact_support(n_cells: int) -> None:
    """内部节点取强度, 端点取一半, 其余为零: 非零牵引只延伸到端点外一个单元."""
    bm.set_backend("numpy")
    problem = CantileverMiddlePatch2d()
    load = _interpolated(n_cells, problem)

    nodes = load.start + load.h * np.arange(n_cells + 1)
    left, right = problem.traction_patch
    t = problem.traction_intensity
    expected = np.where((nodes > left + 1e-12) & (nodes < right - 1e-12), t, 0.0)
    expected[np.isclose(nodes, left) | np.isclose(nodes, right)] = 0.5 * t
    np.testing.assert_array_equal(bm.to_numpy(load.coefficients), expected)

    # 端点外第二个节点起严格为零, 且不变号
    values = bm.to_numpy(load.coefficients)
    assert np.all(values[nodes <= left - load.h + 1e-12] == 0.0)
    assert np.all(values[nodes >= right + load.h - 1e-12] == 0.0)
    assert np.all(np.sign(values[values != 0.0]) == np.sign(t))


@pytest.mark.parametrize("n_cells", [40, 80, 160])
def test_interpolation_preserves_resultant_and_first_moment(n_cells: int) -> None:
    """端点为节点时, 两端的插值误差反对称, 合力与一阶矩都精确."""
    bm.set_backend("numpy")
    problem = CantileverMiddlePatch2d()
    load = _interpolated(n_cells, problem)

    left, right = problem.traction_patch
    assert load.resultant() == pytest.approx(problem.P, rel=1.0e-12)
    assert _first_moment(load) == pytest.approx(0.5 * (left + right) * problem.P, rel=1.0e-12)


def test_interpolation_keeps_one_sided_value_at_line_end() -> None:
    """载荷区紧贴直线端点 (如对称面) 时该端取单侧值, 合力仍精确."""
    bm.set_backend("numpy")
    load = interpolate_patch_traction_to_p1_trace(
        line=(0.0, 80.0), n_cells=160, level=0.0, patch=(79.5, 80.0), intensity=-3.0)

    coefficients = bm.to_numpy(load.coefficients)
    assert coefficients[-1] == -3.0
    assert coefficients[-2] == -1.5
    assert load.resultant() == pytest.approx(-1.5, rel=1.0e-12)


def test_interpolation_rejects_unaligned_endpoints() -> None:
    """端点落在单元内部时插值不保持合力, 必须报错并指向 L2 投影."""
    bm.set_backend("numpy")
    with pytest.raises(ValueError, match="project_patch_traction_to_p1_trace"):
        interpolate_patch_traction_to_p1_trace(
            line=(0.0, 80.0), n_cells=80, level=0.0, patch=(79.5, 80.0), intensity=-3.0)
    with pytest.raises(ValueError):
        interpolate_patch_traction_to_p1_trace(
            line=(0.0, 40.0), n_cells=40, level=80.0, patch=(23.0, 17.0), intensity=-1.0, axis=1)
