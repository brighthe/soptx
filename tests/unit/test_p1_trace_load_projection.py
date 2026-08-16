from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem import project_patch_traction_to_p1_trace
from soptx.problems import FixedFixedBeamCenterLoad2d


def _projected(n_cells: int, problem: FixedFixedBeamCenterLoad2d):
    return project_patch_traction_to_p1_trace(
        line=(problem.domain[0], problem.domain[1]),
        n_cells=n_cells,
        level=problem.traction_level,
        patch=problem.traction_patch,
        intensity=problem.traction_intensity,
    )


@pytest.mark.parametrize("n_cells", [16, 40, 160, 161])
def test_projection_preserves_the_resultant(n_cells: int) -> None:
    """常数在 P1 迹空间内, 因此 L2 投影必须精确保持合力, 与贴片是否对齐无关."""
    bm.set_backend("numpy")
    problem = FixedFixedBeamCenterLoad2d()

    load = _projected(n_cells, problem)

    assert load.resultant() == pytest.approx(problem.P, rel=1.0e-12)


def test_projection_preserves_the_first_moment() -> None:
    """一次多项式同样落在 P1 迹空间内, 因此载荷的一阶矩也被精确保持."""
    bm.set_backend("numpy")
    problem = FixedFixedBeamCenterLoad2d()
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
    baseline = FixedFixedBeamCenterLoad2d()
    problem = FixedFixedBeamCenterLoad2d(traction=_projected(160, baseline))

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

    values = bm.to_numpy(problem.traction_bc(points))
    # 顶边与内部保持零牵引, 底边贴片中心的值被投影抬高, 不再等于 P / load_width
    np.testing.assert_allclose(values[:3], np.zeros((3, 2)), atol=1.0e-14)
    assert values[3, 0] == 0.0
    assert values[3, 1] < problem.traction_intensity
    np.testing.assert_allclose(values, bm.to_numpy(problem.neumann_bc(points)))


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
