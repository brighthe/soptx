"""局部应力约束豁免机制的纯代数回归测试.

覆盖三件事: 掩码的几何判定、掩码沿评价点轴的广播, 以及两条离散路径上
"豁免点既不进约束集合、也不进灵敏度"的等价性. 不启动有限元求解.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.topology.constraints import (
    HuZhangStressConstraint,
    LagrangeStressConstraint,
    EpsilonRelaxedStressFormulation,
    build_exemption_mask,
)
from soptx.topology.constraints.exemption import (
    EXEMPT_CONSTRAINT_VALUE,
    apply_exemption,
    validate_exemption_mask,
)


class _FakeMesh:
    """只提供单元重心的最小网格替身."""

    def __init__(self, barycenter):
        self._barycenter = bm.tensor(barycenter, dtype=bm.float64)

    def entity_barycenter(self, entity):
        assert entity == "cell"
        return self._barycenter

    def number_of_cells(self):
        return self._barycenter.shape[0]


def _mesh():
    # 四个单元重心: 端点正上方、端点外侧、贴片中部、远场
    return _FakeMesh([[80.0, 17.4], [79.5, 16.6], [79.5, 20.0], [10.0, 20.0]])


def test_exemption_mask_selects_cells_within_fixed_physical_radius():
    mask = build_exemption_mask(_mesh(), ((80.0, 17.0), (80.0, 23.0)), 1.5)
    assert np.asarray(bm.to_numpy(mask)).tolist() == [True, True, False, False]


def test_exemption_mask_shrinks_with_radius_and_vanishes_at_zero():
    mesh = _mesh()
    centers = ((80.0, 17.0), (80.0, 23.0))
    tight = np.asarray(bm.to_numpy(build_exemption_mask(mesh, centers, 0.5)))
    empty = np.asarray(bm.to_numpy(build_exemption_mask(mesh, centers, 0.0)))
    assert tight.tolist() == [True, False, False, False]
    assert not empty.any()
    assert validate_exemption_mask(build_exemption_mask(mesh, centers, 0.0), 4) is None


def test_exemption_mask_rejects_negative_radius_and_mismatched_shapes():
    with pytest.raises(ValueError, match="有限非负数"):
        build_exemption_mask(_mesh(), ((80.0, 17.0),), -1.0)
    with pytest.raises(ValueError, match="几何维"):
        build_exemption_mask(_mesh(), ((80.0, 17.0, 0.0),), 1.0)
    with pytest.raises(ValueError, match="形状"):
        validate_exemption_mask(bm.ones((3,), dtype=bm.bool), 4)


@pytest.mark.parametrize("shape", [(4, 3), (4, 2, 3)])
def test_exemption_broadcasts_over_all_evaluation_points_of_a_cell(shape):
    """掩码定义在单元上, 单元内的全部评价点必须一并生效."""
    values = bm.ones(shape, dtype=bm.float64)
    mask = bm.tensor([True, False, True, False], dtype=bm.bool)
    masked = np.asarray(bm.to_numpy(apply_exemption(values, mask, -1.0)))
    assert masked.shape == shape
    assert (masked[0] == -1.0).all() and (masked[2] == -1.0).all()
    assert (masked[1] == 1.0).all() and (masked[3] == 1.0).all()


# 逐单元应力水平: 待豁免的 0 号与 2 号单元最高, 使"豁免前它是全域最大违反者"
# 与"豁免后不再是"这件事可被断言, 而不是两边碰巧相等.
_VON_MISES_BY_CELL = (3.0, 2.0, 2.5, 1.5)


def _state(n_cells: int, n_points: int, key: str) -> dict:
    """构造一组处处违反约束的应力状态, 使豁免与否的差别无处隐藏."""
    stiffness_ratio = bm.full((n_cells,), 0.5, dtype=bm.float64)
    levels = bm.tensor(_VON_MISES_BY_CELL[:n_cells], dtype=bm.float64)
    von_mises = bm.broadcast_to(levels[:, None], (n_cells, n_points))
    stress = bm.ones((n_cells, n_points, 3), dtype=bm.float64)
    return {key: stress, "von_mises": von_mises, "stiffness_ratio": stiffness_ratio}


def _lagrange_constraint(mask, n_cells):
    analyzer = SimpleNamespace(
        interpolation_scheme=SimpleNamespace(n_sub=None),
        disp_mesh=SimpleNamespace(number_of_cells=lambda: n_cells),
        poisson_ratio_interpolated=False,
    )
    return LagrangeStressConstraint(
        analyzer=analyzer,
        stress_limit=1.0,
        formulation=EpsilonRelaxedStressFormulation(epsilon=1.0e-3),
        exemption_mask=mask,
    )


def _huzhang_constraint(mask, n_cells):
    analyzer = SimpleNamespace(
        disp_mesh=SimpleNamespace(number_of_cells=lambda: n_cells),
    )
    return HuZhangStressConstraint(
        analyzer=analyzer,
        stress_limit=1.0,
        formulation=EpsilonRelaxedStressFormulation(epsilon=1.0e-3),
        exemption_mask=mask,
    )


@pytest.mark.parametrize("path", ["lfem", "huzhang"])
def test_exempt_cells_contribute_neither_constraint_value_nor_sensitivity(path):
    n_cells, n_points = 4, 3
    mask = bm.tensor([True, False, True, False], dtype=bm.bool)
    exempt = np.asarray(bm.to_numpy(mask))
    if path == "lfem":
        constraint = _lagrange_constraint(mask, n_cells)
        state = _state(n_cells, n_points, "stress_solid")
    else:
        constraint = _huzhang_constraint(mask, n_cells)
        state = _state(n_cells, n_points, "stress_apparent")

    rho = bm.full((n_cells,), 0.5, dtype=bm.float64)
    g = np.asarray(bm.to_numpy(constraint.fun(rho, state)))
    violation = np.asarray(bm.to_numpy(constraint.compute_relative_violation(rho, state)))
    partial = np.asarray(bm.to_numpy(constraint.compute_partial_gradient_wrt_mE(state)))
    gradient = np.asarray(bm.to_numpy(constraint.compute_gradient_wrt_von_mises(state)))

    assert (g[exempt] == EXEMPT_CONSTRAINT_VALUE).all()
    assert (g[~exempt] > 0.0).all()
    assert (violation[exempt] == EXEMPT_CONSTRAINT_VALUE).all()
    assert np.isfinite(violation).all()
    assert (partial[exempt] == 0.0).all()
    assert (gradient[exempt] == 0.0).all()
    assert not (gradient[~exempt] == 0.0).all()


@pytest.mark.parametrize("path", ["lfem", "huzhang"])
def test_exemption_leaves_constrained_cells_and_array_shape_untouched(path):
    """豁免只改约束集合成员, 不改 AL 的归一化基数 (返回形状不变)."""
    n_cells, n_points = 4, 3
    mask = bm.tensor([True, False, True, False], dtype=bm.bool)
    exempt = np.asarray(bm.to_numpy(mask))
    key = "stress_solid" if path == "lfem" else "stress_apparent"
    build = _lagrange_constraint if path == "lfem" else _huzhang_constraint
    rho = bm.full((n_cells,), 0.5, dtype=bm.float64)

    plain = np.asarray(bm.to_numpy(build(None, n_cells).fun(rho, _state(n_cells, n_points, key))))
    masked = np.asarray(bm.to_numpy(build(mask, n_cells).fun(rho, _state(n_cells, n_points, key))))

    assert plain.shape == masked.shape == (n_cells, n_points)
    assert np.allclose(plain[~exempt], masked[~exempt])
    # 未豁免时该单元本是全域最大违反者, 豁免后必须不再是
    assert plain.max() > masked.max()


@pytest.mark.parametrize("path", ["lfem", "huzhang"])
def test_all_false_mask_is_equivalent_to_no_exemption(path):
    n_cells, n_points = 4, 3
    key = "stress_solid" if path == "lfem" else "stress_apparent"
    build = _lagrange_constraint if path == "lfem" else _huzhang_constraint
    rho = bm.full((n_cells,), 0.5, dtype=bm.float64)
    empty = bm.zeros((n_cells,), dtype=bm.bool)

    constraint = build(empty, n_cells)
    assert constraint.exemption_mask is None
    assert np.allclose(
        np.asarray(bm.to_numpy(constraint.fun(rho, _state(n_cells, n_points, key)))),
        np.asarray(bm.to_numpy(build(None, n_cells).fun(rho, _state(n_cells, n_points, key)))),
    )


@pytest.mark.parametrize("path", ["lfem", "huzhang"])
def test_unexempted_constraint_reports_what_the_mask_hides(path):
    """诊断口径必须能取回豁免单元的真实约束值, 否则只能读回哨兵 -1.0."""
    n_cells, n_points = 4, 3
    mask = bm.tensor([True, False, True, False], dtype=bm.bool)
    exempt = np.asarray(bm.to_numpy(mask))
    key = "stress_solid" if path == "lfem" else "stress_apparent"
    build = _lagrange_constraint if path == "lfem" else _huzhang_constraint
    rho = bm.full((n_cells,), 0.5, dtype=bm.float64)

    constraint = build(mask, n_cells)
    raw = np.asarray(bm.to_numpy(
        constraint.compute_unexempted_constraint(rho, _state(n_cells, n_points, key))))
    plain = np.asarray(bm.to_numpy(
        build(None, n_cells).fun(rho, _state(n_cells, n_points, key))))

    # 未豁免口径与"根本没设掩码"逐项一致, 包括被掩码盖住的那些单元
    assert raw.shape == plain.shape
    assert np.allclose(raw, plain)
    assert (raw[exempt] > 0.0).all()
    # fun 仍按豁免口径返回, 两者不能混用
    masked = np.asarray(bm.to_numpy(constraint.fun(rho, _state(n_cells, n_points, key))))
    assert (masked[exempt] == EXEMPT_CONSTRAINT_VALUE).all()
