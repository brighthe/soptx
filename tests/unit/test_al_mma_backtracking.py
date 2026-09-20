"""AL-MMA 渐近线下限与稳定候选点公式的针对性测试, 不执行有限元求解."""
import numpy as np
import pytest
from fealpy.backend import backend_manager as bm
from soptx.topology.optimizers.al_mma import (
    ALMMMAOptimizer,
    ALMMMAOptions,
    _bound_mma_asymptotes,
    _stable_mma_candidate,
)


def setup(**options):
    bm.set_backend("numpy")
    opt = object.__new__(ALMMMAOptimizer)
    opt.options = ALMMMAOptions(**options)
    return opt


@pytest.mark.parametrize("value", [0.0, -0.01, 10.0, 10.1, np.nan, np.inf])
def test_options_reject_invalid_asymptote_min_distance(value):
    with pytest.raises(ValueError, match="asymptote_min_distance"):
        ALMMMAOptions(asymptote_min_distance=value)


def test_asymptote_min_distance_changes_only_asymptote_floor():
    current = np.array([0.25, 0.75])
    lower_raw = current - 1.0e-8
    upper_raw = current + 1.0e-8

    lower_default, upper_default = _bound_mma_asymptotes(
        current, lower_raw, upper_raw, span=1.0, min_distance=0.01
    )
    lower_control, upper_control = _bound_mma_asymptotes(
        current, lower_raw, upper_raw, span=1.0, min_distance=0.001
    )

    np.testing.assert_allclose(current - lower_default, 0.01)
    np.testing.assert_allclose(upper_default - current, 0.01)
    np.testing.assert_allclose(current - lower_control, 0.001)
    np.testing.assert_allclose(upper_control - current, 0.001)


def test_subproblem_asymptote_floor_preserves_box_and_move_bounds():
    current = np.array([0.25, 0.75])
    gradient = np.array([1.0, -1.0])
    move = 0.005
    results = {}

    for floor in (0.01, 0.001):
        opt = setup(
            move_limit=move,
            asymp_init=1.0e-4,
            asymptote_min_distance=floor,
        )
        opt._epoch = 1
        opt._low = None
        opt._upp = None
        opt._asym_inc_dynamic = opt.options.asymp_incr
        opt._asym_decr_dynamic = opt.options.asymp_decr
        candidate = opt._solve_unconstrained_subproblem(
            gradient, current, current, current
        )
        results[floor] = candidate

        np.testing.assert_allclose(current - opt._low, floor)
        np.testing.assert_allclose(opt._upp - current, floor)
        assert np.all(candidate >= np.maximum(0.0, current - move))
        assert np.all(candidate <= np.minimum(1.0, current + move))

    assert not np.allclose(results[0.01], results[0.001])


def _run_two_cycle(floor, steps, move=0.005, amplitude=0.002, start=0.005):
    """让一个设计变量按固定幅度两值翻转, 返回每步的渐近线距离与候选步幅.

    模拟 AL 罚项折点上的单元: 设计变量在 c +/- amplitude 间交替, 梯度符号随之
    翻转. PolyStress 的振荡规则应使渐近线距离按 AsymDecr 几何收缩, 直至下限 floor.
    """
    opt = setup(move_limit=move, asymptote_min_distance=floor)
    opt._epoch = 3
    opt._asym_inc_dynamic = opt.options.asymp_incr
    opt._asym_decr_dynamic = opt.options.asymp_decr
    center = 0.5
    history = [center + amplitude * (-1.0) ** k for k in range(3)]
    opt._low = np.array([history[-1] - start])
    opt._upp = np.array([history[-1] + start])
    distances, step_sizes = [], []
    for k in range(3, 3 + steps):
        z = np.array([center + amplitude * (-1.0) ** k])
        zold1 = np.array([history[-1]])
        zold2 = np.array([history[-2]])
        gradient = np.sign(z - center)
        candidate = opt._solve_unconstrained_subproblem(gradient, z, zold1, zold2)
        distances.append(float(opt._upp[0] - z[0]))
        step_sizes.append(float(abs(candidate[0] - z[0])))
        history.append(float(z[0]))
    return np.array(distances), np.array(step_sizes)


def test_default_floor_lets_two_cycle_asymptotes_contract_below_move_limit():
    floor = ALMMMAOptions().asymptote_min_distance
    move = 0.005
    distances, step_sizes = _run_two_cycle(floor, steps=30, move=move)

    decr = ALMMMAOptions().asymp_decr
    expected = np.maximum(0.005 * decr ** np.arange(1, 31), floor)
    np.testing.assert_allclose(distances, expected, rtol=1.0e-12)
    assert floor < move
    assert distances[-1] == pytest.approx(floor)
    assert step_sizes[-1] <= 0.9 * floor + 1.0e-15
    assert step_sizes[-1] < step_sizes[0]


def test_legacy_floor_pins_two_cycle_asymptotes_above_move_limit():
    move = 0.005
    distances, _ = _run_two_cycle(0.01, steps=30, move=move)

    assert np.all(distances == pytest.approx(0.01))
    assert distances[-1] > move


def test_stable_candidate_extends_equal_coefficients_to_midpoint():
    lower = np.array([-0.2, 0.1])
    upper = np.array([0.8, 0.9])
    equal = np.array([2.0, 5.0])

    candidate = _stable_mma_candidate(lower, upper, equal, equal)

    np.testing.assert_allclose(candidate, 0.5 * (lower + upper))


def test_stable_candidate_matches_original_formula_for_regular_values():
    lower = np.array([-0.2, 0.1])
    upper = np.array([0.8, 0.9])
    p = np.array([2.0, 5.0])
    q = np.array([0.5, 3.0])
    expected = (
        lower * p - upper * q + (upper - lower) * np.sqrt(p * q)
    ) / (p - q)

    candidate = _stable_mma_candidate(lower, upper, p, q)

    np.testing.assert_allclose(candidate, expected, rtol=1.0e-14, atol=1.0e-14)


def test_stable_candidate_remains_accurate_for_nearly_equal_coefficients():
    lower = np.array([-0.2])
    upper = np.array([0.8])
    p = np.array([1.0])
    q = np.array([1.0 + 1.0e-14])
    p_ref = np.longdouble(p[0])
    q_ref = np.longdouble(q[0])
    expected = (
        np.longdouble(lower[0]) * np.sqrt(p_ref)
        + np.longdouble(upper[0]) * np.sqrt(q_ref)
    ) / (np.sqrt(p_ref) + np.sqrt(q_ref))

    candidate = _stable_mma_candidate(lower, upper, p, q)

    assert np.isfinite(candidate[0])
    assert candidate[0] == pytest.approx(float(expected), rel=1.0e-14, abs=1.0e-14)


def test_stable_candidate_handles_extreme_scales_without_overflow():
    lower = np.zeros(2)
    upper = np.ones(2)
    p = np.array([1.0e-300, 1.0e300])
    q = np.array([1.0e300, 1.0e-300])

    candidate = _stable_mma_candidate(lower, upper, p, q)

    assert np.all(np.isfinite(candidate))
    assert np.all((lower <= candidate) & (candidate <= upper))
    np.testing.assert_allclose(candidate, np.array([1.0, 0.0]), atol=1.0e-14)


def test_zero_gradient_with_symmetric_asymptotes_keeps_current_point():
    current = np.array([0.2, 0.7])
    distance = np.array([0.01, 0.3])
    lower = current - distance
    upper = current + distance
    regularization = 1.0e-6 / (upper - lower)
    p = (upper - current) ** 2 * regularization
    q = (current - lower) ** 2 * regularization

    candidate = _stable_mma_candidate(lower, upper, p, q)

    np.testing.assert_allclose(candidate, current, rtol=0.0, atol=1.0e-15)
