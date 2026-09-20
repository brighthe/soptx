"""C3 持续性、罚因子条件放大与末期移动限制衰减的回归测试, 不执行有限元求解.

test_al_mma_state_consistency.py 把 hold_steps 钉死为 1 并注明不覆盖 C3;
本文件补上 C3, 以及与之交互的两处新增外层步状态 (_c0_first_iter、rho 窗口、
_prev_violation)。所有用例复用该文件的 fakes, 不碰 FEA。
"""
from types import SimpleNamespace

import numpy as np
import pytest
from fealpy.backend import backend_manager as bm
from soptx.topology.objectives.augmented_lagrangian import AugmentedLagrangianObjective
from soptx.topology.optimizers.al_mma import (
    ALMMMAOptions,
    ALMMMAOptimizer,
    coherent_travel_share,
    projected_gradient_residual,
)

from test_al_mma_state_consistency import FakeFilter, FakeObjective


def run_case(updates, violation=-0.1, outer=8, inner=1, initial=None,
             **option_overrides):
    """按外层步逐个投喂设计变量增量, inner=1 使每个外层步恰好消耗一个增量.

    initial 为 None 时 10 个单元全取 0.5; 传入数组可构造实体 / 灰度单元并存的密度场
    (FakeFilter 不滤波, 物理密度等于设计变量)。
    """
    bm.set_backend("numpy")
    objective = FakeObjective(violation)
    optimizer = object.__new__(ALMMMAOptimizer)
    options = dict(
        max_al_iterations=outer, mma_iters_per_al=inner,
        change_tolerance=0.002, stress_tolerance=0.003,
        hold_steps=3,
        use_penalty_continuation=False,
        initialize_problem_params=lambda m, n: None,
    )
    options.update(option_overrides)
    optimizer.options = SimpleNamespace(**options)
    optimizer._al_objective = objective
    optimizer._filter = FakeFilter(False)
    optimizer._log_info = lambda message: None
    optimizer._update_penalty = lambda iter_idx: None
    calls = []

    def update(**kwargs):
        delta = updates[min(len(calls), len(updates) - 1)]
        calls.append(1)
        return kwargs["z"].copy() + delta

    optimizer._solve_unconstrained_subproblem = update
    initial = np.full(10, 0.5) if initial is None else np.asarray(initial, dtype=float)
    rho, history = optimizer.optimize(initial, initial)

    return optimizer, rho, history, len(calls)


def run_projected_gradient_case(gradients, inner=3):
    """用可控梯度验证 fixed-AL 内层协议；不执行有限元求解。"""
    bm.set_backend("numpy")
    objective = FakeObjective(-0.1)
    jac_calls = []

    def jac(density, state):
        np.testing.assert_array_equal(density, state["rho"])
        value = gradients[min(len(jac_calls), len(gradients) - 1)]
        jac_calls.append(1)
        return np.full_like(density, value)

    objective.jac = jac
    optimizer = object.__new__(ALMMMAOptimizer)
    optimizer.options = SimpleNamespace(
        max_al_iterations=1,
        mma_iters_per_al=inner,
        change_tolerance=0.002,
        stress_tolerance=0.003,
        hold_steps=1,
        inner_stop_rule='projected_gradient',
        inner_relative_tolerance=0.1,
        inner_absolute_tolerance=1e-6,
        use_penalty_continuation=False,
        initialize_problem_params=lambda m, n: None,
    )
    optimizer._al_objective = objective
    optimizer._filter = FakeFilter(False)
    optimizer._log_info = lambda message: None
    optimizer._update_penalty = lambda iter_idx: None
    epochs = []

    def update(**kwargs):
        epochs.append(optimizer._epoch)
        return kwargs["z"].copy()

    optimizer._solve_unconstrained_subproblem = update
    initial = np.full(10, 0.5)
    rho, history = optimizer.optimize(initial, initial)
    return optimizer, objective, rho, history, epochs, len(jac_calls)


# =====================================================================
# fixed-AL projected-gradient 内层停止规则
# =====================================================================

def test_projected_gradient_residual_excludes_passive_variables():
    design = np.array([0.5, 0.5, 0.5])
    gradient = np.array([0.2, 0.4, 0.8])
    passive = np.array([False, True, False])
    assert projected_gradient_residual(design, gradient, passive) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "overrides",
    [
        {'inner_stop_rule': 'unknown'},
        {'inner_relative_tolerance': 0.0},
        {'inner_relative_tolerance': 1.0},
        {'inner_absolute_tolerance': 0.0},
        {'inner_stop_rule': 'projected_gradient', 'use_penalty_continuation': True},
        {'inner_stop_rule': 'projected_gradient', 'mma_iters_per_al': 0},
    ],
)
def test_projected_gradient_options_reject_invalid_controls(overrides):
    with pytest.raises(ValueError):
        ALMMMAOptions(**overrides)


def test_projected_gradient_accepts_relative_reduction_before_budget():
    optimizer, objective, _, history, epochs, jac_calls = (
        run_projected_gradient_case([0.5, 0.04], inner=3)
    )
    assert epochs == [1]
    assert len(history.changes) == 1
    assert jac_calls == 2
    assert len(objective.multiplier_states) == 1
    diagnostics = optimizer.last_inner_diagnostics
    assert diagnostics['outer_index'] == 1
    assert diagnostics['beta'] is None
    assert diagnostics['mu'] == pytest.approx(1.0)
    assert diagnostics['initial_residual'] == pytest.approx(0.5)
    assert diagnostics['final_residual'] == pytest.approx(0.04)
    assert diagnostics['target'] == pytest.approx(0.05)
    assert diagnostics['inner_steps'] == 1
    assert diagnostics['accepted'] is True


def test_projected_gradient_budget_failure_skips_outer_update():
    optimizer, objective, _, history, epochs, jac_calls = (
        run_projected_gradient_case([0.5], inner=2)
    )
    assert epochs == [1, 2]
    assert len(history.changes) == 2
    assert jac_calls == 3  # 两个候选起点 + 最后接受态验收
    assert objective.multiplier_states == []
    assert optimizer.converged is False
    assert optimizer.termination_reason.startswith('inner-iteration-limit:')
    assert optimizer.last_inner_diagnostics['accepted'] is False
    assert optimizer.last_inner_diagnostics['inner_steps'] == 2
    assert optimizer.last_inner_diagnostics['change_outer_mean'] == pytest.approx(0.0)
    assert optimizer.last_inner_diagnostics['change_outer_max'] == pytest.approx(0.0)
    assert np.isfinite(optimizer.last_change_outer_mean)
    assert np.isfinite(optimizer.last_change_outer_max)
    assert optimizer.last_multiplier_change == pytest.approx(0.0)


def test_projected_gradient_zero_residual_keeps_one_real_update():
    optimizer, objective, _, history, epochs, jac_calls = (
        run_projected_gradient_case([0.0], inner=1)
    )
    assert epochs == [1]
    assert len(history.changes) == 1
    assert jac_calls == 2
    assert len(objective.multiplier_states) == 1
    assert optimizer.last_inner_diagnostics['accepted'] is True
    assert optimizer.last_inner_diagnostics['final_residual'] == pytest.approx(0.0)


def test_projected_gradient_uses_inner_mma_epoch_after_two_updates():
    optimizer, objective, _, _, epochs, _ = run_projected_gradient_case([0.5], inner=3)
    assert epochs == [1, 2, 3]
    assert objective.multiplier_states == []
    assert optimizer.termination_reason.startswith('inner-iteration-limit:')


# =====================================================================
# C3: 连续 hold_steps 个外层步同时满足 C0-C2 才终止
# =====================================================================

def test_c3_requires_three_consecutive_clean_outer_steps():
    zero = np.zeros(10)
    optimizer, _, _, calls = run_case([zero], outer=8)
    # 每个外层步一次子问题求解; 第 1 步就已达标, 但必须数满 3 步才收敛。
    assert calls == 3
    assert optimizer.converged is True
    assert optimizer.termination_reason.startswith("criterion-met")
    assert "C0-C2" in optimizer.termination_reason


def test_c3_hold_count_restarts_after_a_bad_step():
    zero = np.zeros(10)
    bad = np.full(10, 0.01)          # 超过 change_tolerance, C1 失格
    optimizer, _, _, calls = run_case([zero, bad, zero, zero, zero], outer=8)
    # 达标(1) -> 坏步(归零) -> 达标(1,2,3): 共 5 个外层步。
    assert calls == 5
    assert optimizer.converged is True


def test_c3_not_met_within_budget_reports_max_iterations():
    bad = np.full(10, 0.01)
    optimizer, _, _, calls = run_case([bad], outer=4)
    assert calls == 4
    assert optimizer.converged is False
    assert optimizer.termination_reason.startswith("max-al-iterations-reached")


# =====================================================================
# 罚因子 mu 的条件放大
# =====================================================================

def make_al_objective(rule, tau=0.5, alpha=1.1, mu=100.0, mu_max=1.0e4,
                      lambda_max=None):
    objective = object.__new__(AugmentedLagrangianObjective)
    objective._options = SimpleNamespace(
        alpha=alpha, mu_update_rule=rule, mu_violation_ratio=tau,
        lambda_max=lambda_max)
    objective.mu = mu
    objective.mu_max = mu_max
    objective.lambda_max = lambda_max
    objective.last_capped_count = 0
    objective.lamb = np.zeros((3, 1))
    objective._cache_h = np.zeros((3, 1))
    objective._prev_violation = None
    objective._pending_violation = None

    return objective


def test_unconditional_rule_grows_every_outer_step():
    objective = make_al_objective("unconditional")
    for _ in range(3):
        before = objective.mu
        objective.set_current_violation(0.0)
        objective.update_multipliers()
        assert objective.mu == pytest.approx(1.1 * before)


def test_conditional_rule_freezes_when_violation_drops_enough():
    objective = make_al_objective("conditional")
    objective.set_current_violation(1.0)
    objective.update_multipliers()          # 首步无历史, 退回放大
    grown = objective.mu
    objective.set_current_violation(0.4)    # 0.4 < 0.5 * 1.0, 下降充分
    objective.update_multipliers()
    assert objective.mu == pytest.approx(grown)


def test_conditional_rule_grows_when_violation_stalls():
    objective = make_al_objective("conditional")
    objective.set_current_violation(1.0)
    objective.update_multipliers()
    grown = objective.mu
    objective.set_current_violation(1.0)    # 没有下降
    objective.update_multipliers()
    assert objective.mu == pytest.approx(1.1 * grown)


def test_conditional_rule_falls_back_to_growth_without_measure():
    objective = make_al_objective("conditional")
    objective.set_current_violation(1.0)
    objective.update_multipliers()
    grown = objective.mu
    objective.update_multipliers()          # 优化器没推入度量
    assert objective.mu == pytest.approx(1.1 * grown)


def test_conditional_rule_freezes_once_feasible():
    objective = make_al_objective("conditional")
    objective.set_current_violation(0.0)    # 截断在 0
    objective.update_multipliers()
    grown = objective.mu
    for _ in range(3):
        objective.set_current_violation(-0.2)
        objective.update_multipliers()
    # 真正可行后 v = 0, "0 > tau * 0" 恒假, mu 冻结。
    assert objective.mu == pytest.approx(grown)


# =====================================================================
# 末期移动限制衰减的判别量
# =====================================================================

def test_coherent_share_is_zero_for_period_two_oscillation():
    base = np.full(20, 0.5)
    bump = np.zeros(20)
    bump[:5] = 0.02
    window = [base + bump if j % 2 else base for j in range(11)]
    assert coherent_travel_share(window) == pytest.approx(0.0)


def test_coherent_share_is_one_for_monotone_drift():
    base = np.full(20, 0.5)
    window = [base + 0.01 * j for j in range(11)]
    assert coherent_travel_share(window) == pytest.approx(1.0)


def test_coherent_share_ignores_incoherent_majority():
    # 2 个单元单调重组、18 个单元 period-2 抖动: 全局 net/travel 会被抖动淹没,
    # 而按单元判定的相干行程份额仍能把重组识别出来。
    steps = 10
    window = []
    for j in range(steps + 1):
        state = np.full(20, 0.5)
        state[:2] += 0.02 * j
        state[2:] += 0.02 * (j % 2)
        window.append(state)
    share = coherent_travel_share(window, cell_ratio=0.7)
    assert share == pytest.approx(2.0 / 20.0)


def test_decay_fires_on_limit_cycle_but_not_on_drift():
    decay_options = dict(
        move_limit=0.15, move_limit_decay=0.7, move_limit_min=0.005,
        move_limit_progress_window=4, move_limit_progress_ratio=0.3,
        move_limit_progress_cell=0.7,
    )
    step = np.full(10, 0.01)
    cycle = [step, -step, step, -step, step, -step, step, -step]
    optimizer, _, _, _ = run_case(cycle, outer=8, **decay_options)
    assert optimizer.move_limit_decays > 0
    assert optimizer.effective_move_limit < 0.15

    optimizer, _, _, _ = run_case([step], outer=8, **decay_options)
    assert optimizer.move_limit_decays == 0
    assert optimizer.effective_move_limit == pytest.approx(0.15)


def test_decay_is_off_by_default():
    step = np.full(10, 0.01)
    cycle = [step, -step, step, -step, step, -step, step, -step]
    optimizer, _, _, _ = run_case(cycle, outer=8, move_limit=0.15)
    assert optimizer.move_limit_decays == 0
    assert optimizer.effective_move_limit == pytest.approx(0.15)


# =====================================================================
# 乘子安全阈 lambda_max (safeguarded AL): lambda <- P_[0, lambda_max](lambda + mu h)
# =====================================================================

def test_lambda_cap_clips_multiplier_after_update():
    objective = make_al_objective("unconditional", mu=100.0, lambda_max=250.0)
    objective.lamb = np.array([[200.0], [10.0], [0.0]])
    objective._cache_h = np.array([[1.0], [0.5], [0.0]])   # 300 / 60 / 0
    objective.update_multipliers()
    np.testing.assert_allclose(objective.lamb, [[250.0], [60.0], [0.0]])
    assert objective.last_capped_count == 1
    assert objective.mu == pytest.approx(110.0)             # 罚因子放大不受影响

    objective._cache_h = np.array([[0.0], [0.0], [0.0]])
    objective.update_multipliers()
    assert objective.last_capped_count == 0                 # 无人越界时计数归零


def test_lambda_cap_none_reproduces_unbounded_update():
    objective = make_al_objective("unconditional", mu=100.0, lambda_max=None)
    objective.lamb = np.array([[200.0], [10.0], [0.0]])
    objective._cache_h = np.array([[1.0], [0.5], [0.0]])
    objective.update_multipliers()
    np.testing.assert_allclose(objective.lamb, [[300.0], [60.0], [0.0]])
    assert objective.last_capped_count == 0


def test_lambda_max_option_is_validated():
    ALMMMAOptions(lambda_max=None)
    ALMMMAOptions(lambda_max=3000.0)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="lambda_max"):
            ALMMMAOptions(lambda_max=bad)
    ALMMMAOptions(acceptance_solid_threshold=0.5)
    for bad in (0.0, 1.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="acceptance_solid_threshold"):
            ALMMMAOptions(acceptance_solid_threshold=bad)


# =====================================================================
# C2 的实体验收子集 acceptance_solid_threshold
# =====================================================================

# 前 5 个单元实体 (0.9), 后 5 个灰度 (0.2); FakeFilter 不滤波, rho_phys 即设计变量
_MIXED_DENSITY = np.array([0.9] * 5 + [0.2] * 5)


def test_c2_ignores_grey_cells_when_solid_threshold_set():
    violation = np.array([-0.1] * 5 + [0.01] * 5)          # 灰度单元违约, 实体单元可行
    optimizer, _, history, calls = run_case(
        [np.zeros(10)], violation=violation, outer=8,
        initial=_MIXED_DENSITY, acceptance_solid_threshold=0.5)
    assert optimizer.converged is True
    assert calls == 3                                        # hold_steps=3 后即退出
    assert "max_rel_solid=-1.000e-01" in optimizer.termination_reason
    assert "threshold=0.5" in optimizer.termination_reason
    # 全域诊断量不漂移, C2 量单独入 history
    assert history.scalar_histories["max_relative_violation"][-1] == pytest.approx(0.01)
    assert history.scalar_histories["max_relative_violation_solid"][-1] == pytest.approx(-0.1)
    assert history.scalar_histories["max_multiplier"][-1] == pytest.approx(0.0)


def test_c2_still_fails_on_solid_violation():
    violation = np.array([0.01] * 5 + [-0.1] * 5)          # 实体单元违约
    optimizer, _, _, calls = run_case(
        [np.zeros(10)], violation=violation, outer=8,
        initial=_MIXED_DENSITY, acceptance_solid_threshold=0.5)
    assert optimizer.converged is False
    assert calls == 8
    assert optimizer.termination_reason.startswith("max-al-iterations-reached")


def test_c2_threshold_none_keeps_global_semantics():
    violation = np.array([-0.1] * 5 + [0.01] * 5)
    optimizer, _, history, calls = run_case(
        [np.zeros(10)], violation=violation, outer=8, initial=_MIXED_DENSITY)
    assert optimizer.converged is False
    assert calls == 8
    assert "threshold=global" in optimizer.termination_reason
    assert history.scalar_histories["max_relative_violation_solid"][-1] == pytest.approx(0.01)


def test_c2_solid_subset_falls_back_to_global_when_empty():
    violation = np.full(10, 0.01)
    optimizer, _, _, calls = run_case(
        [np.zeros(10)], violation=violation, outer=8,
        initial=np.full(10, 0.2), acceptance_solid_threshold=0.5)
    assert optimizer.converged is False
    assert calls == 8
