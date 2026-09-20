"""AL-MMA 更新后状态与停止准则的回归测试, 不执行有限元求解."""
from types import SimpleNamespace

import numpy as np
import pytest
from fealpy.backend import backend_manager as bm
from soptx.topology.optimizers.al_mma import ALMMMAOptimizer


class FakeObjective:
    """violation 可为标量 (全单元同值) 或长度等于单元数的数组 (逐单元给值)."""

    def __init__(self, violation):
        self.violation = violation
        self.solved = []
        self.multiplier_states = []
        self._analyzer = SimpleNamespace(
            pde=SimpleNamespace(),
            interpolation_scheme=SimpleNamespace(penalty_factor=3.5),
            solve_state=self.solve,
        )
        self._volume_objective = SimpleNamespace(_v=0.0, _v0=1.0)
        self._stress_constraint = SimpleNamespace(
            compute_stress_measure=self.stress,
            compute_relative_violation=self.relative_violation,
        )
        self.mu = 1.0
        self.lamb = np.zeros(1)

    def solve(self, rho_val):
        self.solved.append(rho_val.copy())
        return {"rho": rho_val.copy()}

    def fun(self, density, state):
        np.testing.assert_array_equal(density, state["rho"])
        self.current = density.copy()
        self._cache_g = self._violation_column(len(density)) * 0.01
        self._volume_objective._v = float(density.mean())
        return self._volume_objective._v

    def jac(self, density, state):
        np.testing.assert_array_equal(density, state["rho"])
        return np.ones_like(density)

    def stress(self, rho, state):
        np.testing.assert_array_equal(rho, state["rho"])
        return state["rho"][:, None]

    def _violation_column(self, n):
        column = np.asarray(self.violation, dtype=float).reshape(-1, 1)
        return np.broadcast_to(column, (n, 1)).copy()

    def relative_violation(self, rho, state):
        np.testing.assert_array_equal(rho, state["rho"])
        return self._violation_column(len(rho))

    def update_multipliers(self):
        self.multiplier_states.append(self.current.copy())


class FakeFilter:
    def __init__(self, continuation=False):
        self.continuation = continuation
        self.calls = 0
        self.offset = 0.0

    def get_initial_density(self, density):
        return density.copy()

    def filter_design_variable(self, design_variable, physical_density):
        return design_variable.copy() + self.offset

    def filter_objective_sensitivities(self, design_variable, obj_grad_rho):
        return obj_grad_rho.copy()

    def continuation_step(self, change):
        self.calls += 1
        if self.continuation and self.calls == 1:
            self.offset = 0.1
            return change, True
        return change, False


def run_case(updates, violation=-0.1, continuation=False, outer=2, inner=2):
    bm.set_backend("numpy")
    objective = FakeObjective(violation)
    optimizer = object.__new__(ALMMMAOptimizer)
    optimizer.options = SimpleNamespace(
        max_al_iterations=outer, mma_iters_per_al=inner,
        change_tolerance=0.002, stress_tolerance=0.003,
        # 本文件的用例查的是状态一致性与相对超限量, 不查 C3 持续性;
        # 取 hold_steps=1 保持"一步达标即退出"的原有语义, 否则每个用例都要
        # 多喂 2 个外层步才退出, 断言的观察点会被推后。
        hold_steps=1,
        use_penalty_continuation=False,
        initialize_problem_params=lambda m, n: None,
    )
    optimizer._al_objective = objective
    optimizer._filter = FakeFilter(continuation)
    optimizer._log_info = lambda message: None
    optimizer._update_penalty = lambda iter_idx: None
    calls = []

    def update(**kwargs):
        delta = updates[min(len(calls), len(updates) - 1)]
        calls.append(1)
        return kwargs["z"].copy() + delta

    optimizer._solve_unconstrained_subproblem = update
    initial = np.full(10, 0.5)
    rho, history = optimizer.optimize(initial, initial)
    return rho, history, objective, len(calls)


def test_local_change_is_not_hidden_by_mean():
    delta = np.zeros(10)
    delta[0] = 0.01  # 平均值 0.001 达标, 最大值 0.01 不达标.
    _, history, _, calls = run_case([delta, np.zeros(10)])
    assert calls == 2
    assert history.changes[0] == pytest.approx(0.01)


def test_updated_density_state_history_and_multiplier_agree():
    """算法 2 步骤 4: MMA 更新后对新设计重新分析, 历史、退出判据与乘子更新同属该设计."""
    rho, history, objective, calls = run_case([np.full(10, 0.001)])
    assert calls == 1
    np.testing.assert_array_equal(objective.solved[-1], rho)
    np.testing.assert_array_equal(objective.multiplier_states[-1], rho)
    assert history.scalar_histories["volfrac"][-1] == pytest.approx(rho.mean())
    assert history.scalar_histories["max_von_mises"][-1] == pytest.approx(rho.max())
    np.testing.assert_array_equal(history.physical_densities[-1], rho)


def test_relative_violation_prevents_early_exit_despite_small_raw_constraint():
    _, history, objective, calls = run_case([np.zeros(10)], violation=0.01)
    assert calls == 4
    assert history.scalar_histories["max_constraint"][-1] == pytest.approx(0.0001)
    assert history.scalar_histories["max_relative_violation"][-1] == pytest.approx(0.01)
    # 状态跨步复用: 初始设计 1 次 + 每次 MMA 更新后 1 次, 内层步头部不重解。
    assert len(objective.solved) == 1 + calls


def test_projection_change_requires_new_state_before_exit():
    rho, history, objective, calls = run_case([np.zeros(10)], continuation=True)
    assert calls == 2
    np.testing.assert_allclose(rho, 0.6)
    np.testing.assert_array_equal(objective.solved[-1], rho)
    assert history.scalar_histories["max_von_mises"][-1] == pytest.approx(0.6)
    # beta 更新重过滤了物理密度, 下一外层步头部必须重解: 1 + 2 次更新 + 1 次补解。
    assert len(objective.solved) == 4
