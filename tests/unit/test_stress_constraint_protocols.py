"""应力约束协议与原问题 KKT 诊断的纯代数回归测试."""

from types import SimpleNamespace

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.topology.objectives import AugmentedLagrangianObjective
from soptx.topology.optimizers.al_mma import ALMMMAOptimizer, ALMMMAOptions



class _FakeVolume:
    def fun(self, density, state):
        return 0.0

    def jac(self, density, state):
        return np.zeros_like(density)


class _FakeConstraint:
    _stress_limit = 1.0

    def fun(self, density, state):
        self.fun_calls = getattr(self, "fun_calls", 0) + 1
        state["stiffness_ratio"] = np.ones_like(density)
        return np.full((density.size, 1), getattr(self, "value", 0.5))

    def compute_gradient_wrt_von_mises(self, state):
        return np.ones((state["stiffness_ratio"].size, 1))

    def compute_partial_gradient_wrt_mE(self, state):
        return np.zeros((state["stiffness_ratio"].size, 1))

    def compute_adjoint_load(self, dPenaldVM, state):
        return dPenaldVM[:, 0]

    def compute_implicit_sensitivity_term(self, adjoint_vector, state):
        return adjoint_vector


class _FakeAnalyzer:
    interpolation_scheme = SimpleNamespace(
        interpolate_material_derivative=lambda material, rho_val: np.ones_like(rho_val)
    )
    material = SimpleNamespace(youngs_modulus=1.0)

    def solve_adjoint(self, rhs, rho_val):
        return rhs


def _fake_al_objective():
    objective = object.__new__(AugmentedLagrangianObjective)
    objective._volume_objective = _FakeVolume()
    objective._stress_constraint = _FakeConstraint()
    objective._is_multiresolution = False
    objective._interpolation_scheme = _FakeAnalyzer.interpolation_scheme
    objective._material = _FakeAnalyzer.material
    objective._analyzer = _FakeAnalyzer()
    objective._cache_g = None
    objective._cache_h = None
    objective.lamb = np.array([[2.0], [3.0]])
    objective._diff_mode = "manual"
    objective.mu = 10.0
    return objective


def test_original_lagrangian_gradient_does_not_use_augmented_weight():
    bm.set_backend("numpy")
    objective = _fake_al_objective()
    density = np.array([0.4, 0.6])
    state = {}

    grad_lagrangian = objective.lagrangian_jac(density=density, state=state)
    grad_augmented = objective.jac(density=density, state=state)

    np.testing.assert_allclose(grad_lagrangian, np.array([1.0, 1.5]))
    assert not np.allclose(grad_lagrangian, grad_augmented)


def test_manual_gradient_refreshes_constraint_cache_without_state_key_contract():
    bm.set_backend("numpy")
    objective = _fake_al_objective()
    density = np.array([0.4, 0.6])
    state = {}

    objective._stress_constraint.value = -0.5
    objective.fun(density=density, state=state)
    np.testing.assert_allclose(objective._cache_h, np.array([[-0.2], [-0.3]]))
    objective.lamb = np.array([[8.0], [8.0]])
    objective.mu = 2.0

    gradient = objective.jac(density=density, state=state)

    np.testing.assert_allclose(gradient, np.array([3.5, 3.5]))
    np.testing.assert_allclose(objective._cache_h, np.full((2, 1), -0.5))



class _IdentityFilter:
    def filter_objective_sensitivities(self, design_variable, obj_grad_rho):
        return np.asarray(obj_grad_rho)


def test_kkt_stationarity_uses_original_box_not_move_limit():
    bm.set_backend("numpy")
    optimizer = object.__new__(ALMMMAOptimizer)
    optimizer._filter = _IdentityFilter()
    optimizer._al_objective = SimpleNamespace(
        lagrangian_jac=lambda density, state: np.array([0.2, -0.3]),
        _cache_g=np.array([[-0.1], [-0.2]]),
        lamb=np.array([[1.0], [0.0]]),
        _NC=2,
    )
    optimizer.options = SimpleNamespace(
        stress_tolerance=0.003,
        move_limit=1.0e-6,
        kkt_diagnostics_enabled=True,
        kkt_acceptance_enabled=False,
    )

    diagnostics = optimizer.kkt_diagnostics(
        design_variable=np.array([0.5, 0.5]),
        density_distribution=np.array([0.5, 0.5]),
        state={},
    )
    assert diagnostics["stationarity"] == pytest.approx(0.3)
    assert diagnostics["stationarity"] > optimizer.options.move_limit
    assert diagnostics["one_constraint_per_cell"] is True


def test_kkt_acceptance_requires_explicit_positive_tolerances():
    with pytest.raises(ValueError, match="三个正的 KKT 容差"):
        ALMMMAOptions(
            kkt_diagnostics_enabled=True,
            kkt_acceptance_enabled=True,
        )


def test_kkt_tolerances_reject_nonfinite_values_even_without_acceptance():
    with pytest.raises(ValueError, match="有限的非负数"):
        ALMMMAOptions(kkt_stationarity_tolerance=float("nan"))
    with pytest.raises(ValueError, match="有限的非负数"):
        ALMMMAOptions(kkt_dual_tolerance=float("inf"))
