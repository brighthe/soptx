"""应力松弛公式与通用离散适配器的回归测试."""

from types import SimpleNamespace

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from soptx.topology.constraints import (
    EpsilonRelaxedStressFormulation,
    HuZhangStressConstraint,
    LagrangeStressConstraint,
    PolynomialVanishingStressFormulation,
)


class _Material:
    youngs_modulus = 1.0

    @staticmethod
    def calculate_von_mises_stress(stress):
        return np.asarray(stress)[..., 0]


class _LagrangeAnalyzer:
    def __init__(self):
        self.material = _Material()
        self.interpolation_scheme = SimpleNamespace(n_sub=None)
        self._cached_stiffness_relative = np.array([0.25, 0.75])
        self.poisson_ratio_interpolated = False
        # 约束类要按单元数校验豁免掩码, 替身也得报出单元数.
        self.disp_mesh = SimpleNamespace(number_of_cells=lambda: 2)

    @staticmethod
    def compute_stress_state(state):
        return {
            "stress_solid": np.array(
                [
                    [[2.0, 0.0, 0.0]],
                    [[0.5, 0.0, 0.0]],
                ]
            )
        }


class _HuZhangAnalyzer:
    def __init__(self):
        self.material = _Material()
        self._E_rho = np.array([0.25, 0.75])
        self.disp_mesh = SimpleNamespace(number_of_cells=lambda: 2)

    @staticmethod
    def compute_stress_state(state, rho_val):
        return {
            "stress_apparent": np.array(
                [
                    [[0.5, 0.0, 0.0]],
                    [[0.375, 0.0, 0.0]],
                ]
            )
        }


def test_epsilon_formulation_is_representation_invariant():
    """等价的实体应力与表观应力输入应给出相同约束值."""
    bm.set_backend("numpy")
    formulation = EpsilonRelaxedStressFormulation(epsilon=0.1)
    solid_ratio = np.array([[2.0], [0.5]])
    stiffness_ratio = np.array([0.25, 0.75])
    apparent_ratio = stiffness_ratio[:, None] * solid_ratio

    solid_g = formulation.constraint_value(
        solid_ratio,
        stiffness_ratio,
        "solid",
    )
    apparent_g = formulation.constraint_value(
        apparent_ratio,
        stiffness_ratio,
        "apparent",
    )

    np.testing.assert_allclose(solid_g, apparent_g)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_epsilon_formulation_accepts_closed_interval_boundaries(epsilon):
    formulation = EpsilonRelaxedStressFormulation(epsilon=epsilon)
    assert formulation.epsilon == epsilon


@pytest.mark.parametrize("epsilon", [-1e-8, 1.000001, np.nan, np.inf])
def test_epsilon_formulation_rejects_invalid_values(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonRelaxedStressFormulation(epsilon=epsilon)


@pytest.mark.parametrize(
    ("formulation", "representation"),
    [
        (EpsilonRelaxedStressFormulation(0.03), "solid"),
        (EpsilonRelaxedStressFormulation(0.03), "apparent"),
        (PolynomialVanishingStressFormulation(), "solid"),
    ],
)
@pytest.mark.parametrize("multiresolution", [False, True])
def test_formulation_partials_match_finite_difference(
    formulation,
    representation,
    multiresolution,
):
    """独立扰动原生应力与刚度, 核对公式给出的两个偏导."""
    bm.set_backend("numpy")
    shape = (2, 3, 2) if multiresolution else (3, 2)
    stress = np.linspace(0.3, 1.8, int(np.prod(shape))).reshape(shape)
    stiffness = np.linspace(0.1, 0.9, int(np.prod(shape[:-1]))).reshape(shape[:-1])
    step = 1e-6

    finite_stress = (
        formulation.constraint_value(stress + step, stiffness, representation)
        - formulation.constraint_value(stress - step, stiffness, representation)
    ) / (2 * step)
    finite_stiffness = (
        formulation.constraint_value(stress, stiffness + step, representation)
        - formulation.constraint_value(stress, stiffness - step, representation)
    ) / (2 * step)

    np.testing.assert_allclose(
        formulation.gradient_wrt_stress_ratio(
            stress,
            stiffness,
            representation,
        ),
        finite_stress,
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        formulation.partial_wrt_stiffness_ratio(
            stress,
            stiffness,
            representation,
        ),
        finite_stiffness,
        rtol=1e-8,
        atol=1e-9,
    )


def test_polynomial_acceptance_proxy_matches_reported_stress_measure():
    """Polynomial 历史验收量应与其展示应力测度保持同一尺度."""
    bm.set_backend("numpy")
    formulation = PolynomialVanishingStressFormulation()
    solid_ratio = np.array([[2.0], [0.5]])
    stiffness_ratio = np.array([0.25, 0.75])

    measure = formulation.stress_measure(
        solid_ratio,
        stiffness_ratio,
        "solid",
    )
    violation = formulation.acceptance_violation(
        solid_ratio,
        stiffness_ratio,
        "solid",
    )

    np.testing.assert_allclose(violation, measure - 1.0)


def test_generic_adapters_match_for_equivalent_epsilon_state():
    """两个原生应力表示通过相同 formulation 后应得到同一离散约束."""
    bm.set_backend("numpy")
    density = np.array([0.25, 0.75])
    formulation = EpsilonRelaxedStressFormulation(0.1)
    lfem = LagrangeStressConstraint(
        analyzer=_LagrangeAnalyzer(),
        stress_limit=1.0,
        formulation=formulation,
    )
    huzhang = HuZhangStressConstraint(
        analyzer=_HuZhangAnalyzer(),
        stress_limit=1.0,
        formulation=formulation,
    )
    lfem_state = {}
    huzhang_state = {}

    lfem_value = lfem.fun(density=density, state=lfem_state)
    huzhang_value = huzhang.fun(density=density, state=huzhang_state)

    np.testing.assert_allclose(lfem_value, huzhang_value)
    np.testing.assert_allclose(
        lfem.compute_relative_violation(density, lfem_state),
        lfem_value,
    )
    np.testing.assert_allclose(
        huzhang.compute_relative_violation(density, huzhang_state),
        huzhang_value,
    )


class _IndependentQuadraticFormulation:
    """不继承内置公式的测试模型, 用于验证结构化协议接入."""

    name = "test_quadratic"

    def constraint_value(self, stress_ratio, stiffness_ratio, stress_representation):
        return stress_ratio**2 - stiffness_ratio[..., None]

    def partial_wrt_stiffness_ratio(
        self,
        stress_ratio,
        stiffness_ratio,
        stress_representation,
    ):
        return -np.ones_like(stress_ratio)

    def gradient_wrt_stress_ratio(
        self,
        stress_ratio,
        stiffness_ratio,
        stress_representation,
    ):
        return 2 * stress_ratio

    def stress_measure(self, stress_ratio, stiffness_ratio, stress_representation):
        return stiffness_ratio[..., None] * stress_ratio

    def acceptance_violation(
        self,
        stress_ratio,
        stiffness_ratio,
        stress_representation,
    ):
        return self.constraint_value(
            stress_ratio,
            stiffness_ratio,
            stress_representation,
        )

    def threshold(self, stiffness_ratio):
        return None


@pytest.mark.parametrize(
    ("constraint_type", "analyzer", "expected"),
    [
        (
            LagrangeStressConstraint,
            _LagrangeAnalyzer(),
            np.array([[3.75], [-0.5]]),
        ),
        (
            HuZhangStressConstraint,
            _HuZhangAnalyzer(),
            np.array([[0.0], [-0.609375]]),
        ),
    ],
)
def test_independent_model_connects_to_both_generic_adapters(
    constraint_type,
    analyzer,
    expected,
):
    """第三种公式应通过注入接入两个适配器, 无需新增约束类."""
    bm.set_backend("numpy")
    formulation = _IndependentQuadraticFormulation()
    constraint = constraint_type(
        analyzer=analyzer,
        stress_limit=1.0,
        formulation=formulation,
    )
    density = np.array([0.25, 0.75])
    state = {}

    values = constraint.fun(density=density, state=state)

    np.testing.assert_allclose(values, expected)
    np.testing.assert_allclose(
        constraint.compute_relative_violation(density, state),
        values,
    )
    assert constraint.formulation is formulation
    assert "eta_threshold" not in state