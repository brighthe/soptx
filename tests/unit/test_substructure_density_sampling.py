"""子结构密度采样器测试。"""

import numpy as np

from soptx.ml.substructure import (
    DensitySamplingConfig,
    SamplingFractions,
    sample_density_fields,
)


def _neighbor_difference(values: np.ndarray) -> float:
    dx = np.abs(values[:, 1:, :] - values[:, :-1, :]).mean()
    dy = np.abs(values[:, :, 1:] - values[:, :, :-1]).mean()
    return float(0.5 * (dx + dy))


def test_mixed_density_sampling_is_reproducible_and_covers_required_states() -> None:
    config = DensitySamplingConfig(shape=(5, 5), design_min=1.0e-3, seed=17)

    first = sample_density_fields(400, config)
    second = sample_density_fields(400, config)

    np.testing.assert_array_equal(first.values, second.values)
    assert first.sources == second.sources
    assert first.values.shape == (400, 5, 5)
    assert float(first.values.min()) >= config.design_min
    assert float(first.values.max()) <= config.design_max
    assert dict(first.source_counts) == {
        "continuous": 100,
        "correlated": 100,
        "low_density": 100,
        "near_binary": 100,
    }

    low = first.values[np.asarray(first.sources) == "low_density"]
    binary = first.values[np.asarray(first.sources) == "near_binary"]
    correlated = first.values[np.asarray(first.sources) == "correlated"]
    continuous = first.values[np.asarray(first.sources) == "continuous"]
    assert float((low < 0.3).mean()) == 1.0
    near_endpoint = (binary < 0.02) | (binary > 0.98)
    assert float(near_endpoint.mean()) > 0.9
    assert _neighbor_difference(correlated) < _neighbor_difference(continuous)


def test_trajectory_sampling_requires_explicit_independent_trajectory() -> None:
    fractions = SamplingFractions(
        continuous=0.0,
        low_density=0.0,
        near_binary=0.0,
        correlated=0.0,
        trajectory=1.0,
    )
    config = DensitySamplingConfig(
        shape=(2, 2),
        design_min=1.0e-3,
        fractions=fractions,
    )

    try:
        sample_density_fields(4, config)
    except ValueError as error:
        assert "trajectory" in str(error)
    else:
        raise AssertionError("缺少轨迹样本时应拒绝 trajectory 采样。")

    trajectory = np.asarray(
        [
            [[0.0, 0.2], [0.8, 1.2]],
            [[0.1, 0.3], [0.7, 0.9]],
        ]
    )
    samples = sample_density_fields(4, config, trajectory=trajectory)
    assert dict(samples.source_counts) == {"trajectory": 4}
    assert float(samples.values.min()) >= config.design_min
    assert float(samples.values.max()) <= config.design_max
