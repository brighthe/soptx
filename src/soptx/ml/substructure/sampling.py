"""子结构代理模型的可复现密度场采样。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np


SAMPLER_VERSION = "mixed-topology-v2"


@dataclass(frozen=True)
class SamplingFractions:
    """混合采样各组成部分的比例。"""

    continuous: float = 0.25
    low_density: float = 0.25
    near_binary: float = 0.25
    correlated: float = 0.25
    trajectory: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "continuous": self.continuous,
            "low_density": self.low_density,
            "near_binary": self.near_binary,
            "correlated": self.correlated,
            "trajectory": self.trajectory,
        }

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(value < 0.0 for value in values.values()):
            raise ValueError("采样比例不能为负数。")
        if not np.isclose(sum(values.values()), 1.0):
            raise ValueError("采样比例之和必须为 1。")


@dataclass(frozen=True)
class DensitySamplingConfig:
    """局部密度场混合采样配置。"""

    shape: tuple[int, ...]
    design_min: float
    design_max: float = 1.0
    seed: int = 2026
    fractions: SamplingFractions = field(default_factory=SamplingFractions)
    low_density_upper: float = 0.3
    correlation_steps: int = 2
    near_binary_transition_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("shape 必须由正整数构成。")
        if not 0.0 < self.design_min < self.design_max <= 1.0:
            raise ValueError("design_min 与 design_max 必须满足 0 < min < max <= 1。")
        if not self.design_min < self.low_density_upper <= self.design_max:
            raise ValueError("low_density_upper 必须位于设计变量范围内。")
        if self.correlation_steps < 1:
            raise ValueError("correlation_steps 必须至少为 1。")
        if not 0.0 <= self.near_binary_transition_fraction <= 1.0:
            raise ValueError("near_binary_transition_fraction 必须位于 [0, 1]。")


@dataclass(frozen=True)
class DensitySamples:
    """采样得到的密度张量及逐样本来源。"""

    values: np.ndarray
    sources: tuple[str, ...]
    sampler_version: str = SAMPLER_VERSION

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim < 2:
            raise ValueError("values 必须包含样本维和至少一个空间维。")
        if values.shape[0] != len(self.sources):
            raise ValueError("sources 数量必须与样本数一致。")
        frozen = np.array(values, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)

    @property
    def source_counts(self) -> Mapping[str, int]:
        unique, counts = np.unique(np.asarray(self.sources), return_counts=True)
        return MappingProxyType(
            {str(name): int(count) for name, count in zip(unique, counts)}
        )


def _allocate_counts(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _smooth_fields(fields: np.ndarray, steps: int) -> np.ndarray:
    result = np.asarray(fields, dtype=np.float64)
    spatial_ndim = result.ndim - 1
    for _ in range(steps):
        accumulated = result.copy()
        weight = 1
        for axis in range(1, spatial_ndim + 1):
            left = np.take(result, np.arange(result.shape[axis]) - 1, axis=axis, mode="clip")
            right = np.take(result, np.arange(result.shape[axis]) + 1, axis=axis, mode="clip")
            accumulated = accumulated + left + right
            weight += 2
        result = accumulated / weight
    return result


def _sample_continuous(
    count: int,
    config: DensitySamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(
        config.design_min,
        config.design_max,
        size=(count, *config.shape),
    )


def _sample_low_density(
    count: int,
    config: DensitySamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    log_min = np.log(config.design_min)
    log_max = np.log(config.low_density_upper)
    return np.exp(rng.uniform(log_min, log_max, size=(count, *config.shape)))


def _sample_near_binary(
    count: int,
    config: DensitySamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    volume_fraction = rng.uniform(0.2, 0.8, size=(count,) + (1,) * len(config.shape))
    solid = rng.random(size=(count, *config.shape)) < volume_fraction
    values = np.where(solid, config.design_max, config.design_min).astype(np.float64)
    transition = rng.random(size=values.shape) < config.near_binary_transition_fraction
    intermediate = rng.uniform(config.design_min, config.design_max, size=values.shape)
    return np.where(transition, intermediate, values)


def _sample_correlated(
    count: int,
    config: DensitySamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(size=(count, *config.shape))
    smooth = _smooth_fields(noise, config.correlation_steps)
    flat = smooth.reshape(count, -1)
    target_fraction = rng.uniform(0.2, 0.8, size=count)
    thresholds = np.asarray(
        [np.quantile(row, 1.0 - fraction) for row, fraction in zip(flat, target_fraction)]
    )
    scale = np.std(flat, axis=1)
    scale = np.maximum(scale, np.finfo(np.float64).eps)
    reshape = (count,) + (1,) * len(config.shape)
    projected = 1.0 / (
        1.0
        + np.exp(
            -6.0
            * (smooth - thresholds.reshape(reshape))
            / scale.reshape(reshape)
        )
    )
    return config.design_min + (config.design_max - config.design_min) * projected


def _sample_trajectory(
    count: int,
    config: DensitySamplingConfig,
    rng: np.random.Generator,
    trajectory: Optional[np.ndarray],
) -> np.ndarray:
    if trajectory is None:
        raise ValueError("trajectory 比例非零时必须提供独立的轨迹密度样本。")
    values = np.asarray(trajectory, dtype=np.float64)
    expected_tail = tuple(config.shape)
    if values.ndim != len(expected_tail) + 1 or tuple(values.shape[1:]) != expected_tail:
        raise ValueError(
            f"trajectory 形状必须为 (n, {expected_tail}); 当前为 {values.shape}。"
        )
    if len(values) == 0:
        raise ValueError("trajectory 不能为空。")
    indices = rng.choice(len(values), size=count, replace=len(values) < count)
    return np.clip(values[indices], config.design_min, config.design_max)


def sample_density_fields(
    n_samples: int,
    config: DensitySamplingConfig,
    *,
    trajectory: Optional[np.ndarray] = None,
) -> DensitySamples:
    """按配置生成覆盖拓扑优化状态的混合局部密度场。"""
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数。")
    rng = np.random.default_rng(config.seed)
    counts = _allocate_counts(n_samples, config.fractions.as_dict())
    generators = {
        "continuous": lambda count: _sample_continuous(count, config, rng),
        "low_density": lambda count: _sample_low_density(count, config, rng),
        "near_binary": lambda count: _sample_near_binary(count, config, rng),
        "correlated": lambda count: _sample_correlated(count, config, rng),
        "trajectory": lambda count: _sample_trajectory(count, config, rng, trajectory),
    }
    blocks: list[np.ndarray] = []
    sources: list[str] = []
    for name, count in counts.items():
        if count == 0:
            continue
        blocks.append(generators[name](count))
        sources.extend([name] * count)
    values = np.concatenate(blocks, axis=0)
    permutation = rng.permutation(n_samples)
    shuffled_sources = tuple(sources[index] for index in permutation)
    return DensitySamples(values[permutation], shuffled_sources)
