"""规则三维网格公共灵敏度过滤器的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.topology.filters import (
    apply_structured_density_filter,
    apply_structured_density_filter_adjoint,
    apply_structured_sensitivity_filter,
    build_structured_cone_kernel,
)


def test_anisotropic_cone_kernel_uses_physical_distance():
    """非立方体单元的权重应由真实物理距离决定."""
    kernel = build_structured_cone_kernel(1.1, (1.0, 0.5, 0.25))
    center = tuple(size // 2 for size in kernel.shape)

    assert kernel.shape == (3, 5, 9)
    assert kernel[center] == pytest.approx(1.1)
    assert kernel[center[0] + 1, center[1], center[2]] == pytest.approx(0.1)
    assert kernel[center[0], center[1] + 2, center[2]] == pytest.approx(0.1)
    assert kernel[center[0], center[1], center[2] + 4] == pytest.approx(0.1)


def test_cone_filter_preserves_constant_sensitivity():
    """均匀密度下的常灵敏度应在边界归一化后保持不变."""
    density = np.full((4, 3, 2), 0.4)
    sensitivity = np.full_like(density, -2.5)

    filtered = apply_structured_sensitivity_filter(
        sensitivity,
        density,
        rmin=1.3,
        spacing=(1.0, 0.7, 0.4),
        kind="cone",
    )

    np.testing.assert_allclose(filtered, sensitivity, rtol=1.0e-14, atol=1.0e-14)


def test_cone_and_sensitivity_are_aliases():
    """CLI 的 ``cone`` 与公共接口的 ``sensitivity`` 应保持同一语义."""
    density = np.linspace(0.2, 0.9, 24).reshape(4, 3, 2)
    sensitivity = -np.arange(1.0, 25.0).reshape(4, 3, 2)
    arguments = (sensitivity, density, 1.5, (1.0, 0.7, 0.4))

    cone = apply_structured_sensitivity_filter(*arguments, kind="cone")
    public = apply_structured_sensitivity_filter(*arguments, kind="sensitivity")

    np.testing.assert_allclose(cone, public, rtol=0.0, atol=0.0)


def test_structured_filter_rejects_shape_mismatch():
    """密度与灵敏度的网格形状不一致时应立即失败."""
    with pytest.raises(ValueError, match="相同的三维形状"):
        apply_structured_sensitivity_filter(
            np.ones((2, 2, 2)),
            np.ones((2, 2, 1)),
            rmin=1.0,
            spacing=(1.0, 1.0, 1.0),
        )


def test_density_filter_preserves_constant_density():
    """边界归一化后的锥形密度过滤应保持常密度场."""
    density = np.full((5, 4, 3), 0.37)
    filtered = apply_structured_density_filter(
        density,
        rmin=1.2,
        spacing=(0.8, 0.5, 0.4),
    )
    np.testing.assert_allclose(filtered, density, rtol=1.0e-14, atol=1.0e-14)


def test_density_filter_adjoint_identity():
    """密度过滤与梯度回传必须满足离散伴随恒等式."""
    rng = np.random.default_rng(2026)
    density = rng.random((5, 4, 3))
    gradient = rng.normal(size=density.shape)
    arguments = (1.2, (0.8, 0.5, 0.4))

    filtered = apply_structured_density_filter(density, *arguments)
    adjoint = apply_structured_density_filter_adjoint(gradient, *arguments)

    assert np.vdot(gradient, filtered) == pytest.approx(
        np.vdot(adjoint, density), rel=1.0e-13, abs=1.0e-13
    )
