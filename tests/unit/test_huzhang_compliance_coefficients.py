"""Hu--Zhang 二维柔度系数的平面假设回归测试."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soptx.fem import HuZhangMFEMAnalyzer


def _analyzer_stub(hypothesis: str):
    """构造只用于柔度系数计算的最小分析器对象."""
    analyzer = object.__new__(HuZhangMFEMAnalyzer)
    analyzer._GD = 2
    analyzer._material = SimpleNamespace(hypothesis=hypothesis)
    return analyzer


def test_plane_stress_uses_plane_stress_compliance_coefficient() -> None:
    """平面应力必须使用 lambda1=nu/E, 不能退化为平面应变公式."""
    youngs_modulus = 30.0
    poisson_ratio = 0.4
    plane_stress = _analyzer_stub("plane_stress")
    plane_strain = _analyzer_stub("plane_strain")

    stress_lambda0, stress_lambda1 = plane_stress._compute_compliance_coefficients(
        youngs_modulus, poisson_ratio
    )
    _, strain_lambda1 = plane_strain._compute_compliance_coefficients(
        youngs_modulus, poisson_ratio
    )

    assert stress_lambda0 == pytest.approx((1.0 + poisson_ratio) / youngs_modulus)
    assert stress_lambda1 == pytest.approx(poisson_ratio / youngs_modulus)
    assert strain_lambda1 == pytest.approx(
        poisson_ratio * (1.0 + poisson_ratio) / youngs_modulus
    )
