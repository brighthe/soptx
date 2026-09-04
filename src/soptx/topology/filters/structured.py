# -*- coding: utf-8 -*-
"""规则三维网格的可扩展灵敏度与密度过滤器."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.ndimage import convolve, uniform_filter


Spacing3d = tuple[float, float, float]
FilterKind = Literal["box", "cone", "sensitivity"]


def _validate_inputs(
    sensitivity: np.ndarray,
    density: np.ndarray,
    rmin: float,
    spacing: Spacing3d,
) -> tuple[np.ndarray, np.ndarray, Spacing3d]:
    """校验并规范化结构化过滤器输入."""
    dc = np.asarray(sensitivity, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    if dc.ndim != 3 or rho.shape != dc.shape:
        raise ValueError(
            "sensitivity 与 density 必须具有相同的三维形状; "
            f"当前为 {dc.shape} 与 {rho.shape}."
        )
    if rmin <= 0.0:
        raise ValueError(f"过滤半径必须为正数; 当前 rmin={rmin}.")
    normalized = tuple(float(value) for value in spacing)
    if len(normalized) != 3 or any(value <= 0.0 for value in normalized):
        raise ValueError(f"spacing 必须包含三个正数; 当前为 {spacing}.")
    return dc, rho, normalized


@lru_cache(maxsize=32)
def build_structured_cone_kernel(
    rmin: float,
    spacing: Spacing3d,
) -> np.ndarray:
    """构造按物理距离度量的三维锥形权重核.

    参数:
        rmin: 物理长度单位下的过滤半径.
        spacing: 三个方向的单元边长 ``(hx, hy, hz)``.

    返回:
        权重 ``max(0, rmin - distance)`` 构成的三维卷积核.
    """
    normalized = tuple(float(value) for value in spacing)
    if rmin <= 0.0 or len(normalized) != 3 or any(
        value <= 0.0 for value in normalized
    ):
        raise ValueError(f"rmin 与 spacing 必须为正; 当前为 {rmin}, {spacing}.")
    radii = tuple(int(np.floor(rmin / value)) for value in normalized)
    offsets = [
        np.arange(-radius, radius + 1, dtype=np.float64) * step
        for radius, step in zip(radii, normalized)
    ]
    dx, dy, dz = np.meshgrid(*offsets, indexing="ij")
    distance = np.sqrt(dx * dx + dy * dy + dz * dz)
    kernel = np.clip(rmin - distance, 0.0, None)
    kernel.setflags(write=False)
    return kernel


@lru_cache(maxsize=32)
def _structured_weight_sum(
    shape: tuple[int, int, int],
    rmin: float,
    spacing: Spacing3d,
) -> np.ndarray:
    """缓存边界截断后的锥形权重和."""
    kernel = build_structured_cone_kernel(rmin, spacing)
    weight_sum = convolve(
        np.ones(shape, dtype=np.float64),
        kernel,
        mode="constant",
        cval=0.0,
    )
    weight_sum.setflags(write=False)
    return weight_sum


def apply_structured_sensitivity_filter(
    sensitivity: np.ndarray,
    density: np.ndarray,
    rmin: float,
    spacing: Spacing3d,
    kind: FilterKind = "cone",
    gamma: float = 1.0e-3,
) -> np.ndarray:
    """过滤规则三维网格上的柔顺度灵敏度.

    ``cone`` 与 ``sensitivity`` 均执行 Sigmund 密度加权锥形过滤;
    ``box`` 保留原平台的均匀平滑对照. 锥形权重使用真实物理距离, 因此允许
    ``hx``, ``hy`` 与 ``hz`` 不相等.
    """
    dc, rho, normalized = _validate_inputs(sensitivity, density, rmin, spacing)
    if kind == "box":
        sizes = tuple(
            max(3, 2 * int(round(rmin / step)) + 1)
            for step in normalized
        )
        return uniform_filter(dc, size=sizes, mode="nearest")
    if kind not in ("cone", "sensitivity"):
        raise ValueError(f"未知过滤类型: {kind}.")
    if gamma <= 0.0:
        raise ValueError(f"gamma 必须为正数; 当前为 {gamma}.")
    kernel = build_structured_cone_kernel(rmin, normalized)
    numerator = convolve(rho * dc, kernel, mode="constant", cval=0.0)
    denominator = np.maximum(gamma, rho) * _structured_weight_sum(
        dc.shape, rmin, normalized
    )
    return numerator / denominator


def _validate_density_input(
    density: np.ndarray,
    rmin: float,
    spacing: Spacing3d,
) -> tuple[np.ndarray, Spacing3d]:
    """校验并规范化结构化密度过滤器输入."""
    rho = np.asarray(density, dtype=np.float64)
    if rho.ndim != 3:
        raise ValueError(f"density 必须为三维数组; 当前形状为 {rho.shape}.")
    if rmin <= 0.0:
        raise ValueError(f"过滤半径必须为正数; 当前 rmin={rmin}.")
    normalized = tuple(float(value) for value in spacing)
    if len(normalized) != 3 or any(value <= 0.0 for value in normalized):
        raise ValueError(f"spacing 必须包含三个正数; 当前为 {spacing}.")
    return rho, normalized


def apply_structured_density_filter(
    density: np.ndarray,
    rmin: float,
    spacing: Spacing3d,
) -> np.ndarray:
    """计算归一化锥形密度过滤 ``rho_phys = D^{-1} H rho``."""
    rho, normalized = _validate_density_input(density, rmin, spacing)
    kernel = build_structured_cone_kernel(rmin, normalized)
    numerator = convolve(rho, kernel, mode="constant", cval=0.0)
    return numerator / _structured_weight_sum(rho.shape, rmin, normalized)


def apply_structured_density_filter_adjoint(
    gradient: np.ndarray,
    rmin: float,
    spacing: Spacing3d,
) -> np.ndarray:
    """应用密度过滤 Jacobian 的转置 ``H^T D^{-1}`` 回传梯度."""
    grad, normalized = _validate_density_input(gradient, rmin, spacing)
    scaled = grad / _structured_weight_sum(grad.shape, rmin, normalized)
    kernel = build_structured_cone_kernel(rmin, normalized)
    return convolve(scaled, kernel, mode="constant", cval=0.0)
