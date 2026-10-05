"""完整接口迹空间."""

from __future__ import annotations

from typing import Any

from fealpy.backend import backend_manager as bm

from .base import TraceBasis


class FullTraceBasis(TraceBasis):
    """保留全部子结构接口自由度的恒等迹基 ``T = I``."""

    name = "full_trace"

    def __init__(self, n_boundary_dofs: int, *, dtype: Any = None) -> None:
        if n_boundary_dofs <= 0:
            raise ValueError(
                f"n_boundary_dofs 必须为正整数; 当前为 {n_boundary_dofs}."
            )
        if dtype is None:
            dtype = bm.float64
        super().__init__(bm.eye(n_boundary_dofs, dtype=dtype))

    @classmethod
    def from_prototype(cls, prototype: Any) -> "FullTraceBasis":
        """由子结构原型的接口自由度数构造完整迹基."""
        return cls(int(prototype.n_b))

    # ``T = I`` 时三个映射都是恒等, 只校验形状, 不执行与单位阵的乘法.

    def project_stiffness(self, stiffness: Any) -> Any:
        """``K_q = K_s``; 完整接口下投影为恒等."""
        value = bm.asarray(stiffness)
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError(
                "stiffness 必须具有形状 (..., n_boundary, n_boundary)."
            )
        self._check_boundary_axis(value, "stiffness")
        return value

    def reduce_recovery(self, recovery: Any) -> Any:
        """``N T = N``; 完整接口下恢复矩阵不变."""
        value = bm.asarray(recovery)
        if value.ndim < 2:
            raise ValueError(
                "recovery 必须具有形状 (..., n_internal, n_boundary)."
            )
        self._check_boundary_axis(value, "recovery")
        return value

    def expand_displacement(self, trace_displacement: Any) -> Any:
        """``u_b = q``; 完整接口下迹位移即边界位移."""
        value = bm.asarray(trace_displacement)
        if value.ndim < 1 or value.shape[-1] != self.n_trace_dofs:
            raise ValueError(
                f"trace_displacement 的末维必须为 {self.n_trace_dofs}; "
                f"当前形状为 {tuple(value.shape)}."
            )
        return value
