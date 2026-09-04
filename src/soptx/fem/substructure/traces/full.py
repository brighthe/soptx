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
