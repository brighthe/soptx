"""角点线性插值接口迹空间."""

from __future__ import annotations

from typing import Any

from .base import TraceBasis


class LinearCornerTraceBasis(TraceBasis):
    """Huang 2023 式 (16) 使用的角点线性接口迹基.

    当前矩阵构造仍由 ``SubstructurePrototype.linear_boundary_matrix`` 提供,
    本类先统一迹空间代数契约. 后续可在不改变调用方的情况下把几何构造职责
    从原型迁入本模块.
    """

    name = "linear_corner"

    @classmethod
    def from_prototype(cls, prototype: Any) -> "LinearCornerTraceBasis":
        """由子结构原型已有的式 (16) 插值矩阵构造迹基."""
        return cls(prototype.linear_boundary_matrix)
