"""物理载荷公共 Protocol 的结构化契约测试."""

from __future__ import annotations

import numpy as np

from soptx.protocols import (
    BodyForce,
    BoundaryTraction,
    LineTraction,
    Load,
    PointForce,
)


class _PointForce:
    kind = "point_force"
    dimension = 3
    support_dimension = 0
    point = (1.0, 0.5, 0.25)

    def force(self, *, time=None):
        return np.array([0.0, -1.0, 0.0])


class _LineTraction:
    kind = "line_traction"
    dimension = 3
    support_dimension = 1

    def is_load_line(self, points):
        return np.ones(points.shape[:-1], dtype=bool)

    def traction(self, points, *, tangents=None, time=None):
        return np.zeros_like(points)


class _BoundaryTraction:
    kind = "boundary_traction"
    dimension = 3
    support_dimension = 2

    def is_load_boundary(self, points):
        return np.ones(points.shape[:-1], dtype=bool)

    def traction(self, points, *, normals=None, time=None):
        return np.zeros_like(points)


class _BodyForce:
    kind = "body_force"
    dimension = 3
    support_dimension = 3

    def body_force(self, points, *, time=None):
        return np.zeros_like(points)


def test_load_protocols_accept_structural_implementations() -> None:
    """无需继承即可按成员结构满足对应载荷契约."""
    implementations = (
        (_PointForce(), PointForce),
        (_LineTraction(), LineTraction),
        (_BoundaryTraction(), BoundaryTraction),
        (_BodyForce(), BodyForce),
    )

    for load, protocol in implementations:
        assert isinstance(load, Load)
        assert isinstance(load, protocol)


def test_specific_load_protocols_remain_distinct() -> None:
    """不同物理载荷契约不能仅凭公共字段相互冒充."""
    point_force = _PointForce()

    assert not isinstance(point_force, LineTraction)
    assert not isinstance(point_force, BoundaryTraction)
    assert not isinstance(point_force, BodyForce)
