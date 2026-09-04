"""Hu--Zhang 对结构化物理载荷对象的解释测试."""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem import HuZhangMFEMAnalyzer
from soptx.problems.loads import BoundaryTractionLoad, PointForceLoad


class _LoadProblem:
    def __init__(self, *loads) -> None:
        self._loads = loads

    def loads(self):
        return self._loads


def _analyzer_with(*loads) -> HuZhangMFEMAnalyzer:
    analyzer = object.__new__(HuZhangMFEMAnalyzer)
    analyzer._pde = _LoadProblem(*loads)
    analyzer._GD = 2
    return analyzer


def test_boundary_traction_is_masked_by_its_physical_support() -> None:
    bm.set_backend("numpy")
    load = BoundaryTractionLoad(
        dimension=2,
        marker=lambda points: np.isclose(points[..., 0], 0.0),
        value=lambda points: np.broadcast_to((1.0, -2.0), points.shape),
    )
    analyzer = _analyzer_with(load)
    points = bm.asarray(((0.0, 0.0), (1.0, 0.0)))

    value = analyzer._prescribed_traction(points)

    np.testing.assert_allclose(
        bm.to_numpy(value),
        np.array(((1.0, -2.0), (0.0, 0.0))),
    )


def test_point_force_requires_trace_regularization_before_huzhang() -> None:
    bm.set_backend("numpy")
    analyzer = _analyzer_with(
        PointForceLoad(point=(1.0, 0.0), vector=(0.0, -1.0))
    )

    with pytest.raises(TypeError, match="投影为 BoundaryTraction"):
        analyzer._validated_loads()
