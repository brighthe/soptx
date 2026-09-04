"""Problem 直接提供物理载荷对象的契约测试."""

from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.problems import (
    BearingDevice2d,
    CantileverRightBottomEdge3d,
    DivergenceFreePolynomialElasticity3D,
    FullMBBBeam3d,
)
from soptx.protocols import (
    BodyForce,
    BoundaryTraction,
    LineTraction,
    LoadProvider,
    PointForce,
)


def test_representative_problems_provide_physical_load_objects() -> None:
    """四类代表 Problem 分别返回对应物理载荷对象."""
    problems_and_protocols = (
        (FullMBBBeam3d(), PointForce),
        (CantileverRightBottomEdge3d(), LineTraction),
        (BearingDevice2d(), BoundaryTraction),
        (DivergenceFreePolynomialElasticity3D(), BodyForce),
    )

    for problem, protocol in problems_and_protocols:
        assert isinstance(problem, LoadProvider)
        loads = tuple(problem.loads())
        assert loads
        assert any(isinstance(load, protocol) for load in loads)


def test_full_mbb_point_force_keeps_the_physical_resultant() -> None:
    """完整三维 MBB 梁载荷对象保存顶面中心点和总力."""
    problem = FullMBBBeam3d(P=-2.5)
    load = problem.loads()[0]

    assert isinstance(load, PointForce)
    assert tuple(load.point) == (60.0, 20.0, 10.0)
    assert tuple(load.force()) == (0.0, -2.5, 0.0)


def test_cantilever_edge_load_is_per_unit_length() -> None:
    """三维悬臂梁棱边载荷强度乘线长等于给定总力."""
    bm.set_backend("numpy")
    problem = CantileverRightBottomEdge3d(P=-4.0)
    load = problem.loads()[0]
    points = bm.array([[60.0, 0.0, 0.0], [60.0, 0.0, 2.0]])

    assert isinstance(load, LineTraction)
    values = bm.to_numpy(load.traction(points))
    np.testing.assert_allclose(values[:, 1], np.array([-1.0, -1.0]))
    line_length = problem.domain[5] - problem.domain[4]
    assert values[0, 1] * line_length == problem.P
