"""Problem 直接提供物理载荷对象的契约测试."""

from __future__ import annotations

import numpy as np

from soptx.backend import backend_manager as bm

from soptx.problems import (
    BearingTraction2d,
    CantileverBottomEdgeLine3d,
    DivergenceFreePolynomialElasticity3D,
    FullMBBBeamPoint2d,
    FullMBBBeamLine3d,
    FullMBBBeamPoint3d,
)
from soptx.protocols import (
    BodyForce,
    BoundaryTraction,
    LineTraction,
    LoadProvider,
    PointForce,
)


def test_representative_problems_provide_physical_load_objects() -> None:
    """代表 Problem 分别返回对应物理载荷对象."""
    problems_and_protocols = (
        (FullMBBBeamPoint2d(), PointForce),
        (FullMBBBeamLine3d(), LineTraction),
        (FullMBBBeamPoint3d(), PointForce),
        (CantileverBottomEdgeLine3d(), LineTraction),
        (BearingTraction2d(), BoundaryTraction),
        (DivergenceFreePolynomialElasticity3D(), BodyForce),
    )

    for problem, protocol in problems_and_protocols:
        assert isinstance(problem, LoadProvider)
        loads = tuple(problem.loads())
        assert loads
        assert any(isinstance(load, protocol) for load in loads)


def test_full_mbb3d_line_load_is_the_extruded_center_force() -> None:
    """完整三维 MBB 梁的载荷为顶面中线线载荷, 强度乘线长等于总力."""
    bm.set_backend("numpy")
    problem = FullMBBBeamLine3d(P=-2.5)
    load = problem.loads()[0]
    points = bm.array([[60.0, 20.0, 0.0], [60.0, 20.0, 7.0], [60.0, 0.0, 7.0], [59.0, 20.0, 7.0]])

    assert isinstance(load, LineTraction)
    np.testing.assert_array_equal(bm.to_numpy(load.is_load_line(points)), [True, True, False, False])
    values = bm.to_numpy(load.traction(points[:2]))
    line_length = problem.domain[5] - problem.domain[4]
    np.testing.assert_allclose(values * line_length, [[0.0, -2.5, 0.0], [0.0, -2.5, 0.0]])


def test_cantilever_edge_load_is_per_unit_length() -> None:
    """三维悬臂梁棱边载荷强度乘线长等于给定总力."""
    bm.set_backend("numpy")
    problem = CantileverBottomEdgeLine3d(P=-4.0)
    load = problem.loads()[0]
    points = bm.array([[60.0, 0.0, 0.0], [60.0, 0.0, 2.0]])

    assert isinstance(load, LineTraction)
    values = bm.to_numpy(load.traction(points))
    np.testing.assert_allclose(values[:, 1], np.array([-1.0, -1.0]))
    line_length = problem.domain[5] - problem.domain[4]
    assert values[0, 1] * line_length == problem.P
