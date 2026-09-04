"""LFEM 节点型载荷投影测试."""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem.load_projection import project_nodal_loads
from soptx.problems.loads import LineTractionLoad, PointForceLoad


def test_point_force_is_written_to_the_unique_interpolation_node() -> None:
    """集中力保持合力并写入唯一匹配节点."""
    bm.set_backend("numpy")
    points = bm.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    load = PointForceLoad(point=(1.0, 0.0), vector=(3.0, -4.0))

    vector = project_nodal_loads((load,), points, dimension=2)

    np.testing.assert_allclose(
        bm.to_numpy(vector),
        np.array((0.0, 0.0, 3.0, -4.0, 0.0, 0.0)),
    )


def test_constant_line_traction_forms_consistent_nodal_forces() -> None:
    """常值线载荷按相邻线段积分，端点与内部节点权重正确."""
    bm.set_backend("numpy")
    points = bm.asarray(((0.0, 0.0), (0.0, 2.0), (0.0, 4.0)))
    load = LineTractionLoad(
        dimension=2,
        marker=lambda p: np.isclose(p[..., 0], 0.0),
        value=lambda p: np.stack(
            (np.zeros(p.shape[:-1]), -np.ones(p.shape[:-1])),
            axis=-1,
        ),
    )

    vector = project_nodal_loads((load,), points, dimension=2)

    np.testing.assert_allclose(
        bm.to_numpy(vector),
        np.array((0.0, -1.0, 0.0, -2.0, 0.0, -1.0)),
    )
    assert bm.to_numpy(vector).reshape(-1, 2)[:, 1].sum() == pytest.approx(-4.0)


def test_point_force_outside_interpolation_nodes_is_rejected() -> None:
    """LFEM 不再把任意物理点静默吸附到最近节点."""
    bm.set_backend("numpy")
    points = bm.asarray(((0.0, 0.0), (1.0, 0.0)))
    load = PointForceLoad(point=(0.5, 0.0), vector=(0.0, -1.0))

    with pytest.raises(ValueError, match="精确落在唯一插值节点"):
        project_nodal_loads((load,), points, dimension=2)
