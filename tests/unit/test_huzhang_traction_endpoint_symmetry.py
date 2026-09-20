"""Hu--Zhang 强施加牵引在载荷区端点上的对称性测试.

顶点自由度被相邻两条边界边各写一次. 历史实现用 ``uh[e2d] = val`` 直接赋值,
是"后写者胜", 胜者由全局边编号决定; 牵引数据在端点两侧不连续时, 端点值被整个
判给其中一侧, 离散载荷因而失去镜像对称并带上净力矩. 本文件锁定改正后的行为:
重复写入取平均.
"""

from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem.spaces import HuZhangFESpace


_TRACTION = -2.0
_Y_LOW, _Y_HIGH = 0.25, 0.75
_Y_MID = 0.5 * (_Y_LOW + _Y_HIGH)


def _patch_traction(points):
    """按边重心整边选取的贴片牵引, 载荷区为 x=1 上 y in [0.25, 0.75]."""
    centers = bm.mean(points, axis=-2)
    marked = (
        (bm.abs(centers[..., 0] - 1.0) < 1e-12)
        & (centers[..., 1] > _Y_LOW)
        & (centers[..., 1] < _Y_HIGH)
    )
    value = bm.zeros(points.shape, **bm.context(points))
    value = bm.set_at(value, (..., 1), _TRACTION)
    return bm.where(marked[..., None, None], value, bm.zeros_like(value))


def _imposed_on_right_boundary(p: int):
    """返回 (节点 y 坐标, 该节点两个迹自由度的强加值) , 只含 x=1 上的节点."""
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=4, ny=4)
    space = HuZhangFESpace(mesh, p=p)

    threshold = mesh.boundary_edge_flag()
    uh, _ = space.set_dirichlet_bc(_patch_traction, threshold=threshold)
    values = np.asarray(bm.to_numpy(uh[:]))

    node = np.asarray(bm.to_numpy(mesh.entity("node")))
    node2dof = np.asarray(bm.to_numpy(space.dof.node_to_internal_dof()[0]))
    on_right = np.flatnonzero(np.abs(node[:, 0] - 1.0) < 1e-12)

    return node[on_right, 1], values[node2dof[on_right, :2]]


def test_patch_endpoints_receive_mirror_equal_traces() -> None:
    """载荷区两端点拿到等量的迹值, 而不是一端满载一端为零."""
    bm.set_backend("numpy")
    for p in (1, 2, 3):
        y, traces = _imposed_on_right_boundary(p)
        low = np.abs(traces[np.argmin(np.abs(y - _Y_LOW))])
        high = np.abs(traces[np.argmin(np.abs(y - _Y_HIGH))])

        np.testing.assert_allclose(low, high, atol=1e-12, err_msg=f"p={p}")
        # 反证: 端点确实处在数据间断上, 平均值是满载的一半而非 0 或满载.
        assert np.max(low) > 0.0, f"p={p}"
        assert np.max(low) < 2.0 * abs(_TRACTION), f"p={p}"


def test_interior_patch_nodes_keep_the_full_traction() -> None:
    """载荷区内部节点两侧数据一致, 取平均是恒等操作."""
    bm.set_backend("numpy")
    for p in (1, 2, 3):
        y, traces = _imposed_on_right_boundary(p)
        inside = (y > _Y_LOW + 1e-12) & (y < _Y_HIGH - 1e-12)
        assert inside.any(), f"p={p}"

        magnitude = np.max(np.abs(traces[inside]), axis=1)
        np.testing.assert_allclose(
            magnitude, 2.0 * abs(_TRACTION), atol=1e-12, err_msg=f"p={p}"
        )


def test_average_of_repeated_writes_is_order_independent() -> None:
    """同一自由度被写多次时取平均, 结果与写入次序无关."""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    space = HuZhangFESpace(mesh, p=1)

    uh = bm.zeros((space.number_of_global_dofs(),), dtype=bm.float64)
    dof_index = bm.asarray([[0, 1], [1, 2]])
    values = bm.asarray([[3.0, 1.0], [5.0, 7.0]])

    result = np.asarray(bm.to_numpy(
        space._average_boundary_writes(uh, dof_index, values)))

    # dof 1 被写两次 (1.0 与 5.0), 取均值 3.0; 单次写入的 dof 保持原值.
    np.testing.assert_allclose(result[:3], np.array([3.0, 3.0, 7.0]))
    np.testing.assert_allclose(result[3:], 0.0)
