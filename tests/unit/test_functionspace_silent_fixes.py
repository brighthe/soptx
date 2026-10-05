"""函数空间中曾静默出错的分支的回归测试.

``TensorFunctionSpace.boundary_interpolate`` 的常数边界值分支曾以
``uh[threshold] = gd`` 赋值, ``threshold=None`` 时写满全部自由度;
``LagrangeFESpace.interpolate`` 的重心坐标分支把相邻单元的值累加, 且多重指标顺序与
``cell_to_dof`` 不一致; ``to_tensor_dof`` 的压缩格式分支固定按 2 分量、自由度优先展开.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.decorator import barycentric
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.functionspace.utils import to_tensor_dof
from soptx.mesh import TriangleMesh


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _tensor_space(shape):
    """2x2 三角形网格上的 P2 向量空间."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
    return TensorFunctionSpace(scalar_space=LagrangeFESpace(mesh, p=2), shape=shape)


@pytest.mark.parametrize("shape", [(-1, 2), (2, -1)], ids=["dof_last", "dof_first"])
def test_constant_boundary_value_only_on_boundary(shape):
    """常数边界值只写到边界自由度上, 内部自由度保持为零."""
    space = _tensor_space(shape)
    uh, is_bd = space.boundary_interpolate(gd=3.0, threshold=None, method="interp")
    values = bm.to_numpy(uh[:])
    mask = bm.to_numpy(is_bd)

    assert mask.any() and (~mask).any()
    np.testing.assert_array_equal(values[mask], 3.0)
    np.testing.assert_array_equal(values[~mask], 0.0)


def test_constant_boundary_value_with_callable_threshold():
    """以函数给出 threshold 时, 常数边界值只写到被选中的边界自由度上."""
    space = _tensor_space((-1, 2))

    def left(points):
        return bm.abs(points[..., 0]) < 1e-12

    uh, is_bd = space.boundary_interpolate(gd=-1.0, threshold=left, method="interp")
    values = bm.to_numpy(uh[:])
    mask = bm.to_numpy(is_bd)

    np.testing.assert_array_equal(values[mask], -1.0)
    np.testing.assert_array_equal(values[~mask], 0.0)
    assert mask.sum() < bm.to_numpy(space.is_boundary_dof(method="interp")).sum()


def test_barycentric_interpolate_is_rejected():
    """重心坐标函数的插值结果不对, 应明确报错而不是返回错误的插值."""
    space = LagrangeFESpace(TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2), p=2)

    @barycentric
    def f(bcs):
        return bcs[..., 0]

    with pytest.raises(NotImplementedError, match="直角坐标"):
        space.interpolate(f)


def _ragged_cell_to_dof():
    """两个单元、局部自由度数分别为 3 与 2 的压缩格式映射, 标量自由度共 4 个."""
    cell2dof = bm.tensor([0, 1, 2, 2, 3], dtype=bm.int64)
    location = bm.tensor([0, 3, 5], dtype=bm.int64)
    return cell2dof, location


def test_ragged_to_tensor_dof_two_components_unchanged():
    """压缩格式在 2 分量、自由度优先时的展开结果保持不变."""
    cell2dof, location = _ragged_cell_to_dof()
    tensor_dof, tensor_loc = to_tensor_dof((cell2dof, location), 2, 4, True)
    np.testing.assert_array_equal(bm.to_numpy(tensor_dof), [0, 1, 2, 4, 5, 6, 2, 3, 6, 7])
    np.testing.assert_array_equal(bm.to_numpy(tensor_loc), [0, 6, 10])


@pytest.mark.parametrize("dof_numel, dof_priority", [(3, True), (2, False)])
def test_ragged_to_tensor_dof_rejects_unsupported(dof_numel, dof_priority):
    """压缩格式不支持的分量数或排列方式应明确报错, 不再静默按 2 分量展开."""
    cell2dof, location = _ragged_cell_to_dof()
    with pytest.raises(NotImplementedError):
        to_tensor_dof((cell2dof, location), dof_numel, 4, dof_priority)
