"""胡张空间工厂与自由度访问器的回归测试.

工厂曾把 ``use_relaxation`` 传给不接受该参数的 ``HuZhangFESpace3d``, 任何三维网格都抛
TypeError; 二维 ``edge_to_dof`` / ``face_to_dof`` 调用即报错; 三维的几个访问器是空函数体,
静默返回 None; 二维 ``basis`` 对布尔掩码算错单元数; 三维 ``basis`` 等忽略 ``index``.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.functionspace import HuZhangFESpace, HuZhangFESpace3d
from soptx.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from soptx.mesh import TetrahedronMesh, TriangleMesh


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _space_2d(p: int = 2) -> HuZhangFESpace2d:
    """2x2 三角形网格上的 p 次胡张空间."""
    return HuZhangFESpace(TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2), p=p)


def _space_3d(p: int = 3) -> HuZhangFESpace3d:
    """1x1x1 四面体网格上的 p 次胡张空间."""
    return HuZhangFESpace(TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1), p=p)


def test_factory_dispatches_3d():
    """三维网格经工厂得到 HuZhangFESpace3d."""
    space = _space_3d()
    assert isinstance(space, HuZhangFESpace3d)
    assert space.cell_to_dof().shape == (space.mesh.number_of_cells(), space.number_of_local_dofs())


def test_factory_rejects_relaxation_in_3d():
    """三维没有角点松弛, 要求松弛时明确报错."""
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1)
    with pytest.raises(ValueError, match="use_relaxation"):
        HuZhangFESpace(mesh, p=3, use_relaxation=True)


def test_2d_edge_and_face_to_dof():
    """二维 edge_to_dof 按 index 选取边, face_to_dof 与之相同."""
    space = _space_2d()
    full = bm.to_numpy(space.dof.edge_to_dof())
    np.testing.assert_array_equal(bm.to_numpy(space.edge_to_dof()), full)
    np.testing.assert_array_equal(bm.to_numpy(space.face_to_dof()), full)

    index = bm.tensor([3, 0, 5], dtype=bm.int64)
    np.testing.assert_array_equal(bm.to_numpy(space.edge_to_dof(index=index)), full[[3, 0, 5]])
    np.testing.assert_array_equal(bm.to_numpy(space.face_to_dof(index=index)), full[[3, 0, 5]])


@pytest.mark.parametrize("name", ["interpolation_points", "is_boundary_dof"])
def test_2d_unimplemented_accessors_raise(name):
    """二维未实现的访问器明确抛 NotImplementedError."""
    with pytest.raises(NotImplementedError):
        getattr(_space_2d(), name)()


@pytest.mark.parametrize("name", ["interpolation_points", "edge_to_dof", "face_to_dof", "is_boundary_dof"])
def test_3d_unimplemented_accessors_raise(name):
    """三维未实现的访问器明确抛 NotImplementedError, 不再静默返回 None."""
    with pytest.raises(NotImplementedError):
        getattr(_space_3d(), name)()


def test_2d_basis_with_boolean_mask():
    """二维 basis 接受布尔掩码, 结果等于全部单元的结果按掩码取行."""
    space = _space_2d()
    bcs = bm.tensor([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2]], dtype=bm.float64)
    full = bm.to_numpy(space.basis(bcs))

    mask = np.zeros(space.mesh.number_of_cells(), dtype=bool)
    mask[[1, 4, 6]] = True
    masked = bm.to_numpy(space.basis(bcs, index=bm.tensor(mask)))
    np.testing.assert_allclose(masked, full[mask], rtol=0, atol=1e-14)

    index = bm.tensor([6, 1], dtype=bm.int64)
    np.testing.assert_allclose(bm.to_numpy(space.basis(bcs, index=index)), full[[6, 1]], rtol=0, atol=1e-14)


def test_3d_cell_subset_is_rejected():
    """三维 basis / value / div_value 尚不支持单元子集, 传 index 时明确报错."""
    space = _space_3d()
    bcs = bm.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=bm.float64)
    uh = bm.zeros(space.number_of_global_dofs(), dtype=bm.float64)
    index = bm.tensor([0, 2], dtype=bm.int64)

    with pytest.raises(NotImplementedError):
        space.basis(bcs, index=index)
    with pytest.raises(NotImplementedError):
        space.value(uh, bcs, index=index)
    with pytest.raises(NotImplementedError):
        space.div_value(uh, bcs, index=index)
    assert space.basis(bcs).shape[:3] == (space.mesh.number_of_cells(), 1, space.number_of_local_dofs())
