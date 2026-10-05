"""插值点在 numpy 与 pytorch 后端下的一致性.

``interpolation_points`` 用整数 ``multi_index`` 归一化出重心权重; torch 下整数相除
得到 float32, 曾在 ``einsum`` 处与 float64 坐标类型不符, 使 pytorch 后端下所有
p >= 2 的 Lagrange 空间不可用.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.functionspace import LagrangeFESpace
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh

MESHES = [
    (TriangleMesh, [0, 1, 0, 1], (2, 2)),
    (QuadrangleMesh, [0, 1, 0, 1], (2, 2)),
    (TetrahedronMesh, [0, 1, 0, 1, 0, 1], (2, 2, 2)),
    (HexahedronMesh, [0, 1, 0, 1, 0, 1], (2, 2, 2)),
]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _interpolation_points(backend: str, mesh_type, box, n, p: int):
    """在指定后端下建网格与 p 次 Lagrange 空间, 返回网格与空间两条入口的插值点."""
    bm.set_backend(backend)
    mesh = mesh_type.from_box(box, *n)
    mesh_ips = bm.to_numpy(mesh.interpolation_points(p))
    space_ips = bm.to_numpy(LagrangeFESpace(mesh, p=p).interpolation_points())
    return mesh_ips, space_ips


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("mesh_type,box,n", MESHES, ids=[m[0].__name__ for m in MESHES])
def test_interpolation_points_match_across_backends(mesh_type, box, n, p):
    """pytorch 后端的插值点与 numpy 一致, 且保持 float64."""
    ref_mesh, ref_space = _interpolation_points("numpy", mesh_type, box, n, p)
    torch_mesh, torch_space = _interpolation_points("pytorch", mesh_type, box, n, p)

    assert torch_mesh.dtype == np.float64
    np.testing.assert_allclose(torch_mesh, ref_mesh, rtol=0, atol=1e-15)
    np.testing.assert_allclose(torch_space, ref_space, rtol=0, atol=1e-15)
