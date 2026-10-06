"""三维胡张空间牵引强施加的回归测试.

``boundary_interpolate`` 按格点上的标架张量 :math:`S` 写值: :math:`Sn = 0` 的不受约束,
:math:`S = \\mathrm{sym}(n \\otimes w)` 的取 :math:`(t \\cdot w) / \\|S\\|_F^2`. 若 ``gd`` 是某个
:math:`\\sigma \\in P_p(\\mathbb S)` 的应力, 写入的值应与 :math:`\\sigma` 在全空间中的系数
完全一致, 包括立方体棱与角点上被多个面重复写入的自由度. 旋转与剪切的立方体检验非坐标
对齐边界: 标架按边界法向对齐, 折棱与角点上联合多个面的牵引求值.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.functionspace import HuZhangFESpace
from soptx.mesh import TetrahedronMesh

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
THETA = 0.3
ROTATION = np.array([[np.cos(THETA), -np.sin(THETA), 0.0], [np.sin(THETA), np.cos(THETA), 0.0], [0.0, 0.0, 1.0]])
SHEAR = np.array([[1.0, 0.4, 0.2], [0.0, 1.0, -0.3], [0.0, 0.0, 1.0]])   # 折棱不再是 90 度
MAPS = {"cube": np.eye(3), "rotated": ROTATION, "sheared": SHEAR, "rotated-sheared": ROTATION @ SHEAR}


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _mapped_cube(name: str, nx: int = 1):
    """单位立方体四面体网格经线性映射 MAPS[name] 后的网格."""
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=nx, ny=nx, nz=nx)
    if name == "cube":
        return mesh
    node = bm.to_numpy(mesh.entity("node")) @ MAPS[name].T
    return TetrahedronMesh(bm.tensor(node), mesh.entity("cell"))


def _random_stress(p: int, rng):
    """随机 P_p 对称张量场, 返回 Voigt [xx, xy, xz, yy, yz, zz] 值的函数."""
    exps = [e for e in np.ndindex(p + 1, p + 1, p + 1) if sum(e) <= p]
    coef = rng.standard_normal((6, len(exps)))

    def value(x):
        mono = np.stack([np.prod(x ** np.array(e), axis=-1) for e in exps], axis=-1)
        return mono @ coef.T

    return value


def _fitted_coefficients(space, value, rng):
    """全局最小二乘求 value 在 space 中的系数 (P_p(S) 被精确复现)."""
    mesh = space.mesh
    ldof, gdof = space.number_of_local_dofs(), space.number_of_global_dofs()
    bcs = bm.tensor(rng.dirichlet(np.ones(4), size=2 * ldof))
    phi = bm.to_numpy(space.basis(bcs))
    NC, NQ = phi.shape[:2]
    c2d = bm.to_numpy(space.cell_to_dof())
    A = np.zeros((NC, NQ, 6, gdof))
    for c in range(NC):
        np.add.at(A[c], (slice(None), slice(None), c2d[c]), phi[c].transpose(0, 2, 1))
    b = value(bm.to_numpy(mesh.bc_to_point(bcs))).reshape(-1)
    coef = np.linalg.lstsq(A.reshape(-1, gdof), b, rcond=None)[0]
    assert np.linalg.norm(A.reshape(-1, gdof) @ coef - b) < 1e-11 * np.linalg.norm(b)
    return coef


@pytest.mark.parametrize("p", [2, 4])
@pytest.mark.parametrize(
    "mesh_name, tangential",
    [("cube", False), ("cube", True), ("rotated", False), ("rotated", True), ("sheared", False),
     ("rotated-sheared", False)],
    ids=["cube-full", "cube-tangential", "rotated-full", "rotated-tangential", "sheared-full",
         "rotated-sheared-full"],
)
def test_written_values_match_global_coefficients(mesh_name, tangential, p):
    """以真实应力场为 gd, 写入的边界自由度值等于该场的全局系数.

    剪切立方体的折棱不是 90 度, 只检验完整牵引; 其上的切向牵引见下一个测试.
    """
    rng = np.random.default_rng(p)
    space = HuZhangFESpace(_mapped_cube(mesh_name), p=p)
    value = _random_stress(p, rng)
    coef = _fitted_coefficients(space, value, rng)

    def gd(points):
        return bm.tensor(value(bm.to_numpy(points)))

    if tangential:
        uh, written = space.set_tangential_traction_bc(gd)
    else:
        uh, written = space.boundary_interpolate(gd)
    written = bm.to_numpy(written)
    assert written.any() and (~written).any()
    np.testing.assert_allclose(bm.to_numpy(uh)[written], coef[written], rtol=1e-9, atol=1e-9 * np.abs(coef).max())


def test_tangential_writes_are_a_subset():
    """切向版本只写入完整版本的一部分自由度 (法向-法向分量保持自由)."""
    space = HuZhangFESpace(TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1), p=2)
    _, full = space.boundary_interpolate(bm.tensor([1.0, 2.0, 3.0]))
    _, tangential = space.set_tangential_traction_bc(bm.tensor([1.0, 2.0, 3.0]))
    full, tangential = bm.to_numpy(full), bm.to_numpy(tangential)
    assert tangential.sum() < full.sum()
    assert not (tangential & ~full).any()


def test_tangential_on_oblique_crease_is_rejected():
    """非 90 度折棱上切向约束与任何单位正交标架都不对齐, 明确报错而不写入错误的值."""
    space = HuZhangFESpace(_mapped_cube("sheared"), p=2)
    with pytest.raises(NotImplementedError, match="约束子空间"):
        space.set_tangential_traction_bc(bm.tensor([1.0, 0.0, 0.0]))


def test_axis_aligned_vertex_frames_stay_cartesian():
    """坐标对齐的网格上顶点保留笛卡尔标架; 只对齐部分牵引面时其余顶点也不变."""
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    on_x1 = np.abs(bm.to_numpy(mesh.entity_barycenter("face"))[:, 0] - 1.0) < 1e-12
    for flag in (None, bm.tensor(on_x1)):
        nframe = bm.to_numpy(HuZhangFESpace(mesh, p=2, traction_face=flag).dof_frame()[0])
        np.testing.assert_array_equal(nframe, np.broadcast_to(np.eye(3), nframe.shape))


@pytest.mark.parametrize("mesh_name", ["rotated", "sheared"])
def test_vertex_frames_contain_traction_normals(mesh_name):
    """非坐标对齐边界上, 邻接法向不全坐标对齐的顶点标架以某个邻接面法向为首向量, 其余保留笛卡尔基."""
    mesh = _mapped_cube(mesh_name, nx=2)
    nframe = bm.to_numpy(HuZhangFESpace(mesh, p=2).dof_frame()[0])
    np.testing.assert_allclose(np.einsum("nij,nkj->nik", nframe, nframe),
                               np.broadcast_to(np.eye(3), nframe.shape), atol=1e-13)
    face = bm.to_numpy(mesh.entity("face"))
    normal = bm.to_numpy(mesh.face_unit_normal())
    bdface = np.nonzero(bm.to_numpy(mesh.boundary_face_flag()))[0]
    for v in np.unique(face[bdface]):
        adjacent = normal[[f for f in bdface if v in face[f]]]
        if np.all(np.abs(adjacent).max(axis=-1) > 1 - 1e-10):
            np.testing.assert_array_equal(nframe[v], np.eye(3))
        else:
            assert np.isclose(np.abs(adjacent @ nframe[v, 0]).max(), 1.0)
