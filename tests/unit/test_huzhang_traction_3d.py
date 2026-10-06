"""三维胡张空间牵引强施加的回归测试.

``boundary_interpolate`` 按格点上的标架张量 :math:`S` 写值: :math:`Sn = 0` 的不受约束,
:math:`S = \\mathrm{sym}(n \\otimes w)` 的取 :math:`(t \\cdot w) / \\|S\\|_F^2`. 若 ``gd`` 是某个
:math:`\\sigma \\in P_p(\\mathbb S)` 的应力, 写入的值应与 :math:`\\sigma` 在全空间中的系数
完全一致, 包括立方体棱与角点上被多个面重复写入的自由度.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.functionspace import HuZhangFESpace
from soptx.mesh import TetrahedronMesh

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


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
@pytest.mark.parametrize("tangential", [False, True], ids=["full", "tangential"])
def test_written_values_match_global_coefficients(p, tangential):
    """以真实应力场为 gd, 写入的边界自由度值等于该场的全局系数."""
    rng = np.random.default_rng(p)
    space = HuZhangFESpace(TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1), p=p)
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


def test_non_axis_aligned_boundary_is_rejected():
    """旋转后的立方体边界与顶点的笛卡尔标架不对齐, 明确报错而不写入错误的值."""
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1)
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    node = bm.to_numpy(mesh.entity("node")) @ R.T
    rotated = TetrahedronMesh(bm.tensor(node), mesh.entity("cell"))
    space = HuZhangFESpace(rotated, p=4)
    with pytest.raises(NotImplementedError, match="标架"):
        space.boundary_interpolate(bm.tensor([1.0, 0.0, 0.0]))
