"""胡张应力空间的代数验证: 二维为对照, 三维为被测对象.

三维空间没有使用者, 也没有端到端算例, 这里不经求解直接检查空间本身:

- 张成: 单元上 ``ldof`` 个基函数线性独立, 个数等于 :math:`\\dim P_p(\\mathbb S)`;
- 多项式复现: 随机 :math:`\\sigma \\in P_p(\\mathbb S)` 由全局最小二乘精确复现, 散度也精确;
- :math:`H(\\mathrm{div})` 协调: 随机全局系数下, 内部面两侧的法向迹 :math:`\\sigma n` 相等;
- 自由度含义: 顶点系数等于 Voigt 重数乘 :math:`\\sigma(v) : S_k`, 二维与三维约定一致.

三维曾在 ``basis_frame_of_S`` 中对 :math:`v \\otimes v` 型张量多乘 ``prod(alpha!)``, 边与面
标架也未归一化; 空间不变, 但自由度系数的含义与二维不同.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.functionspace import HuZhangFESpace
from soptx.mesh import TetrahedronMesh, TriangleMesh


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _mesh(dim: int):
    """二维 2x2 三角形网格或三维 1x1x1 四面体网格."""
    if dim == 2:
        return TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
    return TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1)


CASES = [(2, 2), (2, 3), (3, 2), (3, 4)]
CASE_IDS = [f"{d}d-p{p}" for d, p in CASES]


def _pairs(d: int):
    """Voigt 分量 [xx, xy, ...] 对应的下标对."""
    return [(i, j) for i in range(d) for j in range(i, d)]


def _to_matrix(v: np.ndarray, d: int) -> np.ndarray:
    """Voigt 分量 (..., NS) 转为对称矩阵 (..., d, d)."""
    out = np.zeros(v.shape[:-1] + (d, d))
    for k, (i, j) in enumerate(_pairs(d)):
        out[..., i, j] = v[..., k]
        out[..., j, i] = v[..., k]
    return out


def _random_polynomial_tensor(d: int, p: int, rng):
    """随机 :math:`P_p` 对称张量场及其散度 (单项式基, 解析求导)."""
    exps = [e for e in np.ndindex(*([p + 1] * d)) if sum(e) <= p]
    coef = rng.standard_normal((len(_pairs(d)), len(exps)))

    def monomials(x, shift=None):
        cols = []
        for e in exps:
            e = np.array(e)
            if shift is not None:
                e = e.copy(); e[shift] -= 1
            cols.append(np.prod(x ** np.maximum(e, 0), axis=-1) * (e >= 0).all())
        return np.stack(cols, axis=-1)

    def value(x):
        return monomials(x) @ coef.T

    def divergence(x):
        out = np.zeros(x.shape[:-1] + (d,))
        for k, (i, j) in enumerate(_pairs(d)):
            for a, b in {(i, j), (j, i)}:
                weights = coef[k] * np.array([e[b] for e in exps])
                out[..., a] += monomials(x, shift=b) @ weights
        return out

    return value, divergence


def _fit(space, d, p, rng):
    """在随机采样点上对随机 P_p(S) 场做全局最小二乘, 返回 (系数, 场, 散度, 点, 重心坐标, 相对残差)."""
    mesh = space.mesh
    ldof, gdof = space.number_of_local_dofs(), space.number_of_global_dofs()
    bcs = bm.tensor(rng.dirichlet(np.ones(d + 1), size=2 * ldof))
    phi = bm.to_numpy(space.basis(bcs))                       # (NC, NQ, ldof, NS)
    NC, NQ, _, NS = phi.shape
    value, divergence = _random_polynomial_tensor(d, p, rng)
    points = bm.to_numpy(mesh.bc_to_point(bcs))
    c2d = bm.to_numpy(space.cell_to_dof())
    A = np.zeros((NC, NQ, NS, gdof))
    for c in range(NC):
        np.add.at(A[c], (slice(None), slice(None), c2d[c]), phi[c].transpose(0, 2, 1))
    A, b = A.reshape(-1, gdof), value(points).reshape(-1)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    residual = np.linalg.norm(A @ coef - b) / np.linalg.norm(b)
    return coef, value, divergence, points, bcs, residual


@pytest.mark.parametrize("dim, p", CASES, ids=CASE_IDS)
def test_local_basis_spans_symmetric_polynomials(dim, p):
    """每个单元上的基函数线性独立, 个数等于 dim P_p(S)."""
    space = HuZhangFESpace(_mesh(dim), p=p)
    ldof = space.number_of_local_dofs()
    n_scalar = len([e for e in np.ndindex(*([p + 1] * dim)) if sum(e) <= p])
    assert ldof == len(_pairs(dim)) * n_scalar

    rng = np.random.default_rng(0)
    phi = bm.to_numpy(space.basis(bm.tensor(rng.dirichlet(np.ones(dim + 1), size=3 * ldof))))
    for c in range(phi.shape[0]):
        assert np.linalg.matrix_rank(phi[c].transpose(0, 2, 1).reshape(-1, ldof)) == ldof


@pytest.mark.parametrize("dim, p", CASES, ids=CASE_IDS)
def test_reproduces_symmetric_polynomials_and_divergence(dim, p):
    """随机 P_p(S) 场被精确复现, 有限元函数的散度等于解析散度."""
    rng = np.random.default_rng(1)
    space = HuZhangFESpace(_mesh(dim), p=p)
    coef, _, divergence, points, bcs, residual = _fit(space, dim, p, rng)
    assert residual < 1e-12

    div_h = bm.to_numpy(space.div_value(bm.tensor(coef), bcs))
    exact = divergence(points)
    assert np.abs(div_h - exact).max() < 1e-11 * np.abs(exact).max()


def _value_at(space, uh, cell, x):
    """有限元函数在 cell 内物理点 x 处的 Voigt 值."""
    mesh = space.mesh
    X = bm.to_numpy(mesh.entity("node"))[bm.to_numpy(mesh.entity("cell"))[cell]]
    lam = np.linalg.solve((X[1:] - X[0]).T, x - X[0])
    bc = np.concatenate([[1.0 - lam.sum()], lam])
    phi = bm.to_numpy(space.basis(bm.tensor(bc[None, :])))[cell, 0]
    return phi.T @ uh[bm.to_numpy(space.cell_to_dof())[cell]]


@pytest.mark.parametrize("dim, p", CASES, ids=CASE_IDS)
def test_normal_trace_is_continuous(dim, p):
    """随机全局系数下, 内部面两侧的 sigma n 相等, 而完整张量一般不连续."""
    rng = np.random.default_rng(2)
    space = HuZhangFESpace(_mesh(dim), p=p)
    mesh = space.mesh
    uh = rng.standard_normal(space.number_of_global_dofs())
    node, face = bm.to_numpy(mesh.entity("node")), bm.to_numpy(mesh.entity("face"))
    f2c = bm.to_numpy(mesh.face_to_cell())

    normal_jump = full_jump = scale = 0.0
    for f in np.nonzero(f2c[:, 0] != f2c[:, 1])[0]:
        Y = node[face[f]]
        if dim == 3:
            n = np.cross(Y[1] - Y[0], Y[2] - Y[0])
        else:
            n = np.array([Y[1, 1] - Y[0, 1], Y[0, 0] - Y[1, 0]])
        n = n / np.linalg.norm(n)
        for w in rng.dirichlet(np.ones(dim), size=3):
            x = w @ Y
            s0 = _to_matrix(_value_at(space, uh, f2c[f, 0], x), dim)
            s1 = _to_matrix(_value_at(space, uh, f2c[f, 1], x), dim)
            normal_jump = max(normal_jump, np.abs((s0 - s1) @ n).max())
            full_jump = max(full_jump, np.abs(s0 - s1).max())
            scale = max(scale, np.abs(s0).max())

    assert normal_jump < 1e-12 * scale
    assert full_jump > 1e-3 * scale


@pytest.mark.parametrize("dim, p", [(2, 3), (3, 4)], ids=["2d-p3", "3d-p4"])
def test_vertex_coefficients_follow_voigt_convention(dim, p):
    """顶点系数 = Voigt 重数 x (sigma(v) : S_k); 二维与三维约定相同."""
    rng = np.random.default_rng(3)
    space = HuZhangFESpace(_mesh(dim), p=p)
    mesh = space.mesh
    coef, value, *_ = _fit(space, dim, p, rng)

    node2dof = space.dof.node_to_internal_dof()
    node2dof = node2dof[0] if isinstance(node2dof, tuple) else node2dof
    node2dof = bm.to_numpy(node2dof).reshape(mesh.number_of_nodes(), -1)
    frames = _to_matrix(bm.to_numpy(space.dof_frame_of_S()[0]), dim)      # (NN, NS, d, d)
    sigma = _to_matrix(value(bm.to_numpy(mesh.entity("node"))), dim)       # (NN, d, d)
    weights = np.array([1.0 if i == j else 2.0 for i, j in _pairs(dim)])

    expected = weights * np.einsum("nij,nkij->nk", sigma, frames)
    np.testing.assert_allclose(coef[node2dof], expected, rtol=1e-9, atol=1e-9 * np.abs(expected).max())


def test_3d_frames_are_orthonormal():
    """三维各实体的向量标架都是单位正交基, 与二维一致."""
    space = HuZhangFESpace(TetrahedronMesh.from_box([0, 2, 0, 1, 0, 1], nx=1, ny=1, nz=1), p=4)
    for frame in space.dof_frame():
        frame = bm.to_numpy(frame)
        gram = np.einsum("nij,nkj->nik", frame, frame)
        np.testing.assert_allclose(gram, np.broadcast_to(np.eye(3), gram.shape), atol=1e-13)
