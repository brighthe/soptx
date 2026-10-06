"""胡张分析器的补丁检验: 精确解落在离散空间内时, 离散解应精确到舍入误差.

取二次位移 :math:`u`, 则应力 :math:`\\sigma = \\mathcal C \\varepsilon(u)` 为线性、体力为常数.
当 :math:`u \\in P_{p-1}` 且 :math:`\\sigma \\in P_p(\\mathbb S)` 时 (二维 :math:`p=3`, 三维 :math:`p=4`),
精确解满足离散方程, 而离散问题适定, 故离散解即精确解. 全边界给非齐次位移边界
(自然施加) 检验与维数无关的位移边界项; 把 :math:`x = 1` 换成牵引边界 (强施加) 检验
牵引写值与系统修改. 二维为对照.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.fem.analyzers.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TetrahedronMesh, TriangleMesh
from soptx.problems.loads import BodyForceLoad, BoundaryTractionLoad

E, NU = 1.0, 0.3


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


class QuadraticPatchProblem:
    """二次位移的制造解, 全边界为非齐次位移边界.

    :math:`u_i = a_i + B_{ij} x_j + \\tfrac12 x^{\\mathsf T} H^{(i)} x`, 系数随机.
    """

    def __init__(self, dim: int, seed: int = 0, traction_on_x1: bool = False):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.traction_on_x1 = traction_on_x1
        self.domain = [0.0, 1.0] * dim
        self.a = rng.standard_normal(dim)
        self.B = rng.standard_normal((dim, dim))
        H = rng.standard_normal((dim, dim, dim))
        self.H = 0.5 * (H + H.transpose(0, 2, 1))       # H[i] 对称
        self.lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        self.mu = E / (2 * (1 + NU))

    def _grad_u(self, x):
        """grad u, 形状 (..., d, d), [i, j] = d u_i / d x_j."""
        return self.B + np.einsum("ijk,...k->...ij", self.H, x)

    def displacement(self, x):
        return self.a + x @ self.B.T + 0.5 * np.einsum("...j,ijk,...k->...i", x, self.H, x)

    def stress_matrix(self, x):
        G = self._grad_u(x)
        eps = 0.5 * (G + np.swapaxes(G, -1, -2))
        return 2 * self.mu * eps + self.lam * np.trace(eps, axis1=-2, axis2=-1)[..., None, None] * np.eye(self.dim)

    def body_force(self, x):
        """f = -div sigma, 常数."""
        tr_H = np.einsum("kjj->k", self.H)                # sum_j d2 u_k / dx_j^2 = Δu_k
        grad_div = np.einsum("jji->i", self.H)            # d/dx_i (div u) = sum_j H[j][j, i]
        f = -(self.mu * tr_H + (self.mu + self.lam) * grad_div)
        return np.broadcast_to(f, x.shape).copy()

    def mark_corners(self, node):
        return bm.zeros((0, self.dim), dtype=bm.float64)

    def _on_x1(self, points):
        return bm.abs(points[..., 0] - 1.0) < 1e-12

    @cartesian
    def is_displacement_boundary(self, points):
        if self.traction_on_x1:
            return ~self._on_x1(points)
        return bm.ones(points.shape[:-1], dtype=bm.bool)

    @cartesian
    def is_traction_boundary(self, points):
        if self.traction_on_x1:
            return self._on_x1(points)
        return bm.zeros(points.shape[:-1], dtype=bm.bool)

    def _traction_x1(self, points):
        """x = 1 上的外法向牵引 sigma e_x."""
        return bm.tensor(self.stress_matrix(bm.to_numpy(points))[..., :, 0])

    @cartesian
    def displacement_bc(self, points):
        return bm.tensor(self.displacement(bm.to_numpy(points)))

    def loads(self):
        body = BodyForceLoad(self.dim, cartesian(lambda p: bm.tensor(self.body_force(bm.to_numpy(p)))))
        if not self.traction_on_x1:
            return (body,)
        return (body, BoundaryTractionLoad(self.dim, self._on_x1, self._traction_x1))


def _solve(dim: int, p: int, traction_on_x1: bool = False):
    problem = QuadraticPatchProblem(dim, traction_on_x1=traction_on_x1)
    if dim == 2:
        mesh = TriangleMesh.from_box(problem.domain, nx=2, ny=2)
        hypothesis = "plane_strain"
    else:
        mesh = TetrahedronMesh.from_box(problem.domain, nx=1, ny=1, nz=1)
        hypothesis = "3D"
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=E, poisson_ratio=NU, hypothesis=hypothesis, enable_logging=False
    )
    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh, pde=problem, material=material, interpolation_scheme=None,
        space_degree=p, integration_order=p + 3, use_relaxation=False,
        solve_method="scipy", topopt_algorithm=None,
    )
    state = analyzer.solve_state(solver="scipy")
    return problem, mesh, analyzer, state


@pytest.mark.parametrize("traction_on_x1", [False, True], ids=["displacement", "traction-x1"])
@pytest.mark.parametrize("dim, p", [(2, 3), (3, 4)], ids=["2d-p3", "3d-p4"])
def test_quadratic_patch(dim, p, traction_on_x1):
    """非齐次位移边界 (及 x = 1 上的强施加牵引) 下, 应力与位移都精确复现."""
    problem, mesh, analyzer, state = _solve(dim, p, traction_on_x1)
    bcs = mesh.quadrature_formula(p + 2).get_quadrature_points_and_weights()[0]
    points = bm.to_numpy(mesh.bc_to_point(bcs))

    sigma_h = bm.to_numpy(analyzer.huzhang_space.value(state["stress"][:], bcs))
    pairs = [(i, j) for i in range(dim) for j in range(i, dim)]
    sigma = problem.stress_matrix(points)
    sigma = np.stack([sigma[..., i, j] for i, j in pairs], axis=-1)
    assert np.abs(sigma_h - sigma).max() < 1e-9 * np.abs(sigma).max()

    u_h = bm.to_numpy(state["displacement"](bcs))
    u = problem.displacement(points)
    assert np.abs(u_h - u).max() < 1e-9 * np.abs(u).max()


def test_3d_stress_components_reordered_for_material():
    """三维积分点应力按材料约定 [xx, yy, zz, yz, xz, xy] 重排."""
    problem, mesh, analyzer, state = _solve(3, 4)
    q = analyzer.integration_order
    bcs = mesh.quadrature_formula(q).get_quadrature_points_and_weights()[0]
    points = bm.to_numpy(mesh.bc_to_point(bcs))
    S = problem.stress_matrix(points)
    expected = np.stack([S[..., 0, 0], S[..., 1, 1], S[..., 2, 2], S[..., 1, 2], S[..., 0, 2], S[..., 0, 1]], axis=-1)
    got = bm.to_numpy(analyzer.extract_stress_at_quadrature_points(state["stress"][:]))
    assert np.abs(got - expected).max() < 1e-9 * np.abs(expected).max()
