"""Hu--Zhang 未松弛几何角点上的强约束牵引写入测试.

逐边写入时顶点自由度只有一组标架: 两条法向不平行的强约束边交于一点, 只有与顶点标架
对齐的那条边的条件落到顶点上, 另一条边在顶点处的法向迹自由, 牵引从该边端部漏出,
漏在哪一侧取决于全局边编号. ``set_corner_traction_bc`` 按 Hu & Ma (2021) 式 (4.1)
联立两侧条件写入全部顶点自由度. 本文件锁定: 相容数据下角点值精确复现常应力, 不相容
数据报错, 未松弛的固支块满足截面精确平衡, 伴随解在对称边自由度上为零.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.fem import HuZhangMFEMAnalyzer
from soptx.functionspace import HuZhangFESpace
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TriangleMesh
from soptx.problems.loads import BodyForceLoad, BoundaryTractionLoad
from soptx.topology.interpolation import MaterialInterpolationScheme

_SIGMA0 = np.array([[1.0, 0.5], [0.5, -2.0]])   # 常应力, 用于构造相容的角点数据
_EPS = 1e-12


def _outward_normal(centers):
    """单位正方形边界边的外法向, 按边重心判定 (..., 2)."""
    x, y = centers[..., 0], centers[..., 1]
    n = np.zeros(centers.shape)
    n[np.abs(x) < _EPS] = (-1.0, 0.0)
    n[np.abs(x - 1.0) < _EPS] = (1.0, 0.0)
    n[np.abs(y) < _EPS] = (0.0, -1.0)
    n[np.abs(y - 1.0) < _EPS] = (0.0, 1.0)
    return n


def _constant_stress_traction(points):
    """常应力 _SIGMA0 在单位正方形边界上的外法向牵引, 点集按边成批 (NEb, NP, 2)."""
    pts = np.asarray(bm.to_numpy(points))
    n = _outward_normal(pts.mean(axis=-2))
    g = np.einsum("ij,ej->ei", _SIGMA0, n)
    return bm.asarray(np.broadcast_to(g[:, None, :], pts.shape).copy())


def _corner_tensors(space, uh):
    """四个角点处由顶点自由度重构的对称张量 (Voigt [xx, xy, yy]), 及其自由度."""
    node = np.asarray(bm.to_numpy(space.mesh.entity("node")))
    node2dof = np.asarray(bm.to_numpy(space.dof.node_to_internal_dof()[0]))
    nsframe = np.asarray(bm.to_numpy(space.nsframe))
    values = np.asarray(bm.to_numpy(uh[:]))
    out = []
    for corner in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        v = int(np.argmin(np.linalg.norm(node - np.asarray(corner), axis=1)))
        dofs = node2dof[v]
        out.append((nsframe[v].T @ values[dofs], dofs))
    return out


@pytest.mark.parametrize("p", [1, 2, 3])
def test_traction_corners_reproduce_constant_stress(p: int) -> None:
    """四边全牵引: 联立后四个角点的顶点张量等于常应力, 且 3 个自由度都被固定."""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=3, ny=3)
    traction = mesh.boundary_edge_flag()
    space = HuZhangFESpace(mesh, p=p, traction_edge=traction)

    uh, _ = space.set_dirichlet_bc(_constant_stress_traction, threshold=traction)
    uh, fixed = space.set_corner_traction_bc(_constant_stress_traction, uh[:], traction_edge=traction)

    expected = np.array([_SIGMA0[0, 0], _SIGMA0[0, 1], _SIGMA0[1, 1]])
    fixed = np.asarray(bm.to_numpy(fixed))
    for tensor, dofs in _corner_tensors(space, uh):
        np.testing.assert_allclose(tensor, expected, atol=1e-12, err_msg=f"p={p}")
        assert fixed[dofs].all(), f"p={p}"


def test_symmetry_corners_fix_only_constrained_components() -> None:
    """右边改为只强加切向牵引的对称边: 直角处两侧条件线性相关 (秩 2).

    顶点标架取自牵引边 (上/下边), 角点上只固定 sigma_yy 与 sigma_xy 两个自由度并复现
    常应力的这两个分量; 沿牵引边的正应力 sigma_xx 不受约束, 保持自由.
    """
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=3, ny=3)
    is_bd = np.asarray(bm.to_numpy(mesh.boundary_edge_flag()))
    centers = np.asarray(bm.to_numpy(mesh.entity_barycenter("edge")))
    right = is_bd & (np.abs(centers[:, 0] - 1.0) < _EPS)
    traction, symmetry = bm.asarray(is_bd & ~right), bm.asarray(right)
    space = HuZhangFESpace(mesh, p=2, traction_edge=traction, symmetry_edge=symmetry)

    uh, _ = space.set_dirichlet_bc(_constant_stress_traction, threshold=traction)
    uh, fixed = space.set_corner_traction_bc(_constant_stress_traction, uh[:], traction_edge=traction,
                                             symmetry_edge=symmetry, symmetry_gd=_constant_stress_traction)

    fixed = np.asarray(bm.to_numpy(fixed))
    for tensor, dofs in _corner_tensors(space, uh)[1:3]:   # (1, 0) 与 (1, 1)
        np.testing.assert_allclose(tensor[1:], [_SIGMA0[0, 1], _SIGMA0[1, 1]], atol=1e-12)
        assert int(fixed[dofs].sum()) == 2


def test_inconsistent_corner_traction_raises() -> None:
    """顶边加切向牵引而侧边自由: 顶角两侧数据不相容, 离散应力无解, 须报错."""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    traction = mesh.boundary_edge_flag()
    space = HuZhangFESpace(mesh, p=2, traction_edge=traction)

    def shear_on_top(points):
        pts = np.asarray(bm.to_numpy(points))
        g = np.zeros(pts.shape)
        g[np.abs(pts.mean(axis=-2)[:, 1] - 1.0) < _EPS] = (1.0, 0.0)
        return bm.asarray(g)

    with pytest.raises(ValueError, match="不相容"):
        space.set_corner_traction_bc(shear_on_top, traction_edge=traction)


class _BodyForceBlock:
    """矩形块 [0, L] x [0, H], 底边固支, 竖向体力 (0, -1); 右边可选为对称边 (滚动支承)."""

    dimension = 2

    def __init__(self, L=4.0, H=2.0, roller_right=False):
        self.L, self.H, self._roller = L, H, roller_right
        self.domain = (0.0, L, 0.0, H)

    @cartesian
    def is_displacement_boundary(self, points):
        return bm.abs(points[..., 1]) < _EPS

    @cartesian
    def is_symmetry_boundary(self, points):
        right = bm.abs(points[..., 0] - self.L) < _EPS
        return right & (points[..., 1] > _EPS) if self._roller else bm.zeros_like(right)

    @cartesian
    def is_traction_boundary(self, points):
        x, y = points[..., 0], points[..., 1]
        right = (bm.abs(x - self.L) < _EPS) & (not self._roller)
        return (bm.abs(x) < _EPS) | right | (bm.abs(y - self.H) < _EPS)

    def loads(self):
        def body(points):
            f = bm.zeros(points.shape, **bm.context(points))
            return bm.set_at(f, (..., 1), -1.0)
        zero = cartesian(lambda points: bm.zeros(points.shape, **bm.context(points)))
        return (BodyForceLoad(dimension=2, value=cartesian(body)),
                BoundaryTractionLoad(dimension=2, marker=self.is_traction_boundary, value=zero))


def _solve_block(k, roller_right=False):
    problem = _BodyForceBlock(roller_right=roller_right)
    mesh = TriangleMesh.from_checkerboard_box(box=problem.domain, nx=16, ny=8)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.25, hypothesis="plane_stress")
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="msimp",
        options={"penalty_factor": 3.0, "void_youngs_modulus": 1e-9, "target_variables": ["E"]})
    _, rho = interpolation.setup_density_distribution(
        design_variable_mesh=mesh, displacement_mesh=mesh, relative_density=1.0)
    analyzer = HuZhangMFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=k,
                                   integration_order=2 * k + 2, solve_method="mumps", use_relaxation=False,
                                   topopt_algorithm="density_based", interpolation_scheme=interpolation)
    state = analyzer.solve_state(rho_val=rho)
    return problem, mesh, analyzer, state


@pytest.mark.parametrize("k", [2, 3])
def test_unrelaxed_block_satisfies_section_equilibrium(k: int) -> None:
    """未松弛: 顶部自由角不再漏力, 水平截面上 sigma_yy 的全宽平均精确等于 -(H - y)."""
    bm.set_backend("numpy")
    problem, mesh, analyzer, state = _solve_block(k)
    q = 2 * k + 2
    _, weights = mesh.quadrature_formula(q).get_quadrature_points_and_weights()
    stress = np.asarray(bm.to_numpy(
        analyzer.extract_stress_at_quadrature_points(stress_dof=state["stress"], integration_order=q)))
    area = np.asarray(bm.to_numpy(mesh.entity_measure("cell")))
    cell_integral = np.einsum("q,cq->c", np.asarray(bm.to_numpy(weights)), stress[..., 1]) * area
    centers_y = np.asarray(bm.to_numpy(mesh.entity_barycenter("cell")))[:, 1]
    h = problem.H / 8
    for row in range(8):   # 每一行单元恰好是一条水平带
        band = np.abs(centers_y - (row + 0.5) * h) < 0.5 * h
        mean_yy = cell_integral[band].sum() / area[band].sum()
        np.testing.assert_allclose(mean_yy, -(problem.H - (row + 0.5) * h), atol=1e-10,
                                   err_msg=f"k={k}, row={row}")


def test_adjoint_vanishes_on_symmetry_dofs() -> None:
    """伴随方程与状态方程共用固定集合: 对称边切向自由度上的伴随解为零, 而非伴随载荷本身."""
    bm.set_backend("numpy")
    problem, mesh, analyzer, _ = _solve_block(2, roller_right=True)
    space = analyzer.huzhang_space
    gdof = space.number_of_global_dofs()
    lam = np.asarray(bm.to_numpy(analyzer.solve_adjoint(rhs=bm.ones(gdof, dtype=bm.float64))))

    symmetry = mesh.boundary_edge_flag() & problem.is_symmetry_boundary(mesh.entity_barycenter("edge"))
    zero = cartesian(lambda points: bm.zeros(points.shape, **bm.context(points)))
    _, is_sym_dof = space.set_tangential_traction_bc(zero, threshold=symmetry)
    is_sym_dof = np.asarray(bm.to_numpy(is_sym_dof))
    assert is_sym_dof.any()
    np.testing.assert_allclose(lam[:gdof][is_sym_dof], 0.0, atol=1e-12)
