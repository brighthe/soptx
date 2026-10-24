from __future__ import annotations

import numpy as np

from soptx.backend import backend_manager as bm

from soptx.protocols import DirichletElasticityProblem, ElasticityProblem
from soptx.problems import (
    FullMBBBeamLine3d,
    FullMBBBeamPoint3d,
    HalfMBBBeamLine3d,
    HalfMBBBeamPoint2d,
)
from soptx.problems.loads import PointForceLoad
from soptx.protocols import LineTraction


def test_half_mbb_problem_is_mesh_independent_and_satisfies_lagrange_contract() -> None:
    problem = HalfMBBBeamPoint2d()

    assert isinstance(problem, ElasticityProblem)
    assert isinstance(problem, DirichletElasticityProblem)
    assert not hasattr(problem, "init_mesh")
    assert not hasattr(problem, "get_passive_element_mask")


def test_half_mbb_problem_boundary_conditions_and_concentrated_load() -> None:
    bm.set_backend("numpy")
    problem = HalfMBBBeamPoint2d(P=-2.5)
    points = bm.array([[0.0, 20.0], [60.0, 0.0], [30.0, 10.0]])

    np.testing.assert_allclose(
        bm.to_numpy(problem.dirichlet_bc(points)),
        np.zeros((3, 2)),
        rtol=0.0,
        atol=0.0,
    )

    dirichlet_x, dirichlet_y = problem.is_dirichlet_boundary()
    np.testing.assert_array_equal(
        bm.to_numpy(dirichlet_x(points)),
        np.array([True, False, False]),
    )
    np.testing.assert_array_equal(
        bm.to_numpy(dirichlet_y(points)),
        np.array([False, True, False]),
    )

    loads = problem.loads()
    assert len(loads) == 1
    assert isinstance(loads[0], PointForceLoad)
    assert loads[0].point == (0.0, 20.0)
    assert loads[0].force() == (0.0, -2.5)
    assert not hasattr(problem, "body_force")
    assert not hasattr(problem, "concentrate_load_bc")
    assert not hasattr(problem, "is_concentrate_load_boundary")


MBB3D_DOMAIN = (0.0, 6.0, 0.0, 1.0, 0.0, 1.0)


def _mbb3d_mesh_space(nx: int, ny: int, nz: int):
    """在 MBB3D_DOMAIN 上生成与剖分数一致的六面体网格及 Q1 张量空间."""
    from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
    from soptx.mesh import HexahedronMesh

    mesh = HexahedronMesh.from_box(list(MBB3D_DOMAIN), nx, ny, nz)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=1, ctype="C"), shape=(-1, 3))
    return mesh, space


def _assert_rigid_body_modes_removed(problem, mesh) -> None:
    """消元后的刚度矩阵正定, 即刚体运动全部被消除."""
    from soptx.fem.analyzers import LagrangeFEMAnalyzer
    from soptx.materials import IsotropicLinearElasticMaterial

    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis="3D")
    analyzer = LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                                   integration_order=2, solve_method="scipy")
    K, _ = analyzer.apply_bc(analyzer.assemble_stiff_matrix(), analyzer.assemble_body_force_vector())
    eigenvalues = np.linalg.eigvalsh(K.to_scipy().toarray())
    assert eigenvalues.min() > 1e-8 * eigenvalues.max()


def test_full_mbb3d_defaults_and_extruded_line_load() -> None:
    bm.set_backend("numpy")
    problem = FullMBBBeamLine3d(domain=MBB3D_DOMAIN)
    assert problem.plane_type == "3D"
    assert not hasattr(problem, "support_layout")

    (load,) = problem.loads()
    assert isinstance(load, LineTraction)
    points = bm.array([[3.0, 1.0, 0.0], [3.0, 1.0, 0.4], [3.0, 0.0, 0.4], [2.0, 1.0, 0.4]])
    np.testing.assert_array_equal(bm.to_numpy(load.is_load_line(points)), [True, True, False, False])
    np.testing.assert_allclose(bm.to_numpy(load.traction(points[:1])), [[0.0, -1.0, 0.0]])


def test_full_mbb3d_supports_are_2d_supports_extruded_along_z() -> None:
    """两端底棱为二维支承沿 z 的拉伸, u_z 只在角点 (x0, y0, z0) 约束."""
    bm.set_backend("numpy")
    problem = FullMBBBeamLine3d(domain=MBB3D_DOMAIN)
    points = bm.array([
        [0.0, 0.0, 0.0],   # 左端底棱 z0 端点
        [0.0, 0.0, 0.7],   # 左端底棱中部
        [6.0, 0.0, 0.0],   # 右端底棱 z0 端点
        [3.0, 0.0, 0.5],   # 底面中线, 不约束
        [0.0, 1.0, 0.0],   # 左端顶棱, 不约束
    ])

    dirichlet_x, dirichlet_y, dirichlet_z = problem.is_dirichlet_boundary()
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_x(points)), [True, True, False, False, False])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_y(points)), [True, True, True, False, False])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_z(points)), [True, False, False, False, False])
    np.testing.assert_array_equal(bm.to_numpy(problem.dirichlet_bc(points)), np.zeros((5, 3)))


def test_full_mbb3d_is_statically_determinate_on_any_mesh() -> None:
    """奇数 nz 下同样成立: 线支承加单点 u_z 恰好消除全部刚体运动."""
    bm.set_backend("numpy")
    nx, ny, nz = 6, 2, 3
    problem = FullMBBBeamLine3d(domain=MBB3D_DOMAIN)
    mesh, space = _mbb3d_mesh_space(nx, ny, nz)

    is_ddof = bm.to_numpy(space.is_boundary_dof(threshold=problem.is_dirichlet_boundary(),
                                                method="interp")).reshape(-1, 3)
    np.testing.assert_array_equal(is_ddof.sum(axis=0), [nz + 1, 2 * (nz + 1), 1])
    _assert_rigid_body_modes_removed(problem, mesh)


def test_half_mbb3d_is_statically_determinate_on_any_mesh() -> None:
    """对称面 u_x, 右端底棱 u_y, 单点 u_z; 奇数 nz 下不再出现奇异刚度矩阵."""
    bm.set_backend("numpy")
    nx, ny, nz = 6, 2, 3
    problem = HalfMBBBeamLine3d(domain=MBB3D_DOMAIN)
    assert problem.plane_type == "3D"
    mesh, space = _mbb3d_mesh_space(nx, ny, nz)

    is_ddof = bm.to_numpy(space.is_boundary_dof(threshold=problem.is_dirichlet_boundary(),
                                                method="interp")).reshape(-1, 3)
    np.testing.assert_array_equal(is_ddof.sum(axis=0), [(ny + 1) * (nz + 1), nz + 1, 1])
    _assert_rigid_body_modes_removed(problem, mesh)


def test_mbb3d_line_loads_keep_the_resultant_on_a_mesh() -> None:
    """全域与半域线载荷装配后合力均为 P, 且只作用在载荷线的节点上."""
    from soptx.fem.load_assembly import assemble_non_body_loads

    bm.set_backend("numpy")
    for problem, x_line in ((FullMBBBeamLine3d(domain=MBB3D_DOMAIN, P=-2.0), 3.0),
                            (HalfMBBBeamLine3d(domain=MBB3D_DOMAIN, P=-2.0), 0.0)):
        _, space = _mbb3d_mesh_space(6, 2, 3)
        vector = bm.to_numpy(assemble_non_body_loads(space, problem.loads(), q=2)).reshape(-1, 3)
        points = bm.to_numpy(space.interpolation_points())
        loaded = np.flatnonzero(vector[:, 1])
        np.testing.assert_allclose(vector.sum(axis=0), [0.0, -2.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(points[loaded][:, :2], np.tile([x_line, 1.0], (loaded.size, 1)), atol=1e-14)
        assert loaded.size == 4


def test_full_mbb3d_point_supports_are_the_four_bottom_end_corners() -> None:
    """左端两底角点约束全部分量, 右端两底角点约束 u_y 与 u_z, 底棱中部不约束."""
    bm.set_backend("numpy")
    problem = FullMBBBeamPoint3d(domain=MBB3D_DOMAIN)
    assert problem.plane_type == "3D"
    points = bm.array([
        [0.0, 0.0, 0.0],   # 左端底角 z0
        [0.0, 0.0, 1.0],   # 左端底角 z1
        [6.0, 0.0, 0.0],   # 右端底角 z0
        [6.0, 0.0, 1.0],   # 右端底角 z1
        [0.0, 0.0, 0.5],   # 左端底棱中部, 不约束
        [0.0, 1.0, 0.0],   # 左端顶角, 不约束
    ])

    dirichlet_x, dirichlet_y, dirichlet_z = problem.is_dirichlet_boundary()
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_x(points)), [True, True, False, False, False, False])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_y(points)), [True, True, True, True, False, False])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_z(points)), [True, True, True, True, False, False])
    np.testing.assert_array_equal(bm.to_numpy(problem.dirichlet_bc(points)), np.zeros((6, 3)))


def test_full_mbb3d_point_removes_rigid_body_modes_on_any_mesh() -> None:
    """奇数 nz 下四角点支承同样消除全部刚体运动."""
    bm.set_backend("numpy")
    problem = FullMBBBeamPoint3d(domain=MBB3D_DOMAIN)
    mesh, space = _mbb3d_mesh_space(6, 2, 3)

    is_ddof = bm.to_numpy(space.is_boundary_dof(threshold=problem.is_dirichlet_boundary(),
                                                method="interp")).reshape(-1, 3)
    np.testing.assert_array_equal(is_ddof.sum(axis=0), [2, 4, 4])
    _assert_rigid_body_modes_removed(problem, mesh)


def test_full_mbb3d_point_load_is_the_top_center_force() -> None:
    """顶面中心集中力在非节点位置 (奇数 nz) 装配后合力为 P, 只作用在顶面中线两侧节点上."""
    from soptx.fem.load_assembly import assemble_non_body_loads

    bm.set_backend("numpy")
    problem = FullMBBBeamPoint3d(domain=MBB3D_DOMAIN, P=-2.0)
    (load,) = problem.loads()
    assert isinstance(load, PointForceLoad)
    assert load.point == (3.0, 1.0, 0.5)
    assert load.force() == (0.0, -2.0, 0.0)

    _, space = _mbb3d_mesh_space(6, 2, 3)
    vector = bm.to_numpy(assemble_non_body_loads(space, problem.loads(), q=2)).reshape(-1, 3)
    points = bm.to_numpy(space.interpolation_points())
    loaded = np.flatnonzero(vector[:, 1])
    np.testing.assert_allclose(vector.sum(axis=0), [0.0, -2.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(points[loaded][:, :2], np.tile([3.0, 1.0], (loaded.size, 1)), atol=1e-14)
    np.testing.assert_allclose(np.sort(points[loaded][:, 2]), [1.0 / 3.0, 2.0 / 3.0], atol=1e-14)
