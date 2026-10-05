from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.protocols import DirichletElasticityProblem, ElasticityProblem
from soptx.problems import FullMBBBeam3d, HalfMBBBeamRight2d
from soptx.problems.loads import PointForceLoad


def test_half_mbb_problem_is_mesh_independent_and_satisfies_lagrange_contract() -> None:
    problem = HalfMBBBeamRight2d()

    assert isinstance(problem, ElasticityProblem)
    assert isinstance(problem, DirichletElasticityProblem)
    assert not hasattr(problem, "init_mesh")
    assert not hasattr(problem, "get_passive_element_mask")


def test_half_mbb_problem_boundary_conditions_and_concentrated_load() -> None:
    bm.set_backend("numpy")
    problem = HalfMBBBeamRight2d(P=-2.5)
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
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import HexahedronMesh

    mesh = HexahedronMesh.from_box(list(MBB3D_DOMAIN), nx, ny, nz)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=1, ctype="C"), shape=(-1, 3))
    return mesh, space


def test_full_mbb3d_defaults_keep_centerline_support_and_center_load() -> None:
    bm.set_backend("numpy")
    problem = FullMBBBeam3d(domain=MBB3D_DOMAIN)
    assert problem.support == "centerline"
    assert problem.load_subdivisions is None

    loads = problem.loads()
    assert len(loads) == 1
    assert tuple(loads[0].point) == (3.0, 1.0, 0.5)
    assert tuple(loads[0].force()) == (0.0, -1.0, 0.0)

    # u_z 只约束底面上距中线最近的点, 左端底边上偏离中线的点不约束 u_z
    _, _, dirichlet_z = problem.is_dirichlet_boundary()
    points = bm.array([[3.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [3.0, 1.0, 0.5]])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_z(points)), [True, True, False, False])


def test_full_mbb3d_end_lines_constrains_bottom_end_edges_only() -> None:
    bm.set_backend("numpy")
    problem = FullMBBBeam3d(domain=MBB3D_DOMAIN, support="end_lines")
    points = bm.array([
        [0.0, 0.0, 0.0],   # 左端底边
        [0.0, 0.0, 0.7],   # 左端底边
        [6.0, 0.0, 0.4],   # 右端底边
        [3.0, 0.0, 0.5],   # 底面中线, 不约束
        [0.0, 1.0, 0.5],   # 左端顶边, 不约束
    ])

    dirichlet_x, dirichlet_y, dirichlet_z = problem.is_dirichlet_boundary()
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_x(points)), [True, True, False, False, False])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_y(points)), [True, True, True, False, False])
    np.testing.assert_array_equal(bm.to_numpy(dirichlet_z(points)), [True, True, False, False, False])
    np.testing.assert_array_equal(bm.to_numpy(problem.dirichlet_bc(points)), np.zeros((5, 3)))


def test_full_mbb3d_end_lines_dirichlet_dofs_on_a_mesh() -> None:
    bm.set_backend("numpy")
    nx, ny, nz = 6, 2, 3
    problem = FullMBBBeam3d(domain=MBB3D_DOMAIN, support="end_lines")
    _, space = _mbb3d_mesh_space(nx, ny, nz)

    is_ddof = bm.to_numpy(space.is_boundary_dof(threshold=problem.is_dirichlet_boundary(),
                                                method="interp")).reshape(-1, 3)
    # 左端底边每个节点约束 3 个分量, 右端底边每个节点只约束 u_y
    assert is_ddof.sum() == 3 * (nz + 1) + (nz + 1)
    np.testing.assert_array_equal(is_ddof.sum(axis=0), [nz + 1, 2 * (nz + 1), nz + 1])


def test_full_mbb3d_splits_load_by_subdivision_parity() -> None:
    expected_counts = {(6, 4): 1, (6, 3): 2, (5, 3): 4}
    for (nx, nz), count in expected_counts.items():
        loads = FullMBBBeam3d(domain=MBB3D_DOMAIN, P=-2.0, support="end_lines",
                              load_subdivisions=(nx, nz)).loads()

        assert len(loads) == count
        assert all(isinstance(load, PointForceLoad) for load in loads)
        np.testing.assert_allclose(sum(load.force()[1] for load in loads), -2.0, rtol=0.0, atol=1e-15)
        assert all(load.force()[0] == 0.0 and load.force()[2] == 0.0 for load in loads)
        assert all(load.point[1] == 1.0 for load in loads)
        # 载荷点关于 x = 3, z = 0.5 两个中面对称
        np.testing.assert_allclose(np.mean([load.point[0] for load in loads]), 3.0, atol=1e-14)
        np.testing.assert_allclose(np.mean([load.point[2] for load in loads]), 0.5, atol=1e-14)


def test_full_mbb3d_split_loads_project_exactly_onto_mesh_nodes() -> None:
    from soptx.fem.load_projection import project_nodal_loads

    bm.set_backend("numpy")
    nx, ny, nz = 6, 2, 3
    problem = FullMBBBeam3d(domain=MBB3D_DOMAIN, P=-1.0, support="end_lines",
                            load_subdivisions=(nx, nz))
    _, space = _mbb3d_mesh_space(nx, ny, nz)

    # 分析器以 mode='exact' 投影点力, 作用点不在节点上会报错
    vector = bm.to_numpy(project_nodal_loads(problem.loads(), space.interpolation_points(),
                                             dimension=3)).reshape(-1, 3)
    loaded = np.nonzero(vector[:, 1])[0]

    assert loaded.size == 2
    np.testing.assert_allclose(vector[loaded, 1], [-0.5, -0.5], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(vector.sum(axis=0), [0.0, -1.0, 0.0], rtol=0.0, atol=1e-15)


def test_full_mbb3d_rejects_invalid_options() -> None:
    import pytest

    for kwargs in ({"support": "corner"},
                   {"load_subdivisions": (0, 3)}, {"load_subdivisions": (6, -1)},
                   {"load_subdivisions": (6.0, 3)}, {"load_subdivisions": (True, 3)},
                   {"load_subdivisions": (6, 3, 3)}):
        with pytest.raises(ValueError):
            FullMBBBeam3d(**kwargs)


def test_full_mbb3d_rejects_centerline_with_odd_nz() -> None:
    import pytest

    with pytest.raises(ValueError, match="end_lines"):
        FullMBBBeam3d(support="centerline", load_subdivisions=(6, 3))

    # nz 为偶数时中线落在节点上, centerline 合法; 不给剖分数时不做检查
    assert FullMBBBeam3d(support="centerline", load_subdivisions=(6, 4)).support == "centerline"
    assert FullMBBBeam3d(support="centerline").load_subdivisions is None
    assert FullMBBBeam3d(support="end_lines", load_subdivisions=(6, 3)).support == "end_lines"
