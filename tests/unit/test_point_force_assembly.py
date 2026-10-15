"""集中力形函数分配的守恒性, 虚功与子结构接口一致性."""

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.load_assembly import assemble_point_force, assemble_non_body_loads
from soptx.fem.substructure import GlobalAssembler, StructuredSubstructureLayout
from soptx.fem.substructure.problem_adapter import project_problem_conditions_to_full_system
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.mesh import HexahedronMesh, QuadrangleMesh
from soptx.problems import FullMBBBeamLine3d
from soptx.problems.loads import PointForceLoad


class _PointLoadMBB3d(FullMBBBeamLine3d):
    """顶面中心受集中力的三维 MBB 梁: 本测试检验集中力的形函数分配."""

    def loads(self):
        x0, x1, _, y1, z0, z1 = self.domain
        return (PointForceLoad(point=((x0 + x1) / 2.0, y1, (z0 + z1) / 2.0), vector=(0.0, self.P, 0.0)),)


@pytest.fixture(params=['numpy', 'pytorch'])
def backend(request):
    """在 CPU 的两个张量后端上检查同一物理契约."""
    if request.param == 'pytorch':
        pytest.importorskip('torch')
    bm.set_backend(request.param)
    yield request.param
    bm.set_backend('numpy')


def _space(box=(0., 2., 0., 2., 0., 2.), grid=(2, 2, 2), priority=False):
    """构造测试用 Q1 连续位移空间."""
    mesh = HexahedronMesh.from_box(list(box), *grid)
    return TensorFunctionSpace(LagrangeFESpace(mesh, p=1), shape=(3, -1) if priority else (-1, 3))


def _nodal(space, vector):
    """把任意自由度排序转回节点向量以检查物理量."""
    values = bm.to_numpy(vector)
    return values.reshape(3, -1).T if space.dof_priority else values.reshape(-1, 3)


@pytest.mark.parametrize('priority', [False, True])
@pytest.mark.parametrize('point,count', [((1., 2., 0.25), 2), ((0.25, 2., 0.3), 4),
                                        ((1., 0.25, 0.3), 4), ((0.2, 0.3, 0.4), 8),
                                        ((1., 2., 1.), 1)])
def test_shape_force_preserves_resultant_moment_and_virtual_work(backend, priority, point, count):
    """边, 面, 内部与节点点力均满足合力, 力矩和三线性虚功关系."""
    space = _space(priority=priority)
    force = np.array([2., -3., 5.])
    vector = _nodal(space, assemble_point_force(space, PointForceLoad(point, force)))
    nodes = bm.to_numpy(space.interpolation_points())
    assert np.count_nonzero(np.linalg.norm(vector, axis=1)) == count
    np.testing.assert_allclose(vector.sum(axis=0), force, atol=1e-13)
    np.testing.assert_allclose(np.cross(nodes, vector).sum(axis=0), np.cross(point, force), atol=1e-13)

    def displacement(x):
        return np.stack((x[..., 0] * x[..., 1], x[..., 1] * x[..., 2],
                         x[..., 0] * x[..., 1] * x[..., 2]), axis=-1)

    np.testing.assert_allclose(np.sum(displacement(nodes) * vector),
                               displacement(np.asarray(point)) @ force, atol=1e-13)
    if point == (1., 2., 0.25):
        loaded = vector[np.linalg.norm(vector, axis=1) > 0]
        np.testing.assert_allclose(loaded, np.array([0.75, 0.25])[:, None] * force, atol=1e-13)


def test_shared_face_is_independent_of_cell_numbering(backend):
    """共享面点只计一次, 交换单元编号不改变载荷."""
    space = _space()
    load = PointForceLoad((1., 0.25, 0.3), (0., -1., 0.))
    expected = assemble_point_force(space, load)
    mesh = HexahedronMesh(space.mesh.entity('node'), bm.flip(space.mesh.entity('cell'), axis=0))
    permuted = TensorFunctionSpace(LagrangeFESpace(mesh, p=1), shape=(-1, 3))
    np.testing.assert_allclose(bm.to_numpy(assemble_point_force(permuted, load)), bm.to_numpy(expected), atol=1e-14)


def test_full_system_and_fem_share_physical_force_with_origin_offset(backend):
    """平移物理域后, 子结构局部网格与 FEM 物理网格生成相同全场载荷."""
    domain = (2., 8., -1., 1., 3., 6.)
    problem = _PointLoadMBB3d(domain=domain)
    layout = StructuredSubstructureLayout(domain_size=(6., 2., 3.), n_sub=(2, 1, 1),
                                           n_fine=(3, 2, 3), degree=1,
                                           E_base=1., nu=0.3, hypothesis='3D')
    assembler = GlobalAssembler(layout)
    vector, _ = project_problem_conditions_to_full_system(problem, assembler)
    space = _space(domain, (6, 2, 3))
    reference = assemble_non_body_loads(space, problem.loads(), q=2)
    np.testing.assert_allclose(bm.to_numpy(vector), bm.to_numpy(reference), atol=1e-14)
    values = _nodal(space, reference)
    np.testing.assert_allclose(values[values[:, 1] != 0, 1], [-0.5, -0.5], atol=1e-14)


def test_paper_load_coordinates_on_a_small_local_mesh(backend):
    """首档加载点所在局部网格给出两个指定坐标, 无需创建首档全网格."""
    space = _space((194., 196., 64., 65., 32., 33.), (2, 1, 1))
    vector = _nodal(space, assemble_point_force(space, PointForceLoad((195., 65., 32.5), (0., -1., 0.))))
    loaded = np.flatnonzero(vector[:, 1])
    np.testing.assert_allclose(bm.to_numpy(space.interpolation_points())[loaded], [[195, 65, 32], [195, 65, 33]])
    np.testing.assert_allclose(vector[loaded, 1], [-0.5, -0.5])


def test_outside_and_distorted_cells_are_rejected(backend):
    """不能将域外点吸附到边界, 也不能用畸变单元的包围盒代替实际映射."""
    space = _space()
    with pytest.raises(ValueError, match='网格外'):
        assemble_point_force(space, PointForceLoad((2.1, 2., 0.3), (0., -1., 0.)))
    mesh = HexahedronMesh.from_box([0., 1., 0., 1., 0., 1.], 1, 1, 1)
    nodes = bm.copy(mesh.entity('node'))
    nodes = bm.set_at(nodes, (7, 0), 1.2)
    distorted = TensorFunctionSpace(LagrangeFESpace(HexahedronMesh(nodes, mesh.entity('cell')), p=1), shape=(-1, 3))
    with pytest.raises(NotImplementedError, match='轴对齐'):
        assemble_point_force(distorted, PointForceLoad((0.3, 0.4, 0.2), (0., -1., 0.)))


def test_unsupported_non_nodal_space_retains_exact_node_support(backend):
    """未实现二维非节点定位时明确拒绝, 已有节点点力仍可用."""
    mesh = QuadrangleMesh.from_box([0., 1., 0., 1.], 1, 1)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=1), shape=(-1, 2))
    with pytest.raises(NotImplementedError, match='Q1'):
        assemble_point_force(space, PointForceLoad((0.3, 1.), (0., -1.)))
    vector = bm.to_numpy(assemble_point_force(space, PointForceLoad((0., 1.), (0., -1.))))
    np.testing.assert_allclose(vector.reshape(-1, 2).sum(axis=0), [0., -1.])
