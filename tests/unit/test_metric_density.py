"""``EntityView.metric_density`` 方阵 Jacobi 显式行列式路径的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh


def _reference_density(view):
    """按通用公式 sqrt(det(J^T J)) 计算度量密度, 即显式路径之前的写法."""
    q = view._default_geometry_quadrature_order()
    bcs, _ = view.quadrature_formula(q).get_quadrature_points_and_weights()
    bcs = bcs if isinstance(bcs, tuple) else (bcs, )
    jacobian = np.asarray(bm.to_numpy(view.jacobi_matrix(bcs)))
    metric = np.einsum('cqdr,cqds->cqrs', jacobian, jacobian)
    return bcs, np.sqrt(np.linalg.det(metric))


def _perturbed(mesh_class, box, shape, seed):
    """对内部节点施加随机扰动, 得到非仿射 (非平行六面体) 单元."""
    mesh = mesh_class.from_box(box, *shape)
    node = np.array(bm.to_numpy(mesh.entity('node')), dtype=np.float64)
    lower, upper = np.array(box[0::2]), np.array(box[1::2])
    interior = np.all((node > lower + 1e-12) & (node < upper - 1e-12), axis=1)
    h = (upper - lower) / np.array(shape)
    node[interior] += np.random.default_rng(seed).uniform(-0.2, 0.2, (interior.sum(), node.shape[1])) * h
    return mesh_class(bm.tensor(node), mesh.entity('cell'))


CASES = [
    (TriangleMesh, [0.0, 2.0, 0.0, 1.0], (4, 3)),
    (QuadrangleMesh, [0.0, 2.0, 0.0, 1.0], (4, 3)),
    (TetrahedronMesh, [0.0, 2.0, 0.0, 1.0, 0.0, 1.0], (3, 2, 2)),
    (HexahedronMesh, [0.0, 2.0, 0.0, 1.0, 0.0, 1.0], (3, 2, 2)),
]


@pytest.mark.parametrize('mesh_class, box, shape', CASES)
@pytest.mark.parametrize('perturb', [False, True])
def test_square_jacobian_density_matches_general_formula(mesh_class, box, shape, perturb) -> None:
    bm.set_backend('numpy')
    mesh = _perturbed(mesh_class, box, shape, seed=0) if perturb else mesh_class.from_box(box, *shape)
    view = mesh.entity_view('cell')

    bcs, expected = _reference_density(view)
    density = np.asarray(bm.to_numpy(view.metric_density(bcs)))
    np.testing.assert_allclose(density, expected, rtol=1e-13, atol=0.0)

    # 扰动只移动内部节点, 总体积仍为区域体积
    volume = float(np.sum(bm.to_numpy(mesh.entity_measure('cell'))))
    assert volume == pytest.approx(float(np.prod(np.diff(np.reshape(box, (-1, 2)), axis=1))), rel=1e-12)


def test_embedded_face_density_keeps_general_formula() -> None:
    bm.set_backend('numpy')
    mesh = _perturbed(HexahedronMesh, [0.0, 2.0, 0.0, 1.0, 0.0, 1.0], (3, 2, 2), seed=1)
    view = mesh.entity_view('face')          # 三维空间中的面, J 为 3x2, 走通用公式

    bcs, expected = _reference_density(view)
    density = np.asarray(bm.to_numpy(view.metric_density(bcs)))
    np.testing.assert_allclose(density, expected, rtol=1e-14, atol=0.0)
