"""LagrangeFEMAnalyzer 载荷装配的回归测试.

以离散合力作检验, 不依赖实现细节:

1. 体力: 常值体力 f 在矩形域上的合力为 f * 面积;
2. 体力 + 边界牵引 + 集中力: 合力为三者之和 (``BearingDevice2d`` 顶边均布牵引的合力为 t * 边长);
3. 线载荷: 三维悬臂梁右端底边的总力为 P;
4. 分量优先与节点优先两种自由度排序给出同一组载荷, 只差一个置换.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.fem import boundary_load_resultant
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh, QuadrangleMesh
from soptx.problems import BearingDevice2d, CantileverRightBottomEdge3d
from soptx.problems.loads import BodyForceLoad, PointForceLoad

BODY_FORCE = (0.3, -0.5)
POINT = (60.0, 40.0)
POINT_FORCE = (1.0, -2.0)


@pytest.fixture(autouse=True)
def reset_backend():
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


class _AllLoadsBearing(BearingDevice2d):
    """在顶边均布牵引之外再加常值体力与一个集中力的轴承算例."""

    @cartesian
    def _body_force(self, points):
        value = bm.zeros(points.shape, **bm.context(points))
        value = bm.set_at(value, (..., 0), BODY_FORCE[0])
        return bm.set_at(value, (..., 1), BODY_FORCE[1])

    def loads(self):
        return (BodyForceLoad(dimension=2, value=self._body_force),
                *super().loads(),
                PointForceLoad(point=POINT, vector=POINT_FORCE))


def _analyzer(problem, mesh, dof_priority: bool) -> LagrangeFEMAnalyzer:
    GD = mesh.geo_dimension()
    shape = (GD, -1) if dof_priority else (-1, GD)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=1, ctype='C'), shape=shape)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3,
                                              hypothesis="plane_stress" if GD == 2 else "3D",
                                              enable_logging=False)
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=3, operator_level='fa', solve_method='scipy',
                               tensor_space=space, enable_logging=False)


def _bearing(dof_priority: bool) -> LagrangeFEMAnalyzer:
    problem = _AllLoadsBearing()
    mesh = QuadrangleMesh.from_box(list(problem.domain), nx=12, ny=4)
    return _analyzer(problem, mesh, dof_priority)


def _cantilever(dof_priority: bool) -> LagrangeFEMAnalyzer:
    problem = CantileverRightBottomEdge3d()
    mesh = HexahedronMesh.from_box(list(problem.domain), nx=6, ny=2, nz=2)
    return _analyzer(problem, mesh, dof_priority)


def _resultant(analyzer: LagrangeFEMAnalyzer, F) -> np.ndarray:
    return np.asarray(bm.to_numpy(boundary_load_resultant(
        F, analyzer.disp_mesh.geo_dimension(), dof_priority=analyzer.tensor_space.dof_priority)))


def _node_major(analyzer: LagrangeFEMAnalyzer, F) -> np.ndarray:
    """把载荷向量统一换成节点优先排序, 以便比较两种自由度排序."""
    values = np.asarray(bm.to_numpy(F))
    GD = analyzer.disp_mesh.geo_dimension()
    if analyzer.tensor_space.dof_priority:
        return values.reshape(GD, -1).T.reshape(-1)
    return values


@pytest.mark.parametrize("dof_priority", [False, True])
def test_body_force_resultant(dof_priority: bool) -> None:
    analyzer = _bearing(dof_priority)
    xmin, xmax, ymin, ymax = analyzer.pde.domain
    area = (xmax - xmin) * (ymax - ymin)

    resultant = _resultant(analyzer, analyzer.assemble_body_force_vector())

    np.testing.assert_allclose(resultant, np.array(BODY_FORCE) * area, rtol=1e-12)


@pytest.mark.parametrize("dof_priority", [False, True])
def test_external_load_resultant_with_all_load_types(dof_priority: bool) -> None:
    analyzer = _bearing(dof_priority)
    problem = analyzer.pde
    xmin, xmax, ymin, ymax = problem.domain
    expected = (np.array(BODY_FORCE) * (xmax - xmin) * (ymax - ymin)
                + np.array([0.0, problem.t * (xmax - xmin)])
                + np.array(POINT_FORCE))

    resultant = _resultant(analyzer, analyzer.assemble_external_load())

    np.testing.assert_allclose(resultant, expected, rtol=1e-12)


@pytest.mark.parametrize("dof_priority", [False, True])
def test_line_traction_resultant(dof_priority: bool) -> None:
    analyzer = _cantilever(dof_priority)

    resultant = _resultant(analyzer, analyzer.assemble_external_load())

    np.testing.assert_allclose(resultant, [0.0, analyzer.pde.P, 0.0], rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("make", [_bearing, _cantilever])
def test_dof_orderings_give_the_same_loads(make) -> None:
    gd_first, dof_first = make(False), make(True)

    np.testing.assert_array_equal(_node_major(gd_first, gd_first.assemble_external_load()),
                                  _node_major(dof_first, dof_first.assemble_external_load()))
