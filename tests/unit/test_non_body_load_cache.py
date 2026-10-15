"""纯集中力的非体力载荷向量缓存的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.analyzers import lagrange_fem_analyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import CantileverBottomEdgeLine3d, FullMBBBeamLine3d
from soptx.problems.loads import PointForceLoad
from soptx.topology.interpolation import MaterialInterpolationScheme


class _PointLoadMBB3d(FullMBBBeamLine3d):
    """顶面中心受集中力的三维 MBB 梁: 本测试检验纯集中力载荷的缓存."""

    def loads(self):
        x0, x1, _, y1, z0, z1 = self.domain
        return (PointForceLoad(point=((x0 + x1) / 2.0, y1, (z0 + z1) / 2.0), vector=(0.0, self.P, 0.0)),)


def _analyzer(problem, mesh, operator_level):
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis='3D')
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3.0, 'void_youngs_modulus': 1e-7, 'target_variables': ['E']})
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1,
                               integration_order=2, assembly_method='fast', operator_level=operator_level,
                               solve_method='cg', topopt_algorithm='density_based',
                               interpolation_scheme=interpolation)


def _mbb(operator_level):
    bm.set_backend('numpy')
    grid, domain = (6, 2, 3), (0.0, 6.0, 0.0, 2.0, 0.0, 3.0)
    problem = _PointLoadMBB3d(domain=domain)
    return _analyzer(problem, HexahedronMesh.from_box(list(domain), *grid), operator_level), problem


def _count_assembly(monkeypatch, analyzer):
    """统计分析器装配非体力载荷的次数 (替换分析器模块里引用的装配函数)."""
    calls = []
    original = lagrange_fem_analyzer.assemble_non_body_loads

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lagrange_fem_analyzer, 'assemble_non_body_loads', counted)
    return calls


@pytest.mark.parametrize('operator_level', ['fa', 'ea'])
def test_point_force_loads_are_assembled_once_and_reused_bitwise(monkeypatch, operator_level) -> None:
    analyzer, _ = _mbb(operator_level)
    calls = _count_assembly(monkeypatch, analyzer)
    density = bm.full((analyzer.tensor_space.mesh.number_of_cells(), ), 0.5, dtype=bm.float64)
    K = analyzer.assemble_stiff_matrix(rho_val=density)

    first = np.asarray(analyzer.apply_bc(K, analyzer.assemble_body_force_vector())[1][:]).copy()
    second = np.asarray(analyzer.apply_bc(K, analyzer.assemble_body_force_vector())[1][:])
    loads = analyzer._non_body_loads_by_boundary_type()

    assert len(calls) == 1
    np.testing.assert_array_equal(second, first)
    assert hasattr(loads, 'space')                       # 缓存命中时仍返回 Function
    # 返回副本: 改写它不污染缓存
    loads[:] = 123.0
    assert not np.any(np.asarray(analyzer._non_body_loads_by_boundary_type()[:]) == 123.0)


def test_changed_point_force_is_reassembled(monkeypatch) -> None:
    analyzer, problem = _mbb('fa')
    calls = _count_assembly(monkeypatch, analyzer)
    before = np.asarray(analyzer._non_body_loads_by_boundary_type()[:]).copy()
    problem._P = 2.0 * problem.P                       # 模拟载荷被改动
    after = np.asarray(analyzer._non_body_loads_by_boundary_type()[:])

    assert len(calls) == 2
    np.testing.assert_allclose(after, 2.0 * before, rtol=0.0, atol=1e-15)


def test_line_traction_loads_are_not_cached(monkeypatch) -> None:
    bm.set_backend('numpy')
    domain = (0.0, 6.0, 0.0, 2.0, 0.0, 2.0)
    analyzer = _analyzer(CantileverBottomEdgeLine3d(domain=domain),
                         HexahedronMesh.from_box(list(domain), 6, 2, 2), 'fa')
    calls = _count_assembly(monkeypatch, analyzer)
    first = np.asarray(analyzer._non_body_loads_by_boundary_type()[:]).copy()
    second = np.asarray(analyzer._non_body_loads_by_boundary_type()[:])

    assert len(calls) == 2 and analyzer._non_body_cache is None
    np.testing.assert_array_equal(second, first)
