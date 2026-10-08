"""``assemble_csr`` numpy 端累加方案 (bincount 代替 np.add.at) 的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.matrix import assemble_csr, build_csr_pattern
from soptx.fem.matrix.csr_pattern import _index_dtype, _iter_blocks
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.mesh import HexahedronMesh, TriangleMesh


def _space(kind, mesh_class, shape):
    """kind: 'scalar' 标量空间; 'gd' 分量交错 (dof_priority=False); 'dof' 分量分块 (dof_priority=True)."""
    GD = len(shape)
    mesh = mesh_class.from_box([0.0, 1.0] * GD, *shape)
    scalar = LagrangeFESpace(mesh, p=1, ctype='C')
    if kind == 'scalar':
        return scalar
    return TensorFunctionSpace(scalar, shape=(-1, GD) if kind == 'gd' else (GD, -1))


def _random_element_matrices(space, seed):
    NC = space.mesh.number_of_cells()
    ldof = int(np.asarray(space.cell_to_dof()).reshape(NC, -1).shape[1])
    return np.random.default_rng(seed).standard_normal((NC, ldof, ldof))


def _add_at_reference(K_e, pattern, scale=None):
    """逐分量块 np.add.at 累加的参考实现, 即修改之前的 numpy 路线."""
    buffer = np.zeros(pattern.nnz)
    for a, b, block in _iter_blocks(K_e, pattern.n_cells, pattern.scalar_ldof, pattern.dof_numel,
                                    pattern.dof_priority):
        np.add.at(buffer, np.asarray(pattern.slot_of(a, b)),
                  block if scale is None else block * scale[:, None, None])
    return buffer


CASES = [
    ('scalar', TriangleMesh, (4, 3)),
    ('gd', TriangleMesh, (4, 3)),
    ('dof', TriangleMesh, (4, 3)),
    ('gd', HexahedronMesh, (3, 2, 2)),
    ('dof', HexahedronMesh, (3, 2, 2)),
]


@pytest.mark.parametrize('kind, mesh_class, shape', CASES)
@pytest.mark.parametrize('with_scale', [False, True])
def test_scatter_plan_matches_add_at_bitwise(kind, mesh_class, shape, with_scale) -> None:
    bm.set_backend('numpy')
    space = _space(kind, mesh_class, shape)
    pattern = build_csr_pattern(space)
    K_e = _random_element_matrices(space, seed=0)
    scale = np.random.default_rng(1).uniform(0.0, 2.0, K_e.shape[0]) if with_scale else None

    values = np.array(assemble_csr(K_e, pattern, scale=scale).values)
    np.testing.assert_array_equal(values, _add_at_reference(K_e, pattern, scale))


def test_scatter_plan_is_built_once_and_reused() -> None:
    bm.set_backend('numpy')
    space = _space('gd', HexahedronMesh, (3, 2, 2))
    pattern = build_csr_pattern(space)
    assert pattern.scatter_plan is None

    assemble_csr(_random_element_matrices(space, seed=0), pattern)
    plan = pattern.scatter_plan
    values = np.array(assemble_csr(_random_element_matrices(space, seed=2), pattern).values)

    assert pattern.scatter_plan is plan
    np.testing.assert_array_equal(values, _add_at_reference(_random_element_matrices(space, seed=2), pattern))


def test_pytorch_backend_keeps_add_at_route_and_agrees() -> None:
    pytest.importorskip('torch')
    results = {}
    try:
        for backend in ('numpy', 'pytorch'):
            bm.set_backend(backend)
            space = _space('gd', HexahedronMesh, (3, 2, 2))
            pattern = build_csr_pattern(space)
            K_e = bm.tensor(_random_element_matrices(space, seed=0), dtype=bm.float64)
            results[backend] = np.asarray(bm.to_numpy(assemble_csr(K_e, pattern).values))
            if backend == 'pytorch':
                assert pattern.scatter_plan is None
    finally:
        bm.set_backend('numpy')

    np.testing.assert_allclose(results['pytorch'], results['numpy'], rtol=1e-13, atol=1e-13)


def test_index_dtype_switches_at_int32_bound() -> None:
    assert _index_dtype(10) is np.int32
    assert _index_dtype(2 ** 31) is np.int32          # 最大索引 2^31 - 1 仍可用 int32 表示
    assert _index_dtype(2 ** 31 + 1) is np.int64


def test_scatter_plan_targets_use_int32_for_small_patterns() -> None:
    bm.set_backend('numpy')
    space = _space('gd', HexahedronMesh, (3, 2, 2))
    pattern = build_csr_pattern(space)
    assemble_csr(_random_element_matrices(space, seed=0), pattern)
    sid, _, targets = pattern.scatter_plan

    assert sid.dtype == np.intp
    assert all(target.dtype == np.int32 for target in targets.values())
