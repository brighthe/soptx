"""接口 pattern 装配与独立 SciPy COO 参照的数值及生命周期检查."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from fealpy.backend import backend_manager as bm
from soptx.fem.substructure import GlobalAssembler, build_substructures


@pytest.fixture(autouse=True)
def numpy_backend():
    """各测试使用 NumPy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def setup_case(dim):
    """使用小网格和非对称局部矩阵暴露分量排序错误."""
    assembler = GlobalAssembler((1.0,) * dim, (2,) * dim, (2,) * dim, E_base=1.0, nu=0.3)
    _, meshes, _ = build_substructures(assembler)
    n_b = meshes[0].n_b
    values = np.random.default_rng(41).normal(size=(len(meshes), n_b, n_b))
    return assembler, meshes, values


def reference(assembler, meshes, values):
    """仅借用几何映射, 数值合并完全由 SciPy 完成."""
    global_dofs = assembler.build_interface_dofs(meshes)
    mapping = np.asarray(assembler.interface_indices(meshes, global_dofs))
    shape = values.shape
    rows = np.broadcast_to(mapping[:, :, None], shape).ravel()
    cols = np.broadcast_to(mapping[:, None, :], shape).ravel()
    return coo_matrix((values.ravel(), (rows, cols)), shape=(len(global_dofs),) * 2).tocsr()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_interface_matches_independent_coo(dim, chunk_size):
    """不同分块均与独立 COO 累加一致."""
    assembler, meshes, values = setup_case(dim)
    system = assembler.assemble_interface_system(meshes, SimpleNamespace(K_s=values), chunk_size=chunk_size)
    np.testing.assert_allclose(system.stiffness.to_scipy().toarray(),
                               reference(assembler, meshes, values).toarray(), atol=1e-12, rtol=1e-12)


def test_reassembly_preserves_old_matrix_and_handles_reordering():
    """密度变化不覆盖旧矩阵, 子结构重排不能误用槽位."""
    assembler, meshes, values = setup_case(2)
    first = assembler.assemble_interface_system(meshes, SimpleNamespace(K_s=values))
    cached_pattern = assembler._interface_pattern_cache[1]
    assert cached_pattern.buffer is None
    snapshot = first.stiffness.to_scipy().toarray().copy()
    second_values = values * 0.3
    second = assembler.assemble_interface_system(meshes, SimpleNamespace(K_s=second_values))
    np.testing.assert_array_equal(first.stiffness.to_scipy().toarray(), snapshot)
    np.testing.assert_allclose(second.stiffness.to_scipy().toarray(), snapshot * 0.3, atol=1e-12)
    assert assembler._interface_pattern_cache[1] is cached_pattern
    reordered = list(reversed(meshes))
    third = assembler.assemble_interface_system(reordered, SimpleNamespace(K_s=values))
    assert assembler._interface_pattern_cache[1] is not cached_pattern
    np.testing.assert_allclose(third.stiffness.to_scipy().toarray(),
                               reference(assembler, reordered, values).toarray(), atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(first.stiffness.to_scipy().toarray(), snapshot)


@pytest.mark.parametrize("dim", [2, 3])
def test_streamed_interface_matches_coo(dim):
    """流式缩聚块保留末尾短批并匹配独立参照."""
    assembler, meshes, values = setup_case(dim)
    calls = []

    def chunk(start, stop):
        calls.append((start, stop))
        return values[start:stop]

    condensor = SimpleNamespace(K_s=None, get_chunk_stiffness=chunk)
    system = assembler.assemble_interface_system(meshes, condensor, chunk_size=3)
    assert calls == [(i, min(i + 3, len(meshes))) for i in range(0, len(meshes), 3)]
    np.testing.assert_allclose(system.stiffness.to_scipy().toarray(),
                               reference(assembler, meshes, values).toarray(), atol=1e-12, rtol=1e-12)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("priority", [False, True])
def test_explicit_dofmap_pattern(dim, priority):
    """显式标量映射按两种分量顺序展开后匹配独立 COO."""
    from soptx.fem.matrix.csr_pattern import build_csr_pattern_from_dofmap, assemble_csr
    scalar = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    if priority:
        dofs = (scalar[:, None, :] + np.arange(dim)[None, :, None] * 4).reshape(2, -1)
    else:
        dofs = (scalar[:, :, None] * dim + np.arange(dim)[None, None, :]).reshape(2, -1)
    values = np.random.default_rng(19).normal(size=(2, 3 * dim, 3 * dim))
    pattern = build_csr_pattern_from_dofmap(
        scalar, 4 * dim, dof_numel=dim, dof_priority=priority, allocate_buffer=False,
    )
    assert pattern.buffer is None
    matrix = assemble_csr(values, pattern).to_scipy()
    rows = np.broadcast_to(dofs[:, :, None], values.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], values.shape).ravel()
    expected = coo_matrix((values.ravel(), (rows, cols)), shape=(4 * dim,) * 2).tocsr()
    np.testing.assert_allclose(matrix.toarray(), expected.toarray(), atol=1e-12, rtol=1e-12)