"""局部刚度 Schur 的独立分块参考、稀疏拓扑与公共恢复回归."""

import numpy as np
import pytest
from scipy.sparse import issparse

from soptx.backend import backend_manager as bm
from soptx.fem.substructure.implicit import InteriorBlockSolver
from soptx.fem.substructure.local_assembly import build_local_assembly_map, assemble_local_stiffness
from soptx.fem.substructure.local_schur import (
    LocalSchurBlocks, LocalSchurOperator, build_local_block_patterns,
)


@pytest.mark.parametrize('contrast', [1.0, 1e-7])
@pytest.mark.parametrize('chunk', [1, 2])
def test_local_schur_against_independent_elimination(contrast, chunk):
    """独立全局消元验证共享边界散加、精确对角及内部恢复."""
    bm.set_backend('numpy')
    cells = np.array([[0, 1], [1, 2]], dtype=np.int64)
    mapping = build_local_assembly_map(cells, 3)
    internal, boundary = np.array([1]), np.array([0, 2])
    patterns = build_local_block_patterns(mapping, internal, boundary)
    # 每个单元使用不同系数, 参考矩阵带正定项, 避免测试依赖刚体约束处理.
    coefficients = np.array([[0.7, contrast], [0.3, 0.9]])
    element = coefficients[:, :, None, None] * np.array([[2., -1.], [-1., 2.]])
    local = assemble_local_stiffness(element, mapping)
    interface = np.array([0, 2, 4])
    interior = np.array([[1], [3]])
    local_interface = np.array([[0, 1], [1, 2]])
    global_maps = np.array([[0, 1, 2], [2, 3, 4]])
    full = np.zeros((5, 5))
    for indices, matrix in zip(global_maps, local):
        full[np.ix_(indices, indices)] += matrix
    ki = full[np.ix_(interior.ravel(), interior.ravel())]
    kib = full[np.ix_(interior.ravel(), interface)]
    reference = full[np.ix_(interface, interface)] - kib.T @ np.linalg.solve(ki, kib)

    caches = []
    for explicit in (True, False):
        blocks = LocalSchurBlocks(
            InteriorBlockSolver(interior, 5), interface, local_interface, patterns,
            keep_boundary=not explicit)
        timings = dict(internal_factorization=0., local_blocks=0.,
                       schur_formation=0., schur_diagonal=0.)
        assembled = np.zeros_like(reference)
        for start in range(0, 2, chunk):
            matrices = blocks.prepare_batch(
                start, local[start:start + chunk], internal, boundary,
                explicit=explicit, timings=timings, column_chunk=1)
            if explicit:
                for offset, matrix in enumerate(matrices):
                    ids = local_interface[start + offset]
                    assembled[np.ix_(ids, ids)] += matrix
        np.testing.assert_allclose(blocks.diagonal, reference.diagonal(), rtol=1e-13, atol=1e-14)
        assert all(issparse(block) for block in blocks.coupling)
        assert all(block.nnz == patterns[0].indices.size for block in blocks.coupling)
        if explicit:
            np.testing.assert_allclose(assembled, reference, rtol=1e-13, atol=1e-14)
            assert blocks.boundary is None
        else:
            operator = LocalSchurOperator(blocks, np.array([1, 2]))
            vector = np.array([0.2, -0.4])
            np.testing.assert_allclose(
                operator @ vector, reference[1:, 1:] @ vector, rtol=1e-13, atol=1e-14)
            assert operator.matvec_count == 1
            recovered = operator.recover(vector)
            np.testing.assert_allclose((full @ recovered)[interior.ravel()], 0., atol=1e-14)
        q = np.array([0., 0.2, -0.4])
        actual = blocks.recover(q)
        expected = np.zeros(5)
        expected[interface] = q
        expected[interior.ravel()] = -np.linalg.solve(ki, kib @ q)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        caches.append(blocks)
    np.testing.assert_array_equal(caches[0].interior.factors, caches[1].interior.factors)


def test_local_sparse_patterns_keep_structural_zeros():
    """连接关系内的零和微小条目均保留, 模式不随密度数值改变."""
    bm.set_backend('numpy')
    mapping = build_local_assembly_map(np.array([[0, 1], [1, 2]]), 3)
    coupling, boundary = build_local_block_patterns(mapping, np.array([1]), np.array([0, 2]))
    matrix = np.diag([2., 3., 4.])
    zero = coupling.extract(matrix)
    matrix[1, 0] = matrix[0, 1] = 1e-30
    small = coupling.extract(matrix)
    assert zero.nnz == small.nnz == 2
    np.testing.assert_array_equal(zero.indices, small.indices)
    np.testing.assert_array_equal(zero.indptr, small.indptr)
    assert small[0, 0] == 1e-30
    assert boundary.extract(matrix).nnz == 2


def test_local_patterns_reject_incomplete_partition():
    """局部编号不完整时在构造模式阶段拒绝."""
    bm.set_backend('numpy')
    mapping = build_local_assembly_map(np.array([[0, 1], [1, 2]]), 3)
    with pytest.raises(ValueError, match='完整'):
        build_local_block_patterns(mapping, np.array([1]), np.array([0]))


def test_internal_solve_requires_ready_factor():
    """内部回代必须使用已完成分解且维数相符的块."""
    solver = InteriorBlockSolver(np.array([[1]]), 3)
    with pytest.raises(ValueError, match='尚未分解'):
        solver.solve_block(0, np.ones(1))
    solver.factor_batch(0, np.array([[[2.]]]))
    with pytest.raises(ValueError, match='形状'):
        solver.solve_block(0, np.ones(2))