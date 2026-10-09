"""``soptx.fem.matrix.SymmetricElimination`` 的单元测试.

1. CSR 保结构消元与稠密的删除行列参考一致, 共用原骨架且不改写原值;
2. 骨架与掩码不变时复用槽位缓存, 掩码变了就重算;
3. COO 消元与同一参考一致;
4. 受约束行缺对角元时报错.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.matrix import SymmetricElimination
from soptx.sparse import COOTensor, CSRTensor


@pytest.fixture(autouse=True)
def reset_backend():
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _dense_matrix(n: int = 6, seed: int = 0) -> np.ndarray:
    """带若干结构零的对称矩阵, 对角全非零."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 2.0, (n, n))
    a = a + a.T
    a[np.abs(np.subtract.outer(np.arange(n), np.arange(n))) > 2] = 0.0
    return a


def _csr(a: np.ndarray) -> CSRTensor:
    rows, cols = np.nonzero(a)
    crow = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=a.shape[0]))])
    return CSRTensor(bm.tensor(crow), bm.tensor(cols), bm.tensor(a[rows, cols]), a.shape)


def _coo(a: np.ndarray) -> COOTensor:
    rows, cols = np.nonzero(a)
    return COOTensor(bm.tensor(np.stack([rows, cols])), bm.tensor(a[rows, cols]), a.shape)


def _reference(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[mask, :] = 0.0
    out[:, mask] = 0.0
    out[mask, mask] = 1.0
    return out


MASK = np.array([True, False, False, True, False, True])


def test_csr_matches_row_column_deletion() -> None:
    a = _dense_matrix()
    matrix = _csr(a)
    original = np.array(matrix.values)

    result = SymmetricElimination().apply(matrix, bm.tensor(MASK))

    np.testing.assert_array_equal(result.to_scipy().toarray(), _reference(a, MASK))
    np.testing.assert_array_equal(np.asarray(matrix.values), original)
    assert result.crow is matrix.crow and result.col is matrix.col


def test_csr_slots_are_cached_per_skeleton_and_mask() -> None:
    a = _dense_matrix()
    matrix = _csr(a)
    elimination = SymmetricElimination()

    elimination.apply(matrix, bm.tensor(MASK))
    cache = elimination._cache
    rescaled = CSRTensor(matrix.crow, matrix.col, 2.0 * matrix.values, matrix.sparse_shape)
    result = elimination.apply(rescaled, bm.tensor(MASK))
    assert elimination._cache is cache
    np.testing.assert_array_equal(result.to_scipy().toarray(), _reference(2.0 * a, MASK))

    other = ~MASK
    result = elimination.apply(matrix, bm.tensor(other))
    assert elimination._cache is not cache
    np.testing.assert_array_equal(result.to_scipy().toarray(), _reference(a, other))


def test_coo_matches_row_column_deletion() -> None:
    a = _dense_matrix(seed=1)

    result = SymmetricElimination().apply(_coo(a), bm.tensor(MASK))

    assert isinstance(result, COOTensor)
    np.testing.assert_array_equal(result.to_scipy().toarray(), _reference(a, MASK))


def test_missing_diagonal_is_rejected() -> None:
    a = _dense_matrix()
    a[3, 3] = 0.0

    with pytest.raises(RuntimeError, match="对角元"):
        SymmetricElimination().apply(_csr(a), bm.tensor(MASK))
