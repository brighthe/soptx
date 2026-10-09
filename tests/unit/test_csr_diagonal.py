"""CSRTensor.diagonal 的正确性: 与 scipy 一致, 重复索引求和, 非方阵取短边."""

import numpy as np
import scipy.sparse as sp

from soptx.backend import backend_manager as bm
from soptx.sparse import CSRTensor
from soptx.solvers.base import operator_diagonal


def _random_csr(m, n, seed):
    bm.set_backend('numpy')
    mat = sp.random(m, n, density=0.3, format='csr', random_state=seed)
    mat.sort_indices()
    return mat, CSRTensor.from_scipy(mat)


def test_diagonal_matches_scipy():
    mat, tensor = _random_csr(40, 40, 0)
    np.testing.assert_allclose(tensor.diagonal(), mat.diagonal())


def test_diagonal_sums_duplicates():
    bm.set_backend('numpy')
    crow = np.array([0, 3, 4], dtype=np.int64)
    col = np.array([0, 0, 1, 1], dtype=np.int64)
    values = np.array([1.0, 2.0, 5.0, 7.0])
    tensor = CSRTensor(crow, col, values, (2, 2))
    np.testing.assert_allclose(tensor.diagonal(), [3.0, 7.0])


def test_diagonal_rectangular():
    mat, tensor = _random_csr(30, 50, 1)
    diag = tensor.diagonal()
    assert diag.shape == (30, )
    np.testing.assert_allclose(diag, mat.diagonal())


def test_operator_diagonal_uses_csr_method(monkeypatch):
    """求解层取对角走 diagonal(), 不再经 tocoo."""
    _, tensor = _random_csr(20, 20, 2)
    expected = tensor.diagonal()

    def _forbidden(*args, **kwargs):
        raise AssertionError('operator_diagonal 不应再对 CSRTensor 调 tocoo')

    monkeypatch.setattr(tensor, 'tocoo', _forbidden)
    np.testing.assert_allclose(operator_diagonal(tensor), expected)
