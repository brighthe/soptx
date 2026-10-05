"""稀疏张量中曾静默出错的接口的回归测试.

``bmat`` 不含 None 块时曾走 hstack/vstack 快路径: 单块列静默只返回首块, 单块行返回
列表; 组装路径还会因首块无非零元而重置块行列尺寸, 并改写调用方的块列表.
``CSRTensor.mul(CSRTensor)`` 与几个 ``reshape``/``ravel``/``flatten`` 函数体为空,
静默返回 None.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from soptx.backend import backend_manager as bm
from soptx.sparse import COOTensor, CSRTensor
from soptx.sparse.ops import bmat


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _csr(m: int, n: int, seed: int, density: float = 0.6) -> CSRTensor:
    """形状 ``(m, n)`` 的随机 CSR 矩阵."""
    return CSRTensor.from_scipy(sp.random(m, n, density=density, random_state=seed, format="csr"))


def _dense(a) -> np.ndarray:
    """把 soptx 或 scipy 稀疏矩阵转为稠密数组."""
    return a.to_scipy().toarray() if hasattr(a, "to_scipy") else a.toarray()


A, B, C, D = _csr(3, 2, 1), _csr(3, 4, 2), _csr(2, 2, 3), _csr(2, 4, 4)

LAYOUTS = {
    "1x1": [[A]],
    "1x2": [[A, B]],
    "2x1": [[A], [C]],
    "2x2": [[A, B], [C, D]],
    "with_none": [[A, None], [None, D]],
    "coo_blocks": [[A.tocoo(), B.tocoo()], [C.tocoo(), None]],
}


@pytest.mark.parametrize("layout", LAYOUTS, ids=list(LAYOUTS))
@pytest.mark.parametrize("fmt", ["csr", "coo"])
def test_bmat_matches_scipy(layout, fmt):
    """各种块布局的结果与 scipy.sparse.bmat 逐元素一致, 格式与形状正确."""
    blocks = LAYOUTS[layout]
    expected = sp.bmat([[None if b is None else b.to_scipy() for b in row] for row in blocks]).toarray()

    result = bmat(blocks, format=fmt)

    assert isinstance(result, CSRTensor if fmt == "csr" else COOTensor)
    assert tuple(int(s) for s in result.shape) == expected.shape
    np.testing.assert_array_equal(_dense(result), expected)


def test_bmat_does_not_mutate_blocks():
    """组装不改写调用方的块列表."""
    blocks = [[A, None], [None, D]]
    bmat(blocks)
    assert blocks[0][0] is A and blocks[1][1] is D


def test_bmat_first_block_without_nonzeros():
    """首个块没有非零元时, 后续块的尺寸不被重置."""
    empty = CSRTensor.from_scipy(sp.csr_matrix((3, 2)))
    result = bmat([[empty, B], [C, None]])
    expected = sp.bmat([[empty.to_scipy(), B.to_scipy()], [C.to_scipy(), None]]).toarray()
    np.testing.assert_array_equal(_dense(result), expected)


def test_bmat_rejects_undetermined_block_size():
    """某个块行全为 None 时无法确定行数, 应报错而不是给出错误形状."""
    with pytest.raises(ValueError, match="全为 None"):
        bmat([[A, B], [None, None]])


def test_bmat_dtype_applies_to_general_path():
    """含 None 块时 dtype 参数同样生效."""
    result = bmat([[A, None], [None, D]], dtype=bm.float32)
    assert result.values.dtype == np.float32


def test_unimplemented_sparse_methods_raise():
    """函数体曾为空的方法改为明确抛 NotImplementedError, 不再静默返回 None."""
    with pytest.raises(NotImplementedError):
        A.mul(A)
    with pytest.raises(NotImplementedError):
        A.reshape(6)
    with pytest.raises(NotImplementedError):
        A.ravel()
    with pytest.raises(NotImplementedError):
        A.flatten()
    with pytest.raises(NotImplementedError):
        A.tocoo().reshape(6)
