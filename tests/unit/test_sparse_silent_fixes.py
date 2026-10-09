"""稀疏张量中曾静默出错的接口的回归测试.

``bmat`` 不含 None 块时曾走 hstack/vstack 快路径: 单块列静默只返回首块, 单块行返回
列表; 组装路径还会因首块无非零元而重置块行列尺寸, 并改写调用方的块列表.
``CSRTensor.mul(CSRTensor)`` 与几个 ``reshape``/``ravel``/``flatten`` 函数体为空,
静默返回 None. ``COOTensor.tocsr`` (``CSRTensor.T`` 经它中转) 曾产出 int64 的 ``crow``
配原 dtype 的 ``col``, pytorch 后端在 CUDA 上的 SpMV 因此越界访问.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

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


def _rect_scipy(m: int, n: int, seed: int) -> sp.csr_matrix:
    """形状 ``(m, n)`` 的随机 scipy CSR 矩阵, 索引为 int32."""
    matrix = sp.random(m, n, density=0.2, random_state=seed, format="csr")
    assert matrix.indices.dtype == np.int32
    return matrix


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_tocsr_index_dtype_consistent(backend):
    """经 COO 中转的转置: crow 与 col 同为 COO 索引的 dtype (此处 int32), 数值与 scipy 一致."""
    bm.set_backend(backend)
    P_sp = _rect_scipy(40, 12, 5)
    PT = CSRTensor.from_scipy(P_sp).T
    assert PT.crow.dtype == PT.col.dtype
    assert str(PT.crow.dtype).endswith("int32")
    np.testing.assert_array_equal(PT.to_scipy().toarray(), P_sp.T.toarray())


def test_tocsr_int64_indices_unchanged():
    """COO 索引为 int64 时, tocsr 后 crow 与 col 仍为 int64."""
    indices = np.array([[0, 2, 1, 2], [1, 0, 2, 2]], dtype=np.int64)
    csr = COOTensor(indices, np.array([1.0, 2.0, 3.0, 4.0]), (3, 3)).tocsr()
    assert csr.crow.dtype == np.int64 and csr.col.dtype == np.int64
    expected = sp.coo_matrix(([1.0, 2.0, 3.0, 4.0], (indices[0], indices[1])), shape=(3, 3))
    np.testing.assert_array_equal(csr.to_scipy().toarray(), expected.toarray())


def test_pytorch_csr_spmm_rejects_mixed_index_dtype():
    """pytorch 后端的 csr_spmm 拒绝 dtype 不一致的 crow 与 col, 不交给 torch 去越界."""
    bm.set_backend("pytorch")
    P_sp = _rect_scipy(40, 12, 6)
    crow = torch.from_numpy(P_sp.indptr.astype(np.int64))
    col = torch.from_numpy(P_sp.indices)
    mixed = CSRTensor(crow, col, torch.from_numpy(P_sp.data), P_sp.shape)
    with pytest.raises(ValueError, match="dtype 须一致"):
        mixed @ torch.ones(12, dtype=torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU 支持")
def test_transpose_matvec_cuda():
    """CUDA 上转置 CSR 的 matvec (多重网格的限制步) 与 scipy 一致, 不再越界访问."""
    bm.set_backend("pytorch")
    P_sp = _rect_scipy(400, 50, 7)
    r = np.random.default_rng(7).standard_normal(400)
    PT = CSRTensor.from_scipy(P_sp).device_put("cuda").T
    result = PT @ torch.from_numpy(r).to("cuda")
    torch.cuda.synchronize()
    np.testing.assert_allclose(bm.to_numpy(result), P_sp.T @ r, rtol=1e-13, atol=1e-13)
