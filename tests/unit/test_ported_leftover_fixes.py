"""known-issues "移植后遗留" 中已修各项的回归测试.

覆盖: ``process_coef_func`` 的网格检查位置, ``get_semilinear_coef(coef=None)``,
``ConstIntegrator`` 的缓存开关, 模式张量加稠密张量, ``CSRTensor.sum`` 的轴约定,
numpy 后端张量积重心坐标的 ``bc_to_points``, 三棱柱加密的 ``returnim``, 以及已删除的
``splitter`` / ``from_box(threshold=...)`` 参数.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.fem.bilinear_form import BilinearForm
from soptx.fem.coef import process_coef_func
from soptx.fem.functional import get_semilinear_coef
from soptx.fem.integrator import ConstIntegrator
from soptx.fem.integrators import MassIntegrator
from soptx.functionspace import LagrangeFESpace
from soptx.mesh import QuadrangleMesh, TriangleMesh
from soptx.sparse import COOTensor, CSRTensor


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def test_process_coef_func_checks_mesh_only_for_cartesian():
    """重心坐标函数不需要网格; 直角坐标函数缺网格时给出明确的 RuntimeError."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
    bcs = bm.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=bm.float64)
    index = bm.arange(mesh.number_of_cells())

    def bary(bcs, index=None):
        return bcs[..., 0] + 0.0 * index[:, None]

    value = process_coef_func(bary, bcs=bcs, mesh=None, etype="cell", index=index)
    np.testing.assert_allclose(bm.to_numpy(value), 1 / 3)

    @cartesian
    def cart(points):
        return points[..., 0]

    with pytest.raises(RuntimeError, match="mesh"):
        process_coef_func(cart, bcs=bcs, mesh=None, etype="cell", index=index)
    value = process_coef_func(cart, bcs=bcs, mesh=mesh, etype="cell", index=index)
    np.testing.assert_allclose(bm.to_numpy(value), bm.to_numpy(mesh.bc_to_point(bcs)[..., 0]))


def test_semilinear_coef_none_returns_value():
    """coef 为 None 时原样返回 value."""
    value = bm.tensor([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bm.to_numpy(get_semilinear_coef(value, None)), [1.0, 2.0, 3.0])


def test_const_integrator_does_not_keep_data():
    """ConstIntegrator 不再因旧签名意外开启缓存."""
    assert ConstIntegrator(bm.zeros((2, 3, 3)))._keep_data is False


def _random_csr(seed: int) -> CSRTensor:
    return CSRTensor.from_scipy(sp.random(4, 5, density=0.5, random_state=seed, format="csr"))


@pytest.mark.parametrize("fmt", ["csr", "coo"])
def test_pattern_plus_dense(fmt):
    """模式张量加稠密张量: 每个非零位置加 1."""
    A = _random_csr(0)
    pattern = CSRTensor(A.crow, A.col, None, A.sparse_shape)
    if fmt == "coo":
        pattern = COOTensor(A.tocoo().indices, None, A.sparse_shape)
    dense = np.arange(20.0).reshape(4, 5)
    result = bm.to_numpy(pattern.add(bm.tensor(dense)))
    expected = dense + (A.to_scipy().toarray() != 0)
    np.testing.assert_array_equal(result, expected)


def test_csr_sum_follows_numpy_convention():
    """axis=0 得各列之和, axis=1 得各行之和; 非法 axis 报错."""
    A = _random_csr(1)
    dense = A.to_scipy().toarray()
    np.testing.assert_allclose(bm.to_numpy(A.sum(axis=0)), dense.sum(axis=0))
    np.testing.assert_allclose(bm.to_numpy(A.sum(axis=1)), dense.sum(axis=1))
    with pytest.raises(ValueError):
        A.sum(axis=2)


def test_numpy_bc_to_points_with_tensor_product_bcs():
    """numpy 后端接受张量积重心坐标元组, 与 pytorch 后端结果一致."""
    results = {}
    for backend in ("numpy", "pytorch"):
        bm.set_backend(backend)
        mesh = QuadrangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
        bcs = mesh.quadrature_formula(2).get_quadrature_points_and_weights()[0]
        assert isinstance(bcs, tuple)
        node, cell = mesh.entity("node"), mesh.entity("cell")
        results[backend] = bm.to_numpy(bm.bc_to_points(bcs, node, cell))
    np.testing.assert_allclose(results["numpy"], results["pytorch"], rtol=0, atol=1e-14)


def test_prism_refine_rejects_returnim():
    """三棱柱一致加密尚无延拓矩阵, returnim=True 时明确报错而非返回空列表."""
    from soptx.mesh.transform.uniform import uniform_refine_prism

    with pytest.raises(NotImplementedError):
        uniform_refine_prism(None, n=1, returnim=True)


def test_removed_parameters():
    """splitter 分块装配与 from_box 的 threshold 参数已删除."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
    form = BilinearForm(LagrangeFESpace(mesh, p=1))
    with pytest.raises(TypeError):
        form.add_integrator(MassIntegrator(), splitter=2)
    with pytest.raises(TypeError):
        TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1, threshold=None)
