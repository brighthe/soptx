"""``BilinearForm.__matmul__`` 免装配乘法的回归测试.

多列右端项 ``(B, gdof)`` 时输出按 ``(B, gdof)`` 布局, 散加却曾沿默认的 ``axis=0`` 进行,
结果落在批量轴上 (越界或错位).
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.bilinear_form import BilinearForm
from soptx.fem.integrators import LinearElasticIntegrator
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TriangleMesh


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _form_and_matrix(backend: str):
    """3x3 三角形网格上 P2 线弹性型的免装配双线性型, 及另行装配的全局矩阵."""
    bm.set_backend(backend)
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=3, ny=3)
    space = TensorFunctionSpace(scalar_space=LagrangeFESpace(mesh, p=2), shape=(-1, 2))
    material = IsotropicLinearElasticMaterial(
        lame_lambda=1.0, shear_modulus=0.5, hypothesis="plane_strain", enable_logging=False
    )

    def build():
        form = BilinearForm(space)
        form.add_integrator(LinearElasticIntegrator(material, method="standard"))
        return form

    matrix = build().assembly(format="csr", method="coalesce").to_scipy().toarray()
    return build(), matrix, space.number_of_global_dofs()


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_matmul_single_column(backend):
    """单列: 免装配乘法等于全局矩阵乘向量."""
    form, matrix, gdof = _form_and_matrix(backend)
    u = np.random.default_rng(0).standard_normal(gdof)
    v = bm.to_numpy(form @ bm.tensor(u))
    np.testing.assert_allclose(v, matrix @ u, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
def test_matmul_multiple_columns(backend):
    """多列 (B, gdof): 每一行等于全局矩阵乘对应的向量."""
    form, matrix, gdof = _form_and_matrix(backend)
    U = np.random.default_rng(1).standard_normal((3, gdof))
    V = bm.to_numpy(form @ bm.tensor(U))
    assert V.shape == (3, gdof)
    np.testing.assert_allclose(V, U @ matrix.T, rtol=1e-12, atol=1e-12)
