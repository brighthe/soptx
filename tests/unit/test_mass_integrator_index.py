"""``MassIntegrator`` 按 ``index`` 选单元的回归测试.

``to_global_dof`` 曾返回全体单元的映射, 而 ``assembly`` 只算 ``index`` 选中的单元,
带 ``index`` 装配时 ``BilinearForm.check_local_shape`` 报 ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.bilinear_form import BilinearForm
from soptx.fem.integrators import MassIntegrator
from soptx.functionspace import LagrangeFESpace
from soptx.mesh import TriangleMesh


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def test_index_selects_cells():
    """带 index 的全局质量矩阵等于只由选中单元的局部矩阵组装的结果."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=3, ny=3)
    space = LagrangeFESpace(mesh, p=2)
    index = bm.tensor([0, 4, 7, 11], dtype=bm.int64)

    integrator = MassIntegrator(index=index)
    np.testing.assert_array_equal(
        bm.to_numpy(integrator.to_global_dof(space)),
        bm.to_numpy(space.cell_to_dof())[[0, 4, 7, 11]],
    )

    form = BilinearForm(space)
    form.add_integrator(integrator)
    matrix = form.assembly(format="csr", method="coalesce").to_scipy().toarray()

    full = bm.to_numpy(MassIntegrator().assembly(space))
    cell2dof = bm.to_numpy(space.cell_to_dof())
    expected = np.zeros_like(matrix)
    for c in (0, 4, 7, 11):
        expected[np.ix_(cell2dof[c], cell2dof[c])] += full[c]
    np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-15)
