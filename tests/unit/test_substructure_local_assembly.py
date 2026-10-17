"""独立局部组装内核的矩阵散加与兼容包装检查."""

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.substructure.local_assembly import build_local_assembly_map, assemble_local_stiffness
from soptx.fem.substructure import SubstructurePrototype


@pytest.mark.parametrize('backend_name', ['numpy', 'pytorch'])
def test_local_assembly_matches_explicit_scatter(backend_name):
    """不依赖材料公式, 以显式逐项散加验证共享自由度的累加."""
    if backend_name == 'pytorch':
        pytest.importorskip('torch')
    bm.set_backend(backend_name)
    try:
        dofs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        values = np.arange(16, dtype=np.float64).reshape(2, 2, 2, 2)
        expected = np.zeros((2, 3, 3))
        for batch in range(2):
            for cell in range(2):
                for row in range(2):
                    for col in range(2):
                        expected[batch, dofs[cell, row], dofs[cell, col]] += values[batch, cell, row, col]
        mapping = build_local_assembly_map(bm.asarray(dofs), 3)
        matrices = bm.asarray(values.copy())
        actual = assemble_local_stiffness(matrices, mapping)
        np.testing.assert_array_equal(bm.to_numpy(actual), expected)
        np.testing.assert_array_equal(bm.to_numpy(matrices), values)
        with pytest.raises(ValueError, match='KE_chunk'):
            assemble_local_stiffness(matrices[:, :1], mapping)
    finally:
        bm.set_backend('numpy')


def test_legacy_wrapper_preserves_explicit_backend():
    """全局后端为 NumPy 时仍允许旧训练包装使用独立 PyTorch 后端."""
    torch = pytest.importorskip('torch')
    from copy import copy
    from soptx.backend.manager import BackendManager

    bm.set_backend('numpy')
    prototype = SubstructurePrototype(cell_size=(1., 1.), n_fine=(2, 2), E_base=1., nu=.3)
    rho = np.array([[.2, .4, .6, .8]], dtype=np.float64)
    expected = prototype._assemble_chunk(rho)
    tensor_prototype = copy(prototype)
    tensor_prototype.KE_unit = torch.as_tensor(prototype.KE_unit.copy())
    tensor_prototype._scatter_index = torch.as_tensor(prototype._scatter_index.copy())
    backend = BackendManager(default_backend='pytorch').get_current_backend()
    actual = tensor_prototype._assemble_chunk(torch.as_tensor(rho), backend=backend)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-14)
