"""后端中曾静默出错或行为不一致之处的回归测试.

numpy 后端的 ``unstack`` 曾用 ``np.split``, 拆出的切片保留被拆轴, 与 array API 及
pytorch 后端不一致; pytorch 后端的 ``apply_along_axis`` 在 ``axis != 0`` 时并不按该轴
切片; pytorch 后端的 ``insert`` 残留调试 ``print``.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_unstack_consistent_across_backends(axis):
    """两个后端的 unstack 都去掉被拆轴, 返回元组, 切片数值一致."""
    data = np.arange(24.0).reshape(2, 3, 4)
    results = {}
    for backend in ("numpy", "pytorch"):
        bm.set_backend(backend)
        parts = bm.unstack(bm.tensor(data), axis=axis)
        assert isinstance(parts, tuple)
        results[backend] = [bm.to_numpy(p) for p in parts]

    expected = [np.take(data, i, axis=axis) for i in range(data.shape[axis])]
    for backend, parts in results.items():
        assert len(parts) == len(expected), backend
        for got, want in zip(parts, expected):
            np.testing.assert_array_equal(got, want)


def test_pytorch_apply_along_axis_removed():
    """pytorch 后端不再提供行为错误的 apply_along_axis, 调用即得到明确的错误."""
    bm.set_backend("pytorch")
    with pytest.raises(AttributeError):
        bm.apply_along_axis


def test_pytorch_insert_is_silent(capsys):
    """pytorch 后端的 insert 不再向标准输出打印调试信息."""
    bm.set_backend("pytorch")
    x = bm.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = bm.insert(x, 1, bm.tensor([9.0, 9.0]), axis=0)
    np.testing.assert_array_equal(bm.to_numpy(result), [[1, 2], [9, 9], [3, 4]])
    assert capsys.readouterr().out == ""
