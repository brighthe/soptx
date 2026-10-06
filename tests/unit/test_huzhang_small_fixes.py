"""胡张相关小问题的回归测试.

二维 ``boundary_interpolate`` 对不支持的 ``gd`` 曾抛含义不明的 AttributeError (报错
路径本身取 ``int.shape``); MMA 在胡张分析器下记录应力曾在迭代中途报 KeyError.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import HuZhangMFEMAnalyzer
from soptx.functionspace import HuZhangFESpace
from soptx.mesh import TriangleMesh
from soptx.topology.optimizers.mma import MMAOptimizer


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _space():
    return HuZhangFESpace(TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2), p=2)


def test_boundary_interpolate_rejects_scalar_gd():
    """标量 gd 没有可投影的分量, 明确报 ValueError."""
    with pytest.raises(ValueError, match="标量"):
        _space().boundary_interpolate(1.0)


def test_boundary_interpolate_rejects_wrong_component_count():
    """gd 的分量数既不是 2 也不是 3 时, 报错信息给出实际分量数."""
    with pytest.raises(ValueError, match="得到 4"):
        _space().boundary_interpolate(bm.tensor([1.0, 0.0, 0.0, 0.0]))


def test_boundary_interpolate_constant_matches_callable():
    """常张量 gd 与返回同一常值的函数给出相同结果."""
    space = _space()
    traction = [0.3, -1.2]
    uh_const, flag_const = space.boundary_interpolate(traction)
    uh_func, flag_func = space.boundary_interpolate(
        lambda points: bm.broadcast_to(bm.tensor(traction), points.shape)
    )
    np.testing.assert_array_equal(bm.to_numpy(flag_const), bm.to_numpy(flag_func))
    np.testing.assert_allclose(bm.to_numpy(uh_const), bm.to_numpy(uh_func), rtol=0, atol=1e-15)


def test_mma_store_stress_rejects_huzhang_analyzer():
    """胡张分析器下要求记录应力时, 在开始迭代前明确报错."""

    class _Objective:
        _analyzer = object.__new__(HuZhangMFEMAnalyzer)

    optimizer = object.__new__(MMAOptimizer)
    optimizer._objective = _Objective()
    with pytest.raises(NotImplementedError, match="Lagrange"):
        optimizer.optimize(bm.zeros(4), bm.zeros(4), is_store_stress=True)
