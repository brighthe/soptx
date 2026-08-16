"""胡张空间在单元子集上求值的回归测试.

``HuZhangDof2d.cell_to_dof`` 曾经忽略 ``index`` 参数, 直接返回全网格的自由度
映射; 而 ``basis(bc, index)`` 是按子集计算的, 于是 ``value(uh, bc, index)``
把子集的基函数与全网格的自由度映射配在一起 —— 只有 ``index`` 恰为全集时才
碰巧正确. 边界合力这类只在少量单元上求值的诊断因此会得到完全错误的结果.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem import create_huzhang_checkerboard_mesh
from soptx.fem.spaces import HuZhangFESpace


def _relaxed_space(degree: int) -> Any:
    """构造与角点松弛一致的胡张空间; 角点松弛要求显式给出几何角点坐标.

    HuZhangFESpace 是工厂类, __new__ 返回 2d/3d 实现, 静态类型上看不到方法,
    因此返回类型放宽为 Any.
    """
    bm.set_backend("numpy")
    mesh = create_huzhang_checkerboard_mesh(box=(0.0, 1.0, 0.0, 1.0), nx=2, ny=2)
    corners = bm.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=bm.float64
    )
    return HuZhangFESpace(mesh=mesh, p=degree, use_relaxation=True, corners=corners)


@pytest.mark.parametrize("degree", [2, 3])
def test_value_on_a_cell_subset_matches_the_full_mesh(degree: int) -> None:
    """子集求值必须与全网格求值后再切片逐位一致."""
    space = _relaxed_space(degree)

    coefficients = bm.arange(space.number_of_global_dofs(), dtype=bm.float64)
    bcs = bm.array([[0.5, 0.25, 0.25], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=bm.float64)

    # 取一个既非全集、也非前缀的单元子集, 顺序刻意打乱
    subset = bm.array([5, 1, 6], dtype=bm.int32)
    full = space.value(coefficients, bcs)
    subset_values = space.value(coefficients, bcs, index=subset)

    np.testing.assert_allclose(
        bm.to_numpy(subset_values), bm.to_numpy(full)[bm.to_numpy(subset)], atol=1.0e-14
    )


def test_cell_to_dof_respects_the_index_argument() -> None:
    """cell_to_dof(index) 必须返回对应子集的行, 而不是整张映射表."""
    space = _relaxed_space(3)

    subset = bm.array([7, 0], dtype=bm.int32)
    full = space.cell_to_dof()
    selected = space.cell_to_dof(index=subset)

    assert selected.shape[0] == subset.shape[0]
    np.testing.assert_array_equal(
        bm.to_numpy(selected), bm.to_numpy(full)[bm.to_numpy(subset)]
    )
