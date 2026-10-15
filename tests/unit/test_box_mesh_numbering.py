# -*- coding: utf-8 -*-
"""``from_box`` 单元编号约定单元测试.

测试覆盖范围:
1. 单元 e 位于第 e // N_k 个格子, 格子按 x 最慢、最后一轴最快的字典序排列;
2. 前 N_k 个单元的局部顶点偏移等于 docstring 中写明的模板;
3. 平移类约定: 每个单元的相对顶点坐标等于其代表单元 e % N_k 的 (只差舍入).
"""

import numpy as np
import pytest
from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh


# 各轴格子数互不相同, 区域不对称, 以便暴露轴序错误
BOX_2D = [0.0, 2.0, -1.0, 0.5]
BOX_3D = [0.0, 2.0, -1.0, 0.5, 1.0, 1.75]
# 各网格前 N_k 个单元的局部顶点相对格子最小角点的偏移 (以格子数计), 与 from_box 的 docstring 一致
TEMPLATES = {
    TriangleMesh: (
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 1), (0, 1)),
    ),
    QuadrangleMesh: (
        ((0, 0), (1, 0), (1, 1), (0, 1)),
    ),
    TetrahedronMesh: (
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1)),
        ((0, 0, 0), (1, 0, 1), (1, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)),
        ((0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
        ((0, 1, 0), (1, 1, 1), (0, 1, 1), (1, 0, 0)),
    ),
    HexahedronMesh: (
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    ),
}
CASES = [
    (TriangleMesh, BOX_2D, (3, 2)),
    (QuadrangleMesh, BOX_2D, (3, 2)),
    (TetrahedronMesh, BOX_3D, (3, 2, 4)),
    (HexahedronMesh, BOX_3D, (3, 2, 4)),
]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _mesh_arrays(mesh_class, box, shape):
    """生成网格并返回节点、单元与格子边长."""
    mesh = mesh_class.from_box(box, *shape)
    node = bm.to_numpy(mesh.entity('node'))
    cell = bm.to_numpy(mesh.entity('cell'))
    lower = np.asarray(box[0::2])
    spacing = (np.asarray(box[1::2]) - lower) / np.asarray(shape)
    return node, cell, lower, spacing


@pytest.mark.parametrize("mesh_class,box,shape", CASES)
def test_cells_follow_lexicographic_grid_order(mesh_class, box, shape):
    """单元 e 的最小角点落在第 e // N_k 个格子上, 格子按 x 最慢的字典序排列."""
    node, cell, lower, spacing = _mesh_arrays(mesh_class, box, shape)
    num_classes = len(TEMPLATES[mesh_class])
    assert cell.shape[0] == num_classes * int(np.prod(shape))

    grid = np.stack(np.unravel_index(np.arange(cell.shape[0]) // num_classes, shape), axis=1)
    corner = node[cell].min(axis=1)
    np.testing.assert_allclose(corner, lower + grid * spacing, rtol=0, atol=1e-13)


@pytest.mark.parametrize("mesh_class,box,shape", CASES)
def test_first_cells_match_documented_templates(mesh_class, box, shape):
    """前 N_k 个单元的局部顶点偏移等于 docstring 中的模板."""
    node, cell, lower, spacing = _mesh_arrays(mesh_class, box, shape)
    templates = np.asarray(TEMPLATES[mesh_class], dtype=np.float64)

    offset = (node[cell[:len(templates)]] - lower) / spacing
    np.testing.assert_allclose(offset, templates, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mesh_class,box,shape", CASES)
def test_translation_classes(mesh_class, box, shape):
    """同类单元的相对顶点坐标与代表单元 e % N_k 只差舍入."""
    node, cell, _, _ = _mesh_arrays(mesh_class, box, shape)
    num_classes = len(TEMPLATES[mesh_class])

    rel = node[cell[:, 1:]] - node[cell[:, :1]]  # (NC, NV - 1, GD)
    np.testing.assert_allclose(rel, rel[np.arange(cell.shape[0]) % num_classes], rtol=0, atol=1e-13)
