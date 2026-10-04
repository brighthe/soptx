# -*- coding: utf-8 -*-
"""结构化网格生成器 ``create_box_mesh`` 单元测试.

测试覆盖范围:
1. 四种网格的节点、单元、边 (三维再加面) 与 FEALPy ``from_box`` 逐位一致;
2. 高次空间的 ``cell_to_dof`` 与 FEALPy ``from_box`` 逐位一致;
3. 平移类约定: 每个单元的相对顶点坐标等于其代表单元的 (只差舍入);
4. 非法输入报错.
"""

import numpy as np
import pytest
from soptx.backend import backend_manager as bm
from soptx.functionspace import LagrangeFESpace
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh

from soptx.mesh import create_box_mesh


# 各轴格子数互不相同, 区域不对称, 以便暴露轴序错误
BOX_2D = [0.0, 2.0, -1.0, 0.5]
BOX_3D = [0.0, 2.0, -1.0, 0.5, 1.0, 1.75]
CASES = [
    ('tri', TriangleMesh, BOX_2D, (3, 2), 2),
    ('quad', QuadrangleMesh, BOX_2D, (3, 2), 1),
    ('tet', TetrahedronMesh, BOX_3D, (3, 2, 4), 6),
    ('hex', HexahedronMesh, BOX_3D, (3, 2, 4), 1),
]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _assert_same(a, b):
    """逐位相等, 含 dtype 与形状"""
    a, b = bm.to_numpy(a), bm.to_numpy(b)
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mesh_type,mesh_class,box,shape,_", CASES)
def test_matches_fealpy_from_box(mesh_type, mesh_class, box, shape, _):
    """节点、单元及派生实体与 FEALPy from_box 逐位一致."""
    mesh = create_box_mesh(mesh_type, box, *shape).mesh
    ref = mesh_class.from_box(box, *shape)

    assert type(mesh) is type(ref)
    entities = ('node', 'cell', 'edge') + (('face', ) if len(shape) == 3 else ())
    for name in entities:
        _assert_same(mesh.entity(name), ref.entity(name))


@pytest.mark.parametrize("mesh_type,mesh_class,box,shape,_", CASES)
def test_cell_to_dof_matches_fealpy(mesh_type, mesh_class, box, shape, _):
    """p=2 空间的 cell_to_dof 与 FEALPy from_box 逐位一致."""
    mesh = create_box_mesh(mesh_type, box, *shape).mesh
    ref = mesh_class.from_box(box, *shape)

    _assert_same(LagrangeFESpace(mesh, p=2).cell_to_dof(),
                 LagrangeFESpace(ref, p=2).cell_to_dof())


@pytest.mark.parametrize("mesh_type,_,box,shape,num_classes", CASES)
def test_translation_classes(mesh_type, _, box, shape, num_classes):
    """同类单元的相对顶点坐标与代表单元只差舍入, 类数与单元数符合约定."""
    mesh, classes = create_box_mesh(mesh_type, box, *shape)
    node = bm.to_numpy(mesh.entity('node'))
    cell = bm.to_numpy(mesh.entity('cell'))

    assert classes.num_classes == num_classes
    assert classes.num_cells == cell.shape[0] == num_classes * int(np.prod(shape))

    reps = bm.to_numpy(classes.representatives())
    cls = bm.to_numpy(classes.class_index())
    np.testing.assert_array_equal(reps, np.arange(num_classes))
    np.testing.assert_array_equal(cls, np.arange(cell.shape[0]) % num_classes)

    rel = node[cell[:, 1:]] - node[cell[:, :1]]  # (NC, NV - 1, GD)
    np.testing.assert_allclose(rel, rel[reps][cls], rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    "args,kwargs,error",
    [
        (('prism', BOX_3D, 1, 1, 1), {}, ValueError),
        (('tri', BOX_2D, 2, 2, 2), {}, ValueError),
        (('tri', BOX_3D, 2, 2), {}, ValueError),
        (('tet', BOX_3D, 2, 2), {}, TypeError),
        (('quad', BOX_2D, 0, 2), {}, TypeError),
        (('hex', BOX_3D, 2, 2.0, 2), {}, TypeError),
    ],
)
def test_invalid_input(args, kwargs, error):
    """未知网格类型、维数不符、格子数非正整数均报错."""
    with pytest.raises(error):
        create_box_mesh(*args, **kwargs)
