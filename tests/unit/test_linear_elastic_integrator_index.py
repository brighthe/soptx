# -*- coding: utf-8 -*-
"""``LinearElasticIntegrator`` 的 ``index`` 语义单元测试.

给定单元子集 ``index`` 时, 各装配路径只对这些单元积分: 输出为 (len(index), LDOF, LDOF),
等于不带 ``index`` 的全网格结果按 ``index`` 取行; 单元系数 ``coef`` 的长度也按子集计.

测试覆盖范围:
1. tri / quad / tet / hex x p = 1, 2 x standard / voigt / fast;
2. ``coef`` 取 None 与 (len(index), );
3. ``to_global_dof`` 与输出的单元一一对应.
"""

import pytest
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


RTOL = 1.0e-12

# 区域不对称, 各轴格子数互不相同; 子集乱序且含首尾单元, 以便暴露按全网格取数的错位
BOX_2D = [0.0, 2.0, -1.0, 0.5]
BOX_3D = [0.0, 2.0, -1.0, 0.5, 1.0, 1.75]
MESHES = [
    ('tri', TriangleMesh, BOX_2D, (3, 2)),
    ('quad', QuadrangleMesh, BOX_2D, (3, 2)),
    ('tet', TetrahedronMesh, BOX_3D, (2, 3, 2)),
    ('hex', HexahedronMesh, BOX_3D, (2, 3, 2)),
]
DEGREES = (1, 2)
METHODS = ('standard', 'voigt', 'fast')


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _cases():
    return [pytest.param(mesh_class, box, shape, p, method, id=f"{name}-p{p}-{method}")
            for name, mesh_class, box, shape in MESHES
            for p in DEGREES for method in METHODS]


def _relerr(actual, expected):
    scale = float(bm.max(bm.abs(expected))) or 1.0

    return float(bm.max(bm.abs(actual - expected))) / scale


@pytest.mark.parametrize("mesh_class,box,shape,p,method", _cases())
@pytest.mark.parametrize("with_coef", [False, True], ids=["coef=None", "coef=(len(index),)"])
def test_index_selects_cells(mesh_class, box, shape, p, method, with_coef):
    """带 index 的单元矩阵等于全网格单元矩阵按 index 取行, 再乘子集上的单元系数."""
    mesh = mesh_class.from_box(box, *shape)
    GD = len(shape)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))
    material = IsotropicLinearElasticMaterial(
                    hypothesis='plane_strain' if GD == 2 else '3D',
                    lame_lambda=1.0,
                    shear_modulus=0.75,
                    device=bm.get_device(mesh),
                )
    NC = int(mesh.number_of_cells())
    index = bm.array([NC - 1, 0, NC // 2, 1], dtype=bm.int64)
    coef = bm.array([0.3, 0.9, 0.55, 0.7], dtype=bm.float64) if with_coef else None

    full = LinearElasticIntegrator(material, coef=None, method=method).assembly(space)
    sub_integrator = LinearElasticIntegrator(material, coef=coef, index=index, method=method)
    sub = sub_integrator.assembly(space)

    expected = full[index] if coef is None else coef[:, None, None] * full[index]
    assert tuple(sub.shape) == tuple(expected.shape)
    assert _relerr(sub, expected) < RTOL
    assert bm.all(sub_integrator.to_global_dof(space) == space.cell_to_dof()[index])
