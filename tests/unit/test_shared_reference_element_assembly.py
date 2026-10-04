# -*- coding: utf-8 -*-
"""共享参考 EA ``SharedReferenceElementAssembly`` 单元测试.

以标准 EA (``ElementAssembly``) 为基准: 在 ``from_box`` 结构化网格上, 两者是同一个
离散算子, ``@`` 与 ``diagonal()`` 必须一致到舍入误差 (同类单元的节点坐标只在舍入
意义下相等, 不逐位相同).

测试覆盖范围:
1. tri / quad / tet / hex x p = 1, 2, 单元系数取 None 与 (NC, ), 单列与多列作用;
2. ``update`` 只改 s_e, 结果与按新系数重建的标准 EA 一致;
3. 逐单元参考 (N_k = NC) 在节点扰动后的非结构网格上也与标准 EA 一致;
4. 标准 EA 的 ``update`` 按新系数重新积分;
5. 常驻内存只含 K_k^0、s_e 与 cell2dof;
6. 非法输入报错, 且不在层级注册表中.
"""

import pytest
from soptx.backend import backend_manager as bm
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import ElementRestriction
from soptx.fem.levels import ElementAssembly, SharedReferenceElementAssembly
from soptx.fem.levels.registry import _REGISTRY
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import create_box_mesh


RTOL = 1.0e-11

# 各轴格子数互不相同, 区域不对称, 以便暴露按类重排的错位
BOX_2D = [0.0, 2.0, -1.0, 0.5]
BOX_3D = [0.0, 2.0, -1.0, 0.5, 1.0, 1.75]
MESHES = [
    ('tri', BOX_2D, (3, 2)),
    ('quad', BOX_2D, (3, 2)),
    ('tet', BOX_3D, (2, 3, 2)),
    ('hex', BOX_3D, (2, 3, 2)),
]
DEGREES = (1, 2)


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _make_fixture(mesh_type, box, shape, p, perturb=False):
    """造出空间、材料、平移类与确定性测试数据 (不用随机数, 失败时可直接复现).

    ``perturb`` 为 True 时对节点做确定性小扰动, 同类单元不再只差平移, 平移类失效;
    此时返回的 ``classes`` 不再可用.
    """
    mesh, classes = create_box_mesh(mesh_type, box, *shape)
    if perturb:
        node = mesh.entity('node')
        phase = bm.arange(node.shape[0], dtype=bm.float64)[:, None] \
            * bm.arange(1, node.shape[1] + 1, dtype=bm.float64)
        mesh = type(mesh)(node + 0.02 * bm.sin(phase), mesh.entity('cell'))
    GD = len(shape)
    space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))
    material = IsotropicLinearElasticMaterial(
                    hypothesis='plane_strain' if GD == 2 else '3D',
                    lame_lambda=1.0,
                    shear_modulus=0.75,
                    device=bm.get_device(mesh),
                )
    NC = int(mesh.number_of_cells())
    gdof = int(space.number_of_global_dofs())
    t = bm.linspace(-1.0, 1.0, gdof, dtype=bm.float64)

    return {
        'space': space,
        'material': material,
        'classes': classes,
        'x': bm.sin(13.0 * t) + 0.3 * bm.cos(7.0 * t),
        'X': bm.stack([bm.cos(5.0 * t), bm.sin(3.0 * t) - 0.2], axis=1),
        'coef': 0.4 + 0.5 * bm.abs(bm.sin(bm.linspace(0.0, 9.0, NC, dtype=bm.float64))),
    }


def _standard_ea(fx, coef):
    """标准 EA 基准: 每种 coef 新建积分子 (单元矩阵按积分子实例缓存)."""
    integrator = LinearElasticIntegrator(fx['material'], coef=coef)

    return ElementAssembly.build(fx['space'], integrator)


def _shared_ea(fx, coef):
    """共享参考 EA: 只对各类代表单元积分, G 取全体单元."""
    reps = fx['classes'].representatives()
    K0 = LinearElasticIntegrator(fx['material'], coef=None, index=reps,
                                 method='standard').assembly(fx['space'])
    g = ElementRestriction.from_integrator(LinearElasticIntegrator(fx['material']),
                                           fx['space'], layout='flat')

    return SharedReferenceElementAssembly(fx['space'], restriction=g,
                                          reference_matrices=K0, scale=coef)


def _per_element_ea(fx, coef):
    """逐单元参考 EA: N_k = NC, 参考即不带系数逐单元积分的 K_e^0, 对网格没有要求."""
    integrator = LinearElasticIntegrator(fx['material'], coef=None, method='standard')
    K0 = integrator.assembly(fx['space'])
    g = ElementRestriction.from_integrator(integrator, fx['space'], layout='flat')

    return SharedReferenceElementAssembly(fx['space'], restriction=g,
                                          reference_matrices=K0, scale=coef)


def _relerr(actual, expected):
    scale = float(bm.max(bm.abs(expected))) or 1.0

    return float(bm.max(bm.abs(actual - expected))) / scale


def _cases():
    return [pytest.param(mesh_type, box, shape, p, id=f"{mesh_type}-p{p}")
            for mesh_type, box, shape in MESHES for p in DEGREES]


@pytest.mark.parametrize("mesh_type,box,shape,p", _cases())
@pytest.mark.parametrize("with_coef", [False, True], ids=["coef=None", "coef=(NC,)"])
def test_matches_standard_ea(mesh_type, box, shape, p, with_coef):
    """单列、多列作用与对角都和标准 EA 一致到舍入."""
    fx = _make_fixture(mesh_type, box, shape, p)
    coef = fx['coef'] if with_coef else None
    ea, shared = _standard_ea(fx, coef), _shared_ea(fx, coef)

    assert shared.num_classes == fx['classes'].num_classes
    assert shared.shape == ea.shape
    assert _relerr(shared @ fx['x'], ea @ fx['x']) < RTOL
    assert _relerr(shared @ fx['X'], ea @ fx['X']) < RTOL
    assert _relerr(shared.diagonal(), ea.diagonal()) < RTOL


@pytest.mark.parametrize("mesh_type,box,shape,p", _cases())
def test_update_only_rescales(mesh_type, box, shape, p):
    """update 只换 s_e: K_k^0 不动, 结果与按新系数重建的标准 EA 一致, 且复制持有."""
    fx = _make_fixture(mesh_type, box, shape, p)
    shared = _shared_ea(fx, None)
    K0 = shared.reference_matrices
    coef = bm.copy(fx['coef'])

    shared.update(coef)
    coef[:] = -1.0  # 调用方之后改写自己的数组, 不应影响算子

    ea = _standard_ea(fx, fx['coef'])
    assert shared.reference_matrices is K0
    assert _relerr(shared @ fx['x'], ea @ fx['x']) < RTOL

    shared.update(None)
    assert _relerr(shared @ fx['x'], _standard_ea(fx, None) @ fx['x']) < RTOL


@pytest.mark.parametrize("mesh_type,box,shape,p", _cases())
def test_per_element_reference_on_perturbed_mesh(mesh_type, box, shape, p):
    """N_k = NC 不依赖平移类: 节点扰动后仍与标准 EA 一致, update 后亦然."""
    fx = _make_fixture(mesh_type, box, shape, p, perturb=True)
    NC = fx['coef'].shape[0]
    per_element = _per_element_ea(fx, None)
    K0 = per_element.reference_matrices

    assert per_element.num_classes == NC
    for coef in (None, fx['coef']):
        per_element.update(coef)
        ea = _standard_ea(fx, coef)
        assert per_element.reference_matrices is K0
        assert _relerr(per_element @ fx['x'], ea @ fx['x']) < RTOL
        assert _relerr(per_element @ fx['X'], ea @ fx['X']) < RTOL
        assert _relerr(per_element.diagonal(), ea.diagonal()) < RTOL


@pytest.mark.parametrize("mesh_type,box,shape,p", _cases())
def test_standard_ea_update_reintegrates(mesh_type, box, shape, p):
    """标准 EA 的 update 按新系数重新积分, 与按新系数新建的标准 EA 一致."""
    fx = _make_fixture(mesh_type, box, shape, p)
    ea = _standard_ea(fx, None)

    ea.update(fx['coef'])
    assert _relerr(ea @ fx['x'], _standard_ea(fx, fx['coef']) @ fx['x']) < RTOL

    ea.update(None)
    assert _relerr(ea @ fx['x'], _standard_ea(fx, None) @ fx['x']) < RTOL


def test_persistent_bytes():
    """常驻只含 K_k^0、s_e 与 cell2dof, 不含 (NC, L, L) 的单元矩阵."""
    fx = _make_fixture('tet', BOX_3D, (2, 3, 2), 1)
    shared = _shared_ea(fx, fx['coef'])

    expected = (shared.reference_matrices.nbytes + shared.scale.nbytes
                + shared.restriction.cell2dof.nbytes)
    assert shared.persistent_bytes() == expected
    assert tuple(shared.reference_matrices.shape) == (6, 12, 12)


def test_invalid_input():
    """布局、K_k^0 形状、类数整除与系数形状不符均报错."""
    fx = _make_fixture('tet', BOX_3D, (2, 3, 2), 1)
    space, material = fx['space'], fx['material']
    shared = _shared_ea(fx, None)
    K0, g = shared.reference_matrices, shared.restriction

    component = ElementRestriction.from_integrator(LinearElasticIntegrator(material), space)
    with pytest.raises(ValueError):
        SharedReferenceElementAssembly(space, component, K0)
    with pytest.raises(ValueError):
        SharedReferenceElementAssembly(space, g, K0[:, :6, :6])
    with pytest.raises(ValueError):
        SharedReferenceElementAssembly(space, g, bm.concatenate([K0, K0[:1]], axis=0))
    with pytest.raises(ValueError):
        SharedReferenceElementAssembly(space, g, K0, scale=fx['coef'][:-1])
    with pytest.raises(ValueError):
        shared.update(bm.ones((g.n_cells, 4), dtype=bm.float64))


def test_not_registered():
    """共享参考 EA 只能显式构造, 不进层级注册表."""
    assert SharedReferenceElementAssembly not in _REGISTRY.values()
