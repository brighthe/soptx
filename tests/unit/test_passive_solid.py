"""载荷引入垫片中"实体保留 (passive solid)"一侧的回归测试.

覆盖四件事: 保留单元的物理密度在过滤链末端被钉为满密度 (而不是只钉设计
变量), Function 与裸张量两条路径行为一致, 保留行的密度灵敏度被清零, 以及
全 False 掩码与不设掩码严格等价. 不启动有限元求解.

之所以要单独测"过滤之后"这件事: 应力算例的 rmin = 6 远大于网格尺寸 h = 1,
只把设计变量钉成 1 时, 保留单元的物理密度仍由邻域加权决定, 根本到不了满
密度; 掩码必须施加在投影之后才成立.
"""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh

from soptx.topology.constraints import apply_passive_solid
from soptx.topology.constraints.exemption import PASSIVE_SOLID_DENSITY
from soptx.topology.filters import Filter

# 过滤半径取 2.0, 远大于单元尺寸 0.25, 使"只钉设计变量"必然达不到满密度.
_RMIN = 2.0
_PROJECTION = {"projection_type": "tanh", "beta": 4.0, "eta": 0.5}


def _mesh():
    return TriangleMesh.from_box([0.0, 2.0, 0.0, 1.0], nx=8, ny=4)


def _mask(mesh, n_true: int = 5):
    """取前若干个单元作为保留单元; 具体是哪几个不影响本组断言."""
    mask = np.zeros(mesh.number_of_cells(), dtype=bool)
    mask[:n_true] = True
    return bm.tensor(mask, dtype=bm.bool)


def _filter(mesh, mask):
    return Filter(
        design_mesh=mesh,
        filter_type="projection",
        rmin=_RMIN,
        density_location="element",
        projection_params=dict(_PROJECTION),
        passive_mask=mask,
        enable_logging=False,
    )


def _function(mesh, values):
    space = LagrangeFESpace(mesh, p=0, ctype="D")
    return space.function(bm.tensor(values, dtype=bm.float64))


def test_apply_passive_solid_pins_masked_cells_to_full_density():
    values = bm.tensor([0.1, 0.4, 0.9], dtype=bm.float64)
    mask = bm.tensor([True, False, True], dtype=bm.bool)

    pinned = np.asarray(bm.to_numpy(apply_passive_solid(values, mask)))

    assert pinned.tolist() == [PASSIVE_SOLID_DENSITY, 0.4, PASSIVE_SOLID_DENSITY]
    assert np.asarray(bm.to_numpy(apply_passive_solid(values, None))).tolist() == [0.1, 0.4, 0.9]


def test_initial_density_is_pinned_on_the_passive_region():
    """初值也必须落在可行的实体保留状态上, 否则第 0 步的应力评价口径就不对."""
    mesh = _mesh()
    mask = _mask(mesh)
    passive = np.asarray(bm.to_numpy(mask))
    raw = bm.full((mesh.number_of_cells(),), 0.4, dtype=bm.float64)

    plain = np.asarray(bm.to_numpy(_filter(mesh, None).get_initial_density(bm.copy(raw))))
    pinned = np.asarray(bm.to_numpy(_filter(mesh, mask).get_initial_density(bm.copy(raw))))

    assert (plain[passive] < 0.9).all()
    assert np.allclose(pinned[passive], PASSIVE_SOLID_DENSITY)
    # 非保留单元不受影响
    assert np.allclose(plain[~passive], pinned[~passive])


def test_pinning_design_variables_alone_does_not_reach_full_density():
    """rmin >> h 时, 只把设计变量钉成 1 的做法达不到满密度, 故必须钉在过滤之后."""
    mesh = _mesh()
    mask = _mask(mesh)
    passive = np.asarray(bm.to_numpy(mask))
    design_variable = bm.full((mesh.number_of_cells(),), 0.4, dtype=bm.float64)
    design_variable = bm.set_at(design_variable, passive, 1.0)

    diluted = np.asarray(bm.to_numpy(
        _filter(mesh, None)
        .filter_design_variable(
            design_variable, _function(mesh, np.full(mesh.number_of_cells(), 0.4))
        )[:]
    ))

    assert (diluted[passive] < 0.9).all()


def test_filter_design_variable_pins_function_in_place():
    mesh = _mesh()
    mask = _mask(mesh)
    passive = np.asarray(bm.to_numpy(mask))
    design_variable = bm.full((mesh.number_of_cells(),), 0.4, dtype=bm.float64)
    physical_density = _function(mesh, np.full(mesh.number_of_cells(), 0.4))

    returned = _filter(mesh, mask).filter_design_variable(design_variable, physical_density)

    assert returned is physical_density
    pinned = np.asarray(bm.to_numpy(physical_density[:]))
    assert np.allclose(pinned[passive], PASSIVE_SOLID_DENSITY)
    assert (pinned[~passive] < PASSIVE_SOLID_DENSITY).all()


@pytest.mark.parametrize("kind", ["objective", "constraint"])
def test_passive_rows_are_removed_from_the_sensitivity_chain(kind):
    """rho_phys 在保留单元上与设计变量无关, 这些行进链式法则前必须先清零."""
    mesh = _mesh()
    mask = _mask(mesh)
    passive = np.asarray(bm.to_numpy(mask))
    n_cells = mesh.number_of_cells()
    design_variable = bm.full((n_cells,), 0.4, dtype=bm.float64)

    def filtered(active_mask, grad_values):
        density_filter = _filter(mesh, active_mask)
        density_filter.filter_design_variable(
            design_variable, _function(mesh, np.full(n_cells, 0.4))
        )
        grad = bm.tensor(grad_values, dtype=bm.float64)
        method = getattr(density_filter, f"filter_{kind}_sensitivities")
        key = "obj_grad_rho" if kind == "objective" else "con_grad_rho"
        return np.asarray(bm.to_numpy(
            method(design_variable=design_variable, **{key: grad})
        ))

    gradient = np.full(n_cells, -1.0)
    # 与"先把保留行清零再过滤"的参考结果逐项比较: 只清零, 不额外改动别的行
    reference = gradient.copy()
    reference[passive] = 0.0

    assert np.allclose(filtered(mask, gradient), filtered(None, reference))
    # 清零确实改变了结果, 断言不是平凡成立的
    assert not np.allclose(filtered(mask, gradient), filtered(None, gradient))


def test_all_false_passive_mask_is_equivalent_to_no_passive_region():
    mesh = _mesh()
    empty = bm.zeros((mesh.number_of_cells(),), dtype=bm.bool)
    raw = bm.full((mesh.number_of_cells(),), 0.4, dtype=bm.float64)

    density_filter = _filter(mesh, empty)
    assert density_filter.passive_mask is None
    assert np.allclose(
        np.asarray(bm.to_numpy(density_filter.get_initial_density(bm.copy(raw)))),
        np.asarray(bm.to_numpy(_filter(mesh, None).get_initial_density(bm.copy(raw)))),
    )
