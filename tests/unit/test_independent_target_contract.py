"""独立条目提供器的参数边界与材料标度契约测试."""

from copy import deepcopy

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.ml.substructure.independent_checkpoints import decoder_metadata_matches
from soptx.ml.substructure.independent_contract import provider_metadata_matches


@pytest.mark.parametrize("override, message", [
    ({"cell_size": (1.0,)}, "等长"),
    ({"n_fine": (2, 2, 2)}, "等长"),
    ({"cell_size": (np.nan, 1.0)}, "有限"),
    ({"cell_size": (np.inf, 1.0)}, "有限"),
    ({"cell_size": (0.0, 1.0)}, "有限"),
    ({"cell_size": (-1.0, 1.0)}, "有限"),
    ({"cell_size": (True, 1.0)}, "有限"),
    ({"n_fine": (2.5, 2)}, "至少 2"),
    ({"n_fine": (True, 2)}, "至少 2"),
    ({"n_fine": (1, 2)}, "至少 2"),
    ({"n_fine": (0, 2)}, "至少 2"),
    ({"chunk_size": 0}, "chunk_size"),
    ({"chunk_size": -1}, "chunk_size"),
    ({"chunk_size": 1.5}, "chunk_size"),
    ({"chunk_size": True}, "chunk_size"),
])
def test_invalid_configuration_rejected_before_assembly(monkeypatch, override, message):
    """非法配置应在创建有限元原型前明确拒绝."""
    def unexpected_assembly(*args, **kwargs):
        pytest.fail("非法参数不应触发有限元原型构建")

    monkeypatch.setattr(
        "soptx.fem.substructure.independent_targets.SubstructurePrototype",
        unexpected_assembly,
    )
    options = {"cell_size": (1.0, 1.0), "n_fine": (2, 2)}
    options.update(override)
    with pytest.raises(ValueError, match=message):
        IndependentTargetProvider(**options)


@pytest.mark.parametrize("trace_kind", ["linear_corner", "full_trace"])
def test_uniform_modulus_scaling_and_legacy_metadata(trace_kind):
    """刚度随模量线性缩放, 延拓不变, 旧元数据可按固定约定比较."""
    bm.set_backend("numpy")
    provider = IndependentTargetProvider(
        cell_size=(1.0, 2.0), n_fine=(np.int64(2), np.int64(2)),
        chunk_size=np.int64(1), trace_kind=trace_kind,
    )
    targets = provider(np.array([[1.0] * 4, [0.25] * 4]))
    np.testing.assert_allclose(targets["shape"][1], targets["shape"][0], atol=1e-12)
    np.testing.assert_allclose(
        targets["stiffness"][1], 0.25 * targets["stiffness"][0], atol=1e-12,
    )
    current = provider.metadata()
    legacy = deepcopy(current)
    del legacy["material_scaling"]
    original = deepcopy(legacy)
    assert provider_metadata_matches(legacy, current)
    assert legacy == original
    assert provider_metadata_matches(current, current)
    for field, value in (
        ("reference_young_modulus", 2.0), ("penal", 3.0),
        ("rho_min", 0.01), ("stiffness_quantity", "physical"),
        ("recovery_quantity", "unknown"),
    ):
        incompatible = deepcopy(current)
        incompatible["material_scaling"][field] = value
        assert not provider_metadata_matches(incompatible, current)
    legacy["poisson_ratio"] = 0.2
    assert not provider_metadata_matches(legacy, current)
    assert not provider_metadata_matches(None, current)

@pytest.mark.parametrize("trace_kind", ["linear_corner", "full_trace"])
def test_plane_hypothesis_changes_stiffness_and_metadata(trace_kind):
    """平面应变应改变标签且不兼容平面应力权重元数据."""
    bm.set_backend("numpy")
    options = dict(cell_size=(1.0, 1.0), n_fine=(2, 2), trace_kind=trace_kind)
    stress = IndependentTargetProvider(**options)
    strain = IndependentTargetProvider(**options, hypothesis="plane_strain")
    x = np.ones((1, 4))
    assert stress.metadata()["material_hypothesis"] == "plane_stress"
    assert strain.metadata()["material_hypothesis"] == "plane_strain"
    assert not provider_metadata_matches(stress.metadata(), strain.metadata())
    assert not np.allclose(stress(x)["stiffness"], strain(x)["stiffness"])
    # nu=0 时两种二维本构一致, 标签也应一致.
    zero_stress = IndependentTargetProvider(**options, nu=0.0)
    zero_strain = IndependentTargetProvider(**options, nu=0.0, hypothesis="plane_strain")
    for name in ("shape", "stiffness"):
        np.testing.assert_allclose(zero_stress(x)[name], zero_strain(x)[name], atol=1e-12)


@pytest.mark.parametrize("hypothesis", ["plane_stress", "plane_strain"])
def test_global_and_local_material_hypothesis_match(hypothesis):
    """全局材料与铺开的子结构须使用相同二维本构."""
    from soptx.fem.substructure import GlobalAssembler, build_substructures

    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        (2.0, 1.0), (2, 1), (2, 2), hypothesis=hypothesis,
    )
    prototype, meshes, _ = build_substructures(assembler)
    provider = IndependentTargetProvider(
        cell_size=(1.0, 1.0), n_fine=(2, 2), hypothesis=hypothesis,
    )
    assert assembler.material.hypothesis == hypothesis
    assert prototype.material.hypothesis == hypothesis
    assert all(mesh.prototype is prototype for mesh in meshes)
    np.testing.assert_allclose(prototype.KE_unit, provider.prototype.KE_unit)


@pytest.mark.parametrize("hypothesis", ["plane_stress", "plane_strain"])
def test_3d_provider_rejects_plane_hypothesis(hypothesis):
    """三维提供器不能采用二维材料假设."""
    with pytest.raises(ValueError, match="hypothesis"):
        IndependentTargetProvider(hypothesis=hypothesis)


@pytest.mark.parametrize("trace_kind", ["linear_corner", "full_trace"])
@pytest.mark.parametrize("dim", [2, 3])
def test_exact_labels_match_exact_route_condensation(dim, trace_kind):
    """训练标签与精确路线的缩聚加迹投影一致.

    精确路线先由 ExactSchurCondensation 得到完整接口的 K_s 与 N, 再经
    project_stiffness 与 reduce_recovery 变换到接口空间; 标签直接对 K_ib T
    求解. 两者数学等价, 差异应只来自舍入.
    """
    from soptx.fem.substructure import ExactSchurCondensation

    bm.set_backend("numpy")
    provider = IndependentTargetProvider(
        cell_size=(1.0,) * dim, n_fine=(2,) * dim, trace_kind=trace_kind,
    )
    rng = np.random.default_rng(2026)
    modulus = rng.uniform(1e-3, 1.0, (3, provider.prototype.n_cells))
    exact = provider.exact_matrices(modulus)

    condensation = ExactSchurCondensation(
        provider.prototype.i_dofs, provider.prototype.b_dofs,
    )
    K_s, N = condensation.condense(bm.asarray(exact["local_stiffness"]))
    stiffness = bm.to_numpy(provider.trace.project_stiffness(K_s))
    shape = bm.to_numpy(provider.trace.reduce_recovery(N))

    assert exact["stiffness"].shape == stiffness.shape
    assert exact["shape"].shape == shape.shape
    scale = np.abs(stiffness).max()
    np.testing.assert_allclose(
        exact["stiffness"], stiffness, rtol=1e-10, atol=1e-12 * scale,
    )
    np.testing.assert_allclose(exact["shape"], shape, rtol=1e-10, atol=1e-12)


def test_decoder_contract_handles_roundoff():
    """允许相容几何与刚体基的舍入差异, 仍拒绝独立条目编号变化."""
    saved = {
        "cell_size": [0.1, 1.0],
        "rigid_basis": [[1.0], [0.0]],
        "rigid_interior": [[1.0]],
        "pivot_indices": [0],
        "free_indices": [1],
        "roundtrip_tolerance": {"rtol": 1e-8, "atol": 1e-10},
    }
    current = dict(
        saved,
        cell_size=[0.1 * (1.0 + 1e-13), 1.0],
        rigid_basis=[[1.0 + 1e-10], [0.0]],
    )
    assert decoder_metadata_matches(saved, current)
    assert not decoder_metadata_matches(saved, dict(current, pivot_indices=[1]))
    assert not decoder_metadata_matches(saved, dict(current, cell_size=[0.2, 1.0]))
