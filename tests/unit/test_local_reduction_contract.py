"""统一 LocalReduction 契约及旧缩聚器兼容适配测试."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    CondensationReductionAdapter,
    ExactSchurReduction,
    LocalReduction,
    LocalReductionBatchResult,
    PIMLShapeReduction,
    PIMLStiffnessReduction,
    SubstructurePrototype,
)


def _prototype_and_local_stiffness():
    prototype = SubstructurePrototype(
        cell_size=(1.0, 1.0),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    density_grid = bm.asarray(
        np.linspace(0.55, 0.9, 4).reshape(1, 2, 2),
        dtype=bm.float64,
    )
    density_cell = prototype.grid_to_cell_field(density_grid)
    stiffness = prototype.assemble_local_stiffness_batch(density_cell)
    return prototype, stiffness[0], density_grid[0]


class _CountingCondensor:
    def __init__(self) -> None:
        self.calls = 0

    def condense(self, local_stiffness, density=None):
        self.calls += 1
        return local_stiffness, local_stiffness


def test_adapter_calls_legacy_condense_exactly_once() -> None:
    bm.set_backend("numpy")
    legacy = _CountingCondensor()
    reduction = CondensationReductionAdapter(
        legacy,
        requested_method="test",
        stiffness_source="test",
        recovery_source="test",
    )
    matrix = bm.eye(2, dtype=bm.float64)

    result = reduction.reduce(matrix)

    assert legacy.calls == 1
    assert result.stiffness is matrix
    assert result.recovery is matrix


def test_exact_result_matches_legacy_and_records_sources() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)

    result = reduction.reduce(stiffness, density)
    stiffness_legacy, recovery_legacy = reduction.legacy.condense(stiffness, density)

    assert isinstance(reduction, LocalReduction)
    assert result.diagnostics.stiffness_source == "exact_schur"
    assert result.diagnostics.recovery_source == "exact_schur"
    assert result.used_fallback is False
    np.testing.assert_allclose(
        bm.to_numpy(result.stiffness), bm.to_numpy(stiffness_legacy)
    )
    np.testing.assert_allclose(
        bm.to_numpy(result.recovery), bm.to_numpy(recovery_legacy)
    )


def test_result_recovery_is_independent_of_later_legacy_state() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    first = reduction.reduce(stiffness, density)
    boundary = bm.ones((int(prototype.n_b),), dtype=bm.float64)
    recovered_before = first.recover(boundary)

    reduction.reduce(1.2 * stiffness, density)
    recovered_after = first.recover(boundary)

    np.testing.assert_allclose(
        bm.to_numpy(recovered_after), bm.to_numpy(recovered_before)
    )


def test_scalar_reduce_rejects_batch_and_reduce_many_preserves_items() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    stiffness_batch = bm.stack([stiffness, 1.2 * stiffness], axis=0)
    density_batch = bm.stack([density, density], axis=0)

    with pytest.raises(ValueError, match="reduce_many"):
        reduction.reduce(stiffness_batch, density_batch)

    results = reduction.reduce_many(stiffness_batch, density_batch)
    assert isinstance(results, LocalReductionBatchResult)
    assert len(results) == 2
    assert all(
        item.stiffness.shape == (prototype.n_b, prototype.n_b)
        for item in results
    )
    assert all(
        item.recovery.shape == (prototype.n_i, prototype.n_b)
        for item in results
    )
    assert results.stiffness.shape == (2, prototype.n_b, prototype.n_b)
    assert results.recovery.shape == (2, prototype.n_i, prototype.n_b)


def test_exact_reduce_many_calls_vectorized_legacy_once() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    stiffness_batch = bm.stack([stiffness, 1.2 * stiffness], axis=0)
    density_batch = bm.stack([density, density], axis=0)
    legacy_condense = reduction.legacy.condense
    calls = 0

    def counted_condense(local_stiffness, local_density=None):
        nonlocal calls
        calls += 1
        return legacy_condense(local_stiffness, local_density)

    expected_reduction = ExactSchurReduction(
        prototype.i_dofs, prototype.b_dofs
    )
    expected_stiffness, expected_recovery = expected_reduction.legacy.condense(
        stiffness_batch, density_batch
    )
    reduction.legacy.condense = counted_condense
    result = reduction.reduce_many(stiffness_batch, density_batch)

    np.testing.assert_allclose(
        bm.to_numpy(result.stiffness), bm.to_numpy(expected_stiffness)
    )
    np.testing.assert_allclose(
        bm.to_numpy(result.recovery), bm.to_numpy(expected_recovery)
    )
    assert calls == 1
    assert len(result.diagnostics) == 2
    assert all(
        item.stiffness_source == "exact_schur"
        and item.recovery_source == "exact_schur"
        for item in result.diagnostics
    )


class _ConstantBatchModel(torch.nn.Module):
    def __init__(self, output: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            "output",
            torch.as_tensor(output, dtype=torch.float32).reshape(1, -1),
        )
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return self.output.expand(inputs.shape[0], -1)


def test_piml_shape_reduce_many_vectorizes_route_a_and_matches_scalar() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    exact = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    exact_result = exact.reduce(stiffness, density)
    target = exact_result.recovery @ prototype.deformation_basis
    model = _ConstantBatchModel(bm.to_numpy(target))
    reduction = PIMLShapeReduction(
        prototype.i_dofs,
        prototype.b_dofs,
        model=model,
        rigid_basis=prototype.rigid_basis,
        deformation_basis=prototype.deformation_basis,
        rigid_interior=prototype.rigid_interior_modes,
    )
    stiffness_batch = bm.stack([stiffness, stiffness], axis=0)
    density_batch = bm.stack([density, density], axis=0)

    batch_result = reduction.reduce_many(stiffness_batch, density_batch)

    assert model.calls == 1
    assert all(not item.used_fallback for item in batch_result.diagnostics)
    scalar_stiffness, scalar_recovery = reduction.legacy.condense(
        stiffness, density
    )
    np.testing.assert_allclose(
        bm.to_numpy(batch_result.stiffness[0]),
        bm.to_numpy(scalar_stiffness),
    )
    np.testing.assert_allclose(
        bm.to_numpy(batch_result.recovery[0]),
        bm.to_numpy(scalar_recovery),
    )


class _MixedFiniteBatchModel(_ConstantBatchModel):
    def forward(self, inputs):
        output = super().forward(inputs).clone()
        output[1, 0] = torch.nan
        return output


def test_piml_shape_reduce_many_falls_back_only_nonfinite_rows() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    exact = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    exact_result = exact.reduce(stiffness, density)
    target = exact_result.recovery @ prototype.deformation_basis
    model = _MixedFiniteBatchModel(bm.to_numpy(target))
    reduction = PIMLShapeReduction(
        prototype.i_dofs,
        prototype.b_dofs,
        model=model,
        rigid_basis=prototype.rigid_basis,
        deformation_basis=prototype.deformation_basis,
        rigid_interior=prototype.rigid_interior_modes,
    )
    stiffness_batch = bm.stack([stiffness, stiffness], axis=0)
    density_batch = bm.stack([density, density], axis=0)

    result = reduction.reduce_many(stiffness_batch, density_batch)

    assert [item.used_fallback for item in result.diagnostics] == [False, True]
    assert result.diagnostics[1].fallback_reason == "nonfinite_prediction"
    np.testing.assert_allclose(
        bm.to_numpy(result.stiffness[1]),
        bm.to_numpy(exact_result.stiffness),
    )
    np.testing.assert_allclose(
        bm.to_numpy(result.recovery[1]),
        bm.to_numpy(exact_result.recovery),
    )


def test_piml_shape_gate_rejection_records_failed_gate() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    exact = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    exact_result = exact.reduce(stiffness, density)
    target = exact_result.recovery @ prototype.deformation_basis
    model = _ConstantBatchModel(10.0 * bm.to_numpy(target))
    reduction = PIMLShapeReduction(
        prototype.i_dofs,
        prototype.b_dofs,
        model=model,
        rigid_basis=prototype.rigid_basis,
        deformation_basis=prototype.deformation_basis,
        rigid_interior=prototype.rigid_interior_modes,
    )
    stiffness_batch = bm.stack([stiffness], axis=0)
    density_batch = bm.stack([density], axis=0)

    result = reduction.reduce_many(stiffness_batch, density_batch)

    diagnostics = result.diagnostics[0]
    assert diagnostics.used_fallback is True
    assert diagnostics.fallback_reason == "gate_rejected"
    assert diagnostics.metrics["failed_gate"] == "excess_ratio"
    assert diagnostics.metrics["excess_ratio"] > diagnostics.metrics["gate_limit"]


def test_piml_shape_missing_model_reports_exact_sources_and_reason() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    reduction = PIMLShapeReduction(
        prototype.i_dofs,
        prototype.b_dofs,
        model=None,
        rigid_basis=prototype.rigid_basis,
        deformation_basis=prototype.deformation_basis,
        rigid_interior=prototype.rigid_interior_modes,
    )

    result = reduction.reduce(stiffness, density)

    assert result.used_fallback is True
    assert result.diagnostics.fallback_reason == "model_missing"
    assert result.diagnostics.stiffness_source == "exact_schur"
    assert result.diagnostics.recovery_source == "exact_schur"


def test_piml_stiffness_missing_model_reports_exact_sources_and_reason() -> None:
    bm.set_backend("numpy")
    prototype, stiffness, density = _prototype_and_local_stiffness()
    reduction = PIMLStiffnessReduction(
        prototype.i_dofs,
        prototype.b_dofs,
        model=None,
        is_cholesky=True,
        range_basis=prototype.deformation_basis,
    )

    result = reduction.reduce(stiffness, density)

    assert result.used_fallback is True
    assert result.diagnostics.fallback_reason == "model_missing"
    assert result.diagnostics.stiffness_source == "exact_schur"
    assert result.diagnostics.recovery_source == "exact_schur"
