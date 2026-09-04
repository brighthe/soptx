"""子结构模型 artifact 签名测试。"""

from dataclasses import replace

import pytest

from soptx.ml import ShapeFunctionSurrogateNet
from soptx.ml.substructure import (
    ArtifactCompatibilityError,
    ModelSignature,
    load_checkpoint,
    load_legacy_state_dict,
    save_checkpoint,
)


def _signature() -> ModelSignature:
    return ModelSignature(
        n_fine=(2, 2),
        input_dim=4,
        output_dim=6,
        n_interior_dofs=2,
        n_reduced=3,
        sampler_version="test-v1",
    )


def _model() -> ShapeFunctionSurrogateNet:
    return ShapeFunctionSurrogateNet(4, 6, hidden_dim=8)


def test_checkpoint_round_trip_and_signature_mismatch(tmp_path) -> None:
    path = tmp_path / "shape_function.pt"
    model = _model()
    signature = _signature()
    save_checkpoint(path, model, signature, {"validation_gate_pass_rate": 0.95})

    restored, summary = load_checkpoint(path, _model, signature)

    assert restored.training is False
    assert summary["validation_gate_pass_rate"] == 0.95
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert expected.equal(actual)

    incompatible = replace(signature, sampler_version="different-v2")
    with pytest.raises(ArtifactCompatibilityError, match="sampler_version"):
        load_checkpoint(path, _model, incompatible)


def test_legacy_state_dict_requires_explicit_loader(tmp_path) -> None:
    import torch

    path = tmp_path / "legacy.pt"
    model = _model()
    torch.save(model.state_dict(), path)

    with pytest.raises(ArtifactCompatibilityError, match="旧裸 state_dict"):
        load_checkpoint(path, _model, _signature())

    restored = load_legacy_state_dict(path, _model())
    assert restored.training is False
