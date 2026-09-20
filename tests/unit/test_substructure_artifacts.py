"""子结构模型 artifact 签名测试。"""

from dataclasses import replace

import pytest

import torch
import torch.nn as nn

from soptx.ml.substructure import (
    ArchitectureSignature,
    ArtifactCompatibilityError,
    ModelSignature,
    ShapeFunctionSurrogateNet,
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
    return ShapeFunctionSurrogateNet(4, 6, (8, 8))


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


def test_architecture_is_recorded_and_checked(tmp_path) -> None:
    """架构随 checkpoint 登记, 层数或激活不符一律报错。"""
    path = tmp_path / "shape_function.pt"
    signature = _signature()
    save_checkpoint(path, _model(), signature, {})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["architecture"] == {
        "hidden_dims": (8, 8),
        "activation": "SiLU",
    }

    load_checkpoint(path, _model, signature)

    def wrong_activation() -> ShapeFunctionSurrogateNet:
        return ShapeFunctionSurrogateNet(4, 6, (8, 8), activation=nn.Tanh)

    def wrong_depth() -> ShapeFunctionSurrogateNet:
        return ShapeFunctionSurrogateNet(4, 6, (8, 8, 8))

    for factory in (wrong_activation, wrong_depth):
        with pytest.raises(ArtifactCompatibilityError, match="网络架构不匹配"):
            load_checkpoint(path, factory, signature)


def test_checkpoint_without_architecture_degrades_to_warning(tmp_path) -> None:
    """早于架构登记的 checkpoint 只警告, 不阻断加载。"""
    path = tmp_path / "shape_function.pt"
    legacy_path = tmp_path / "legacy_schema.pt"
    signature = _signature()
    save_checkpoint(path, _model(), signature, {})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["architecture"]
    torch.save(payload, legacy_path)

    with pytest.warns(RuntimeWarning, match="未登记网络架构"):
        load_checkpoint(legacy_path, _model, signature)


def test_architecture_signature_requires_registered_model() -> None:
    """未登记架构属性的模型不能写入 checkpoint。"""
    with pytest.raises(TypeError, match="未登记 hidden_dims/activation_name"):
        ArchitectureSignature.from_model(nn.Linear(2, 2))
