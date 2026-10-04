"""通用单网络构建与历史权重兼容性回归测试."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.ml.substructure.independent_contract import (
    ACTIVATIONS, HIDDEN_DIMS, SCHEMA, build_network, provider_metadata_matches,
)
from soptx.ml.substructure.nets import (
    DirectStiffnessNet, IndependentOutputNet, SplitOutputNet,
    SplitShapeFunctionNet,
)

META = {
    "spatial_dimension": 2, "trace": "linear_corner", "n_fine": [2, 2],
    "n_trace": 8, "n_rigid": 3, "n_i": 2, "n_cells": 4,
    "n_shape_targets": 10, "n_stiffness_targets": 15,
}


@pytest.mark.parametrize("route", ["shape", "stiffness"])
@pytest.mark.parametrize("count", [1, 2])
def test_network_structure_depends_on_count(route, count):
    """两条路线使用相同的网络数量选择规则."""
    model = build_network(META, route=route, num_networks=count)
    assert type(model) is (IndependentOutputNet if count == 1 else SplitOutputNet)
    output = model(torch.ones(2, 4, dtype=torch.float64))
    assert output.shape == (2, META[f"n_{route}_targets"])
    assert torch.isfinite(output).all()


def _checkpoint_module():
    from soptx.ml.substructure import independent_checkpoints
    return independent_checkpoints


def _loader():
    return _checkpoint_module().load_analysis_networks


def _save_checkpoint(directory, route, model, metadata=META):
    output_dim = metadata[f"n_{route}_targets"]
    torch.save({
        "schema": SCHEMA, "route": route,
        "dataset": {"complete": True, "provider": metadata,
                    "input_quantity": "normalized_young_modulus"},
        "architecture": {
            "input_dim": metadata["n_cells"], "output_dim": output_dim,
            "hidden_dims": HIDDEN_DIMS,
            "activations": [cls.__name__ for cls in ACTIVATIONS],
            "num_networks": 1, "output_groups": (tuple(range(output_dim)),),
            "model_class": type(model).__name__, "dtype": "float64",
        },
        "model_state": model.state_dict(),
    }, directory / f"{route}_best.pt")


@pytest.mark.parametrize("shape_class", [SplitOutputNet, SplitShapeFunctionNet])
def test_legacy_single_network_predictions_preserved(tmp_path, shape_class):
    """旧形函数包装权重与旧刚度权重加载后保持预测一致."""
    old_shape = shape_class(4, 10, HIDDEN_DIMS,
                            output_groups=(tuple(range(10)),),
                            activation=ACTIVATIONS).double()
    old_stiffness = DirectStiffnessNet(4, 15, HIDDEN_DIMS,
                                       activation=ACTIVATIONS).double()
    _save_checkpoint(tmp_path, "shape", old_shape)
    _save_checkpoint(tmp_path, "stiffness", old_stiffness)
    models, _ = _loader()(tmp_path, META, route="stiffness")
    x = torch.ones(2, 4, dtype=torch.float64)
    for route, old in (("shape", old_shape), ("stiffness", old_stiffness)):
        assert type(models[route]) is IndependentOutputNet
        torch.testing.assert_close(models[route](x), old(x), rtol=0, atol=0)
    old_shape.restore_order[0] = 1
    _save_checkpoint(tmp_path, "shape", old_shape)
    with pytest.raises(ValueError, match="输出索引顺序"):
        _loader()(tmp_path, META, route="shape")


def test_new_single_network_checkpoint_roundtrip(tmp_path):
    """新类名权重可按原接口恢复."""
    model = build_network(META, route="shape", num_networks=1)
    _save_checkpoint(tmp_path, "shape", model)
    loaded, _ = _loader()(tmp_path, META, route="shape")
    x = torch.ones(2, 4, dtype=torch.float64)
    torch.testing.assert_close(loaded["shape"](x), model(x), rtol=0, atol=0)

@pytest.mark.parametrize("dim", [2, 3])
def test_analysis_provider_restores_checkpoint_configuration(tmp_path, dim):
    """整体分析从形函数权重恢复二维或三维子结构配置."""
    hypothesis = "plane_strain" if dim == 2 else None
    provider = IndependentTargetProvider(
        cell_size=(1.0,) * dim, n_fine=(2,) * dim,
        nu=0.3, trace_kind="linear_corner", hypothesis=hypothesis,
    )
    metadata = provider.metadata()
    model = build_network(metadata, route="shape", num_networks=1)
    _save_checkpoint(tmp_path, "shape", model, metadata)
    restored = _checkpoint_module().load_analysis_provider(tmp_path)
    assert provider_metadata_matches(metadata, restored.metadata())


def test_analysis_loader_requires_shape_checkpoint(tmp_path):
    """所有整体分析路线均须先由 shape_best.pt 恢复配置和位移延拓."""
    with pytest.raises(FileNotFoundError, match="shape_best"):
        _checkpoint_module().load_analysis_provider(tmp_path)


def test_analysis_loader_requires_stiffness_checkpoint_for_stiffness_route(tmp_path):
    """刚度路线缺少 stiffness_best.pt 时在分析结果目录创建前拒绝."""
    model = build_network(META, route="shape", num_networks=1)
    _save_checkpoint(tmp_path, "shape", model)
    with pytest.raises(FileNotFoundError, match="stiffness_best"):
        _loader()(tmp_path, META, route="stiffness")


def test_analysis_loader_rejects_mismatched_route_metadata(tmp_path):
    """两条权重的接口空间或独立分量元数据不一致时拒绝组合."""
    shape = build_network(META, route="shape", num_networks=1)
    stiffness = build_network(META, route="stiffness", num_networks=1)
    _save_checkpoint(tmp_path, "shape", shape)
    mismatched = dict(META, trace="full_trace")
    _save_checkpoint(tmp_path, "stiffness", stiffness, mismatched)
    with pytest.raises(ValueError, match="配置不匹配"):
        _loader()(tmp_path, META, route="stiffness")


def test_analysis_loader_rejects_both(tmp_path):
    """加载入口拒绝 both, 不尝试读取权重文件."""
    with pytest.raises(ValueError, match="route 必须为 shape 或 stiffness"):
        _loader()(tmp_path, META, route="both")


@pytest.mark.parametrize("count", [1, 2])
def test_build_network_accepts_custom_hidden_layers(count):
    """隐藏层宽度与激活类可显式指定, 单个激活类按层数展开记录."""
    from torch import nn
    from soptx.ml.substructure.independent_contract import activation_names

    model = build_network(META, route="shape", num_networks=count,
                          hidden_dims=(8, 6), activation=nn.SiLU)
    assert tuple(model.hidden_dims) == (8, 6)
    assert activation_names(model) == ["SiLU", "SiLU"]
    assert model(torch.ones(2, 4, dtype=torch.float64)).shape == (2, 10)
    default = build_network(META, route="shape", num_networks=count)
    assert tuple(default.hidden_dims) == HIDDEN_DIMS
    assert activation_names(default) == [cls.__name__ for cls in ACTIVATIONS]


class _MatrixCodec:
    """将形函数标签还原为矩阵, 不依赖有限元求解."""

    def decode(self, values):
        return values.reshape(-1, 2, 5)


@pytest.mark.parametrize("count", [1, 2])
def test_custom_hidden_layers_round_trip_through_checkpoint(tmp_path, count):
    """非默认结构训练后保存的记录可被加载器按原结构重建."""
    import json

    import numpy as np
    from torch import nn
    from soptx.ml.substructure.independent_training import train_network
    from soptx.ml.substructure.training import TrainingConfig

    dataset = tmp_path / "data"
    dataset.mkdir()
    manifest = {"schema": SCHEMA, "complete": True, "provider": META,
                "input_quantity": "normalized_young_modulus",
                "counts": {"train": 2, "validation": 2}}
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        np.save(dataset / f"{split}_inputs.npy", rng.uniform(0.1, 1.0, (2, 4)))
        np.save(dataset / f"{split}_shape_targets.npy", rng.standard_normal((2, 10)))
    model = build_network(META, route="shape", num_networks=count,
                          hidden_dims=(8, 6), activation=(nn.Tanh, nn.ELU))
    result = train_network(
        dataset, tmp_path / "training", route="shape", network=model,
        provider=SimpleNamespace(codecs={"shape": _MatrixCodec()}, metadata=lambda: META),
        config=TrainingConfig(1, 2, optimizer_params={"lr": 1e-3}),
    )

    payload = torch.load(result["checkpoint"], weights_only=True)
    assert tuple(payload["architecture"]["hidden_dims"]) == (8, 6)
    assert payload["architecture"]["activations"] == ["Tanh", "ELU"]
    loaded, source = _checkpoint_module().load_independent_network(
        result["checkpoint"], META, route="shape",
    )
    assert tuple(loaded.hidden_dims) == (8, 6)
    assert type(loaded) is type(model)
    x = torch.rand(3, 4, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("change, message", [
    ({"activations": ["Linear"] * len(HIDDEN_DIMS)}, "激活类"),
    ({"activations": ["NotAModule"] * len(HIDDEN_DIMS)}, "激活类"),
    ({"activations": ["Tanh"]}, "隐藏层"),
    ({"hidden_dims": None}, "隐藏层"),
])
def test_loader_rejects_invalid_hidden_layer_record(tmp_path, change, message):
    """结构记录缺失、层数不符或激活类不可识别时拒绝加载."""
    model = build_network(META, route="shape", num_networks=1)
    _save_checkpoint(tmp_path, "shape", model)
    path = tmp_path / "shape_best.pt"
    payload = torch.load(path, weights_only=True)
    payload["architecture"].update(change)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        _checkpoint_module().load_independent_network(path, META, route="shape")


@pytest.mark.parametrize("route", ["shape", "stiffness"])
def test_default_num_networks_is_four_for_both_routes(route):
    """两条路线默认均拆分为 4 个网络; None 不再表示按路线取默认."""
    model = build_network(META, route=route)
    assert type(model) is SplitOutputNet
    assert len(model.nets) == 4
    with pytest.raises(ValueError, match="正整数"):
        build_network(META, route=route, num_networks=None)
