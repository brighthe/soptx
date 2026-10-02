"""优化器配置、训练接线及 checkpoint 记录的回归测试."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from soptx.ml.substructure.training import TrainingConfig, train_surrogate
from soptx.ml.substructure.independent_contract import SCHEMA, build_network
from soptx.ml.substructure.independent_training import train_network


@pytest.mark.parametrize("name, optimizer_type", [
    ("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
])
def test_surrogate_optimizer_matches_torch_step(name, optimizer_type):
    """一次更新应与显式创建的对应优化器一致, 包括衰减与动量."""
    model = torch.nn.Linear(2, 1)
    expected = deepcopy(model)
    x = torch.tensor([[1., 2.], [3., 4.]])
    y = torch.tensor([[0.], [1.]])
    momentum = 0.9 if name == "sgd" else 0.0

    options = {"lr": 0.01, "weight_decay": 0.1}
    if name == "sgd":
        options["momentum"] = momentum
    config = TrainingConfig(1, 2, optimizer=name, optimizer_params=options)
    optimizer = optimizer_type(expected.parameters(), **options)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(expected(x), y).backward()
    optimizer.step()
    train_surrogate(model, x, y, x, y, config)
    for actual, reference in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("name, params", [
    ("unknown", {}), ("adam", {"momentum": 0.0}),
    ("adamw", {"momentum": 0.9}), ("sgd", {"momentum": -1.0}),
    ("sgd", {"momentum": float("nan")}), ("sgd", {"momentum": float("inf")}),
    ("adam", {"weight_decay": -1.0}), ("adam", {"weight_decay": float("nan")}),
    ("adam", {"weight_decay": float("inf")}), ("adam", {"lr": 0.0}),
    ("adam", {"lr": float("inf")}), ("adam", {"lr": "0.01"}),
    ("adam", {"lr": True}), ("adam", {"unknown": 0.1}), ("adam", None),
])
def test_invalid_optimizer_config(name, params):
    """无效参数和不适用的参数名称应明确拒绝."""
    with pytest.raises(ValueError):
        TrainingConfig(1, 2, optimizer=name, optimizer_params=params)


def test_optimizer_params_are_copied_and_serialized_once():
    """配置复制输入字典, 默认值与序列化只使用嵌套参数来源."""
    options = {"lr": 0.01}
    config = TrainingConfig(1, 2, optimizer_params=options)
    options["lr"] = 1.0
    assert config.optimizer_params == {"lr": 0.01, "weight_decay": 0.0}
    saved = json.loads(json.dumps(vars(config)))
    restored = TrainingConfig(**saved)
    assert restored == config
    assert not {"learning_rate", "weight_decay", "momentum"} & saved.keys()
    default = TrainingConfig(1, 2)
    assert default.optimizer == "adam"
    assert default.optimizer_params == {"lr": 1e-3, "weight_decay": 0.0}
    with pytest.raises(TypeError):
        TrainingConfig(1, 2, learning_rate=0.01)


class MatrixCodec:
    """将测试标签还原为矩阵, 不依赖有限元求解."""

    def decode(self, values):
        return values.reshape(-1, 2, 5)


def _provider(meta):
    """只提供 train_network 所需 codecs 与 metadata() 的接口替身."""
    return SimpleNamespace(codecs={"shape": MatrixCodec()}, metadata=lambda: meta)


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd"])
def test_independent_training_records_optimizer(tmp_path, name):
    """微型数据集训练应保存所选优化器的配置及状态."""
    meta = {
        "spatial_dimension": 2, "trace": "linear_corner", "n_fine": [2, 2],
        "n_trace": 8, "n_rigid": 3, "n_i": 2, "n_cells": 4,
        "n_shape_targets": 10, "n_stiffness_targets": 15,
    }
    dataset = tmp_path / "data"
    dataset.mkdir()
    manifest = {"schema": SCHEMA, "complete": True, "provider": meta,
                "input_quantity": "normalized_young_modulus",
                "counts": {"train": 2, "validation": 2}}
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split in ("train", "validation"):
        np.save(dataset / f"{split}_inputs.npy", np.ones((2, 4), dtype=np.float64))
        np.save(dataset / f"{split}_shape_targets.npy", np.ones((2, 10), dtype=np.float64))
    model = build_network(meta, route="shape", num_networks=1,
                          hidden_dims=(8,), activation=torch.nn.Tanh)
    options = {"lr": 0.01}
    if name == "sgd":
        options["momentum"] = 0.9
    config = TrainingConfig(1, 2, optimizer=name, optimizer_params=options)
    output = tmp_path / "training"
    result = train_network(dataset, output, route="shape", network=model,
                           provider=_provider(meta), config=config)
    payload = torch.load(result["checkpoint"], weights_only=True)
    record = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert payload["training_config"]["optimizer"] == name
    assert record["training_config"]["optimizer"] == name
    assert payload["training_config"]["optimizer_params"] == config.optimizer_params
    assert record["training_config"]["optimizer_params"] == config.optimizer_params
    assert not {"learning_rate", "weight_decay", "momentum"} & payload["training_config"].keys()
    state = next(iter(payload["optimizer_state"]["state"].values()))
    assert ("momentum_buffer" if name == "sgd" else "exp_avg") in state
    assert np.isfinite(result["best_validation_loss"])
    assert record["route"] == "shape"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["results"] == {"shape": result}


def test_independent_training_requires_single_route(tmp_path):
    """train_network 只接受单一路线, 缺省或 both 在读取数据与建目录前报错."""
    options = dict(network=torch.nn.Linear(4, 10).double(), provider=_provider({}))
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="shape 或 stiffness"):
        train_network(tmp_path / "missing", output, route="both", **options)
    with pytest.raises(TypeError):
        train_network(tmp_path / "missing", output, **options)
    assert not output.exists()
