"""验证刚度辅助监督的物理重构, 相对尺度及梯度累积."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from soptx.backend import backend_manager as bm
from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.ml.substructure.independent_contract import build_network
from soptx.ml.substructure.independent_training import (
    _epoch, _reconstructed_stiffness, _relative_stiffness_loss, _stiffness_operators,
)


@pytest.fixture
def sample_problem():
    """提供小网格的真实材料, 精确标签及能量算子."""
    bm.set_backend("numpy")
    provider = IndependentTargetProvider(cell_size=(2., 2., 2.), n_fine=(2, 2, 2))
    inputs = np.random.default_rng(2045).uniform(.02, 1., (5, 8))
    inputs[0] *= 1e-4
    exact = provider.exact_matrices(inputs)
    labels = provider(inputs)
    operators = _stiffness_operators(provider.metadata(), torch.device("cpu"))
    return provider, inputs, exact, labels, operators


def test_element_energy_matches_exact_stiffness(sample_problem):
    """精确形函数的单元能量重构必须匹配精确静力缩聚刚度."""
    _, x, exact, _, operators = sample_problem
    reconstructed = _reconstructed_stiffness(torch.from_numpy(exact["shape"]),
                                             torch.from_numpy(x), operators)
    torch.testing.assert_close(reconstructed, torch.from_numpy(exact["stiffness"]),
                               rtol=1e-11, atol=1e-12)


def test_relative_stiffness_loss_scale_and_gradient(sample_problem):
    """同比材料缩放不改变相对损失, 形函数梯度通过有限差分检查."""
    _, x, exact, _, operators = sample_problem
    rng = np.random.default_rng(2046)
    shape = torch.from_numpy(exact["shape"] + rng.normal(0., .02, exact["shape"].shape))
    shape.requires_grad_()
    values = torch.from_numpy(x)
    reference = torch.from_numpy(exact["stiffness"])
    def loss(t, scale=1.):
        return _relative_stiffness_loss(_reconstructed_stiffness(t, scale * values, operators),
                                        scale * reference)
    torch.testing.assert_close(loss(shape), loss(shape, .01), rtol=1e-12, atol=1e-14)
    direction = torch.from_numpy(rng.normal(0., .1, shape.shape))
    analytic = (torch.autograd.grad(loss(shape), shape)[0] * direction).sum()
    eps = 1e-6
    finite = (loss(shape.detach() + eps * direction)
              - loss(shape.detach() - eps * direction)) / (2 * eps)
    torch.testing.assert_close(analytic, finite, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("weight, hard", [(0., False), (1., False), (1., True)])
def test_microbatch_keeps_optimizer_batch(sample_problem, weight, hard):
    """末块不足长度时仍按样本加权, 单次更新与整批更新一致."""
    provider, x, _, labels, operators = sample_problem
    full = build_network(provider.metadata(), num_networks=1, hidden_dims=(8,),
                         activation=torch.nn.Tanh, input_normalization="per_sample_max")
    chunked = deepcopy(full)
    arguments = dict(stiffness_targets=labels["stiffness"],
                     stiffness_codec=provider.stiffness_codec, operators=operators,
                     stiffness_loss_weight=weight,
                     stiffness_sample_weights=(np.array([5., 1., 1., 5., 1.]) / 2.6 if hard else None))
    metrics = []
    for net, chunk in ((full, None), (chunked, 2)):
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
        metrics.append(_epoch(net, provider.shape_codec, x, labels["shape"],
                              np.arange(5), 5, torch.device("cpu"), optimizer,
                              loss_chunk_size=chunk, return_components=True, **arguments))
    for first, second in zip(full.parameters(), chunked.parameters()):
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
    for key in metrics[0]:
        assert metrics[0][key] == pytest.approx(metrics[1][key], rel=1e-12, abs=1e-12)


def test_weighted_relative_loss():
    """不同样本权重改变相对误差目标, 不按当前块重新归一化."""
    exact = torch.ones((3, 2, 2), dtype=torch.float64)
    prediction = exact * torch.tensor([1.1, 1.2, 1.3])[:, None, None]
    weights = torch.tensor([5., 1., 1.], dtype=torch.float64) / (7. / 3.)
    relative = (prediction - exact).square().mean(dim=(-2, -1))
    torch.testing.assert_close(_relative_stiffness_loss(prediction, exact, weights),
                               (relative * weights).mean())
    with pytest.raises(ValueError):
        _relative_stiffness_loss(prediction, exact, torch.ones(2))


def test_training_weights_provenance(tmp_path):
    """拒绝其他数据集或被改动的训练权重, 保存全局均值归一化语义."""
    import hashlib
    import json
    from soptx.ml.substructure.independent_training import _load_training_stiffness_weights

    dataset = tmp_path / "samples"
    dataset.mkdir()
    manifest = dataset / "manifest.json"
    manifest.write_text('{"complete": true}', encoding="utf-8")
    path = tmp_path / "weights.npy"
    np.save(path, np.array([1., 5.]))
    source = {"schema": "training_stiffness_weights_v1", "selection_split": "train",
              "counts": 2, "dataset_dir": str(dataset),
              "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(source), encoding="utf-8")
    weights, record = _load_training_stiffness_weights(path, dataset, 2)
    np.testing.assert_allclose(weights, [1. / 3., 5. / 3.])
    assert record["validation_weighting"] == "none"
    with pytest.raises(ValueError):
        _load_training_stiffness_weights(path, dataset, 3)
    np.save(path, np.array([1., 4.]))
    with pytest.raises(ValueError):
        _load_training_stiffness_weights(path, dataset, 2)
