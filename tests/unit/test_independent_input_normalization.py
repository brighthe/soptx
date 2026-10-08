"""检查独立分量网络的输入尺度不变性, 梯度及权重恢复契约."""

import pytest
import torch

from soptx.ml.substructure.independent_checkpoints import load_independent_network
from soptx.ml.substructure.independent_contract import (
    SCHEMA, activation_names, build_network,
)


@pytest.fixture
def metadata():
    """提供三维小网格的独立分量尺寸契约."""
    return dict(spatial_dimension=3, trace="linear_corner", n_fine=(2, 2, 2),
                n_trace=24, n_rigid=6, n_i=3, n_cells=8,
                n_shape_targets=54, n_stiffness_targets=171)


@pytest.mark.parametrize("count", [1, 4])
def test_scale_invariance_and_directional_gradient(metadata, count):
    """检查不同材料尺度下输出相同, 且梯度与方向有限差分一致."""
    net = build_network(metadata, num_networks=count, hidden_dims=(8,),
                        activation=torch.nn.Tanh, input_normalization="per_sample_max")
    x = torch.linspace(.1, .9, 16, dtype=torch.float64).reshape(2, 8)
    torch.testing.assert_close(net(x), net(.01 * x), rtol=1e-12, atol=1e-12)
    before = x.clone()
    x.requires_grad_()
    direction = torch.linspace(-.1, .2, 16, dtype=torch.float64).reshape(2, 8)
    derivative = (torch.autograd.grad(net(x).square().sum(), x)[0] * direction).sum()
    epsilon = 1e-6
    finite = (net(x + epsilon * direction).square().sum()
              - net(x - epsilon * direction).square().sum()) / (2 * epsilon)
    torch.testing.assert_close(derivative, finite, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("mode", ["none", "per_sample_max"])
def test_checkpoint_preserves_input_processing(metadata, count, mode, tmp_path):
    """旧权重缺失字段按原始输入恢复, 新权重保持预测及归一化契约."""
    net = build_network(metadata, num_networks=count, hidden_dims=(8,),
                        activation=torch.nn.Tanh, input_normalization=mode)
    architecture = dict(input_dim=8, output_dim=54, hidden_dims=net.hidden_dims,
                        activations=activation_names(net), num_networks=count,
                        output_groups=getattr(net, "output_groups", (tuple(range(54)),)),
                        model_class=type(net).__name__, dtype="float64")
    payload = dict(schema=SCHEMA, route="shape", architecture=architecture,
                   model_state=net.state_dict(), dataset=dict(complete=True,
                   input_quantity="normalized_young_modulus", provider=metadata))
    if mode != "none":
        payload["input_normalization"] = mode
    path = tmp_path / "shape_best.pt"
    torch.save(payload, path)
    restored, source = load_independent_network(path, metadata, route="shape")
    x = torch.linspace(.1, .9, 8, dtype=torch.float64)
    torch.testing.assert_close(restored(x), net(x), rtol=0, atol=0)
    assert source["input_normalization"] == mode


def test_rejects_unimplemented_stiffness_scaling(metadata):
    """直接刚度目标带材料尺度, 禁止未经输出尺度恢复的归一化."""
    with pytest.raises(ValueError, match="仅支持 shape"):
        build_network(metadata, route="stiffness", input_normalization="per_sample_max")
