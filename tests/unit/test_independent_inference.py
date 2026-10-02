"""独立条目网络公共推理接口的设备、精度及输出契约测试."""

import numpy as np
import pytest
import torch
from torch import nn

from soptx.ml.substructure.inference import predict_independent_outputs


def test_prediction_preserves_entries_and_uses_eval_mode():
    """推理输出保留网络独立条目, 并使用 CPU float64 评估模式."""
    model = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]], dtype=torch.float64))
        model.bias.fill_(0.25)
    model.train()
    inputs = np.array([[0.4, 0.3], [0.2, 0.6]], dtype=np.float64)
    predicted = predict_independent_outputs(model, inputs, "shape")
    np.testing.assert_allclose(predicted, [[0.75], [0.05]], atol=1e-14)
    assert predicted.dtype == np.float64
    assert not model.training


@pytest.mark.parametrize("device,dtype,message", [
    ("cpu", torch.float32, "float64"),
    ("meta", torch.float64, "CPU"),
])
def test_prediction_rejects_incompatible_parameters(device, dtype, message):
    """设备或参数精度不相容时在网络执行前拒绝."""
    model = nn.Linear(2, 1, device=device, dtype=dtype)
    with pytest.raises(ValueError, match=message):
        predict_independent_outputs(model, np.ones((2, 2)), "stiffness")


@pytest.mark.parametrize("invalid_output,message", [
    ("shape", "形状"),
    ("nonfinite", "非有限值"),
])
def test_prediction_rejects_invalid_outputs(invalid_output, message):
    """输出须为匹配 batch 的二维有限矩阵."""
    class InvalidOutputNet(nn.Module):
        """返回预先指定的非法输出."""

        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, inputs):
            if invalid_output == "shape":
                return torch.ones(len(inputs), dtype=inputs.dtype)
            return torch.full((len(inputs), 1), float("nan"), dtype=inputs.dtype)

    with pytest.raises(ValueError, match=message):
        predict_independent_outputs(InvalidOutputNet(), np.ones((2, 2)), "shape")
