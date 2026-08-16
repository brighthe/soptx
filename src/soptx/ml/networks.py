"""与具体物理问题无关的神经网络骨干."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """由线性层和逐层激活函数构成的全连接前馈网络.

    输出层不添加激活函数, 以便回归, 物理残差和算子预测等任务自行定义输出语义.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (),
        activation: Callable[[], nn.Module] = nn.Tanh,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """初始化 MLP.

        参数:
            input_dim: 单个样本的输入特征数.
            output_dim: 单个样本的输出特征数.
            hidden_dims: 各隐藏层的宽度. 空元组表示无隐藏层.
            activation: 隐藏层激活模块的工厂. 每个隐藏层调用一次.
            dtype: 线性层参数的数据类型.
            device: 线性层参数所在的设备.
        """
        super().__init__()
        dimensions = (input_dim,) + hidden_dims + (output_dim,)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("input_dim, output_dim 与 hidden_dims 必须全部为正整数")

        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(
                nn.Linear(
                    in_features,
                    out_features,
                    dtype=dtype,
                    device=device,
                )
            )
            if index < len(dimensions) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算网络输出.

        参数:
            x: 形状为 ``(..., input_dim)`` 的输入张量.

        返回:
            形状为 ``(..., output_dim)`` 的线性输出张量.
        """
        return self.net(x)
