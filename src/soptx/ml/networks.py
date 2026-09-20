"""与具体物理问题无关的神经网络骨干."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn


#: 隐藏层激活的规格: 单个模块工厂表示全部隐藏层共用, 工厂序列表示逐层取用.
ActivationSpec = Callable[[], nn.Module] | Sequence[Callable[[], nn.Module]]


def activation_signature(activation: ActivationSpec) -> str:
    """把激活规格归约为可登记进 checkpoint 的字符串.

    参数:
        activation: 单个激活模块工厂, 或逐层的工厂序列.

    返回:
        单一工厂返回其类名, 如 ``"SiLU"``; 工厂序列返回逗号连接的类名序列,
        如 ``"Tanh,ELU,Tanh"``.

    说明:
        匿名工厂 (如 lambda) 得到的名字不具可比性, 架构校验会因此失效;
        登记进 checkpoint 的网络应使用具名的激活模块类.
    """
    if isinstance(activation, Sequence):
        return ",".join(_factory_name(factory) for factory in activation)
    return _factory_name(activation)


def _factory_name(factory: Callable[[], nn.Module]) -> str:
    """取激活模块工厂的名字."""
    return getattr(factory, "__name__", None) or type(factory).__name__


class MLP(nn.Module):
    """由线性层和逐层激活函数构成的全连接前馈网络.

    输出层不添加激活函数, 以便回归, 物理残差和算子预测等任务自行定义输出语义.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (),
        activation: Callable[[], nn.Module]
        | Sequence[Callable[[], nn.Module]] = nn.Tanh,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """初始化 MLP.

        参数:
            input_dim: 单个样本的输入特征数.
            output_dim: 单个样本的输出特征数.
            hidden_dims: 各隐藏层的宽度. 空元组表示无隐藏层.
            activation: 隐藏层激活模块的工厂. 传入单个工厂时所有隐藏层共用;
                传入工厂序列时逐层取用, 其长度须与 hidden_dims 相同.
                每个工厂对应的隐藏层调用一次.
            dtype: 线性层参数的数据类型.
            device: 线性层参数所在的设备.
        """
        super().__init__()
        dimensions = (input_dim,) + hidden_dims + (output_dim,)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("input_dim, output_dim 与 hidden_dims 必须全部为正整数")

        if isinstance(activation, Sequence):
            activations = tuple(activation)
            if len(activations) != len(hidden_dims):
                raise ValueError(
                    "activation 为序列时其长度须与 hidden_dims 相同, "
                    f"当前分别为 {len(activations)} 与 {len(hidden_dims)}"
                )
        else:
            activations = (activation,) * len(hidden_dims)

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
                layers.append(activations[index]())
        self.net = nn.Sequential(*layers)

        #: 各隐藏层宽度. 与 activation_name 一并作为架构的唯一事实来源,
        #: 供 checkpoint 签名登记与校验.
        self.hidden_dims = tuple(hidden_dims)
        #: 激活规格的字符串登记形式.
        self.activation_name = activation_signature(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算网络输出.

        参数:
            x: 形状为 ``(..., input_dim)`` 的输入张量.

        返回:
            形状为 ``(..., output_dim)`` 的线性输出张量.
        """
        return self.net(x)
