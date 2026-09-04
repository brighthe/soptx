"""用于子结构静力缩聚与形函数预测的 PyTorch 神经网络代理模型."""

from __future__ import annotations

import torch
import torch.nn as nn

from .networks import MLP


class PIMLSurrogateNet(MLP):
    """用于预测子结构 Cholesky 下三角独立条目的 PyTorch MLP 代理网络."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        """初始化 PIML 子结构刚度代理网络.

        参数:
            input_dim: 展平后的子结构单元密度特征数.
            output_dim: Cholesky 下三角因子独立条目的数量.
            hidden_dim: 两个 SiLU 隐藏层的共同宽度.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            activation=nn.SiLU,
        )


class ShapeFunctionSurrogateNet(MLP):
    """用于预测形函数在变形子空间上分量的 PyTorch MLP 代理网络.

    输出被 reshape 为 ``(n_i, m)`` 的 ``M``, 由有限元形函数缩聚器与固定的
    刚体分量合成完整形函数. 网络结构与 ``PIMLSurrogateNet`` 保持一致.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        """初始化形函数代理网络.

        参数:
            input_dim: 展平后的子结构单元密度特征数.
            output_dim: ``n_i * m``, 即变形子空间上形函数分量的个数.
            hidden_dim: 两个 SiLU 隐藏层的共同宽度.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            activation=nn.SiLU,
        )
