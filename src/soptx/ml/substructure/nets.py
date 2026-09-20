"""用于子结构静力缩聚与形函数预测的 PyTorch 神经网络代理模型."""

from __future__ import annotations

import torch.nn as nn

from ..networks import MLP, ActivationSpec


class PIMLSurrogateNet(MLP):
    """用于预测子结构降阶刚度 Cholesky 因子下三角独立条目的 PyTorch MLP 代理网络.

    分解对象不是 Schur 补 ``K_s`` 本身, 而是它在变形子空间上的限制 ``R^T K_s R``,
    其中 ``R`` 为刚体模态的正交补; 限制后的算子严格正定, Cholesky 分解无需正则项.
    与 ShapeFunctionSurrogateNet 的网络结构一致, 仅 output_dim 的语义不同.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: ActivationSpec = nn.SiLU,
    ) -> None:
        """初始化 PIML 子结构刚度代理网络.

        参数:
            input_dim: 展平后的子结构单元密度特征数.
            output_dim: ``R^T K_s R`` 的 Cholesky 因子下三角独立条目数.
            hidden_dims: 各隐藏层的宽度, 层数由其长度决定. 不设默认值: 层数与宽度
                均须由调用点写明.
            activation: 隐藏层激活模块的工厂或工厂序列, 语义同 MLP.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
        )


class ShapeFunctionSurrogateNet(MLP):
    """用于预测形函数在变形子空间上分量的 PyTorch MLP 代理网络.

    输出被 reshape 为 ``(n_i, m)`` 的 ``M``, 由有限元形函数缩聚器与固定的
    刚体分量合成完整形函数. 网络结构与 PIMLSurrogateNet 保持一致.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: ActivationSpec = nn.SiLU,
    ) -> None:
        """初始化形函数代理网络.

        参数:
            input_dim: 展平后的子结构单元密度特征数.
            output_dim: ``n_i * m``, 即变形子空间上形函数分量的个数.
            hidden_dims: 各隐藏层的宽度, 层数由其长度决定. 不设默认值: 层数与宽度
                均须由调用点写明.
            activation: 隐藏层激活模块的工厂或工厂序列, 语义同 MLP.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
        )
