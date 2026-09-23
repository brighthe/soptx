"""用于子结构静力缩聚与形函数预测的 PyTorch 神经网络代理模型."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn as nn

from ..networks import MLP, ActivationSpec, activation_signature


class SubstructureSurrogateNet(MLP):
    """两条子结构代理路线共用的网络骨架.

    相对骨干 ``MLP`` 只收紧三处: ``hidden_dims`` 由可选改为必填, 默认激活由
    ``nn.Tanh`` 改为 ``nn.SiLU``, 不透传 ``dtype`` 与 ``device``. 这三处对两条
    路线完全一致, 因此构造只写在本类; 子类之间没有结构差异, 只有 ``output_dim``
    的语义差异, 由各自的类 docstring 给出.

    本类不直接实例化: 路线信息只存在于子类名与调用点, 网络本身不做校验, 传错
    ``output_dim`` 不会被网络拦下, 只会在下游缩聚器重构时暴露.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: ActivationSpec = nn.SiLU,
    ) -> None:
        """初始化子结构代理网络.

        参数:
            input_dim: 展平后的子结构单元密度特征数.
            output_dim: 输出特征数, 其语义由子类给出.
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


class ReducedStiffnessSurrogateNet(SubstructureSurrogateNet):
    """用于预测子结构降阶刚度 Cholesky 因子下三角独立条目的代理网络.

    ``output_dim`` 为 ``n_r * (n_r + 1) // 2``, 即变形子空间上 Cholesky 因子的
    下三角独立条目数. 分解对象不是 Schur 补 ``K_s`` 本身, 而是它在变形子空间上
    的限制 ``R^T K_s R``, 其中 ``R`` 为刚体模态的正交补; 限制后的算子严格正定,
    Cholesky 分解无需正则项. 重构见 ``ReducedStiffnessCondensation``.
    """


class ShapeFunctionSurrogateNet(SubstructureSurrogateNet):
    """用于预测形函数在变形子空间上分量的代理网络.

    ``output_dim`` 为 ``n_i * n_r``, 输出被 reshape 为 ``(n_i, n_r)`` 的 ``M``,
    由有限元形函数缩聚器与固定的刚体分量合成完整形函数. 重构见
    ``ShapeFunctionCondensation``.
    """


class SplitOutputNet(nn.Module):
    """使用多个独立网络分组预测独立输出分量.

    Parameters
    ----------
    input_dim : int
        每个样本的输入特征数, 各网络共用相同输入.
    output_dim : int
        所选路线的独立输出分量总数.
    hidden_dims : tuple[int, ...]
        各隐藏层宽度, 所有子网络使用相同配置.
    output_groups : Sequence[Sequence[int]]
        各网络负责的标准输出索引, 必须无重复且完整覆盖输出.
    activation : ActivationSpec
        隐藏层激活模块工厂或逐层工厂序列.

    Notes
    -----
    网络数量由 output_groups 决定, 可用于形函数或直接刚度路线.
    输出排列必须与训练标签及约束补全器一致, 本类不执行约束补全.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        output_groups: Sequence[Sequence[int]],
        activation: ActivationSpec,
    ) -> None:
        super().__init__()
        groups = tuple(tuple(group) for group in output_groups)
        if output_dim <= 0:
            raise ValueError("output_dim 必须为正整数")
        if not groups or any(not group for group in groups):
            raise ValueError("必须提供非空输出分组")

        flat_indices = [index for group in groups for index in group]
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in flat_indices
        ):
            raise ValueError("输出索引必须为整数")
        if sorted(flat_indices) != list(range(output_dim)):
            raise ValueError("输出分组必须无重复且完整覆盖标准输出索引")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_groups = groups
        self.hidden_dims = tuple(hidden_dims)
        self.activation_name = activation_signature(activation)
        self.nets = nn.ModuleList(
            MLP(
                input_dim=input_dim,
                output_dim=len(group),
                hidden_dims=self.hidden_dims,
                activation=activation,
            )
            for group in groups
        )

        # 拼接结果按分组排列, 用逆置换恢复训练标签的标准顺序.
        restore_order = torch.argsort(
            torch.tensor(flat_indices, dtype=torch.long)
        )
        self.register_buffer("restore_order", restore_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """预测并组合独立输出分量.

        Parameters
        ----------
        x : torch.Tensor
            形状为 (..., input_dim) 的输入特征.

        Returns
        -------
        torch.Tensor
            形状为 (..., output_dim) 的标准顺序独立分量.
        """
        grouped_output = torch.cat([net(x) for net in self.nets], dim=-1)
        return grouped_output.index_select(-1, self.restore_order)


class SplitShapeFunctionNet(SplitOutputNet):
    """保留形函数分组网络的原有接口, 实现复用通用分组网络."""


class DirectStiffnessNet(SubstructureSurrogateNet):
    """直接预测缩聚刚度的独立条目.

    Parameters
    ----------
    input_dim : int
        每个样本的输入特征数.
    output_dim : int
        独立刚度条目数, 三维角点接口配置取 171.
    hidden_dims : tuple[int, ...]
        各隐藏层宽度, 由调用端指定.
    activation : ActivationSpec, optional
        隐藏层激活模块工厂或逐层工厂序列, 默认 nn.SiLU.
        15 隐藏层配置须显式传入对应序列.

    Notes
    -----
    复用 SubstructureSurrogateNet 的构造和前向计算.
    输出是独立刚度条目, 不是 Cholesky 因子. 对称性与刚体零空间
    约束由后续补全器施加, 不能使用 ReducedStiffnessCondensation
    的 Cholesky 重构逻辑解码.
    """
