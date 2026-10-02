"""定义数据格式与网络结构.

Notes
-----
样本生成, 训练与权重加载共用本模块: 数据集与权重的格式版本 SCHEMA,
提供器元数据的一致性判定, 以及按元数据构建网络并记录其结构.
"""

from __future__ import annotations

import json
from math import prod
from numbers import Integral

import torch
from torch import nn

from .nets import IndependentOutputNet, SplitOutputNet


HIDDEN_DIMS = (60, 80, 100, 120, 140, 160, 180, 200, 180, 160, 140, 120, 100, 80, 60)
ACTIVATIONS = (
    nn.Tanh, nn.ELU, nn.Tanh, nn.ELU, nn.Tanh,
    nn.ELU, nn.Tanh, nn.ELU, nn.ELU, nn.Tanh,
    nn.ELU, nn.Tanh, nn.ELU, nn.Tanh, nn.ELU,
)
SCHEMA = "independent_entries_v1"


def provider_metadata_matches(saved, current):
    """比较提供器元数据, 兼容旧记录隐含的固定材料标度.

    Parameters
    ----------
    saved, current : dict
        已保存及当前的提供器元数据. 仅缺失的 material_scaling 字段
        按旧版固定约定补齐, 其余字段仍严格比较.

    Returns
    -------
    bool
        配置是否一致, 不修改传入记录.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    scaling = {
        "reference_young_modulus": 1.0,
        "penal": 1.0,
        "rho_min": 0.0,
        "stiffness_quantity": "K_physical / E_reference",
        "recovery_quantity": "dimensionless",
    }
    saved = dict(saved)
    current = dict(current)
    saved.setdefault("material_scaling", scaling)
    current.setdefault("material_scaling", scaling)
    return saved == current


def build_network(
    provider_metadata, *, route="shape", seed=2026, num_networks=4,
    hidden_dims=HIDDEN_DIMS, activation=ACTIVATIONS,
):
    """根据接口空间元数据构建一条路线的全连接模型.

    Parameters
    ----------
    provider_metadata : dict
        标签提供器的元数据, 决定输入与独立输出维度.
    route : str
        单条预测路线, 当前支持 shape 或 stiffness.
    seed : int
        模型初始化种子.
    num_networks : int
        该路线的输出拆分网络数量, 两条路线默认均为 4; 为 1 时构建单个
        全连接网络. 
    hidden_dims : tuple of int
        各隐藏层宽度, 层数由其长度决定. 默认 HIDDEN_DIMS 为正式的
        15 隐藏层配置.
    activation : type or sequence of type
        隐藏层激活模块类, 语义同 MLP: 单个类为各层共用, 序列须与
        hidden_dims 等长并逐层取用. 默认 ACTIVATIONS 与 HIDDEN_DIMS 配套.

    Returns
    -------
    nn.Module
        CPU 上的 float64 模型, 可包含多个输出拆分子网络.
        各子网络共用同一隐藏层配置.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    widths = metadata_widths(provider_metadata)
    if (isinstance(num_networks, bool) or not isinstance(num_networks, Integral)
            or num_networks <= 0):
        raise ValueError("num_networks 必须为正整数")
    count = num_networks
    output_dim = widths[f"{route}_targets"]
    if count > output_dim:
        raise ValueError(f"{route} 的网络数量不能超过独立输出数")

    torch.manual_seed(seed)
    # 连续划分输出索引, 余数优先分配给前面的组.
    size, remainder = divmod(output_dim, count)
    groups = []
    start = 0
    for i in range(count):
        stop = start + size + (i < remainder)
        groups.append(tuple(range(start, stop)))
        start = stop
    # 两条路线按网络数量选择相同骨架, 单网络保留直接 MLP 权重键格式.
    model = (
        IndependentOutputNet(
            input_dim=widths["inputs"],
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
        )
        if count == 1
        else SplitOutputNet(
            input_dim=widths["inputs"],
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            output_groups=tuple(groups),
            activation=activation,
        )
    )
    return model.to(dtype=torch.float64)


def activation_names(model):
    """返回模型逐隐藏层的激活类名, 供权重记录与加载核对.

    Parameters
    ----------
    model : nn.Module
        build_network 构建的模型, 须具有 hidden_dims 与 activation_name.

    Returns
    -------
    list of str
        与 hidden_dims 等长; 各层共用单个激活类时按层数展开.
    """
    names = model.activation_name.split(",")
    if len(names) != len(model.hidden_dims):
        names = names * len(model.hidden_dims)
    return names


def write_json(path, data):
    """写入可读的 JSON 元数据.

    Parameters
    ----------
    path : Path
        目标文件路径, 已存在时覆盖.
    data : dict
        可 JSON 序列化的记录, 以 UTF-8 保存且保留中文.
    """
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def metadata_widths(meta):
    """校验二维/三维两种接口空间的维度关系, 返回数据列宽.

    Parameters
    ----------
    meta : dict
        提供器元数据, 即 provider.metadata() 或权重中保存的记录.

    Returns
    -------
    dict
        inputs, shape_targets 与 stiffness_targets 三类数组的列宽.
    """
    dim = meta["spatial_dimension"]
    if dim not in (2, 3) or meta["trace"] not in ("linear_corner", "full_trace"):
        raise ValueError("仅支持二维或三维 linear_corner/full_trace 子结构")
    n_fine = meta["n_fine"]
    if len(n_fine) != dim or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 2
        for n in n_fine
    ):
        raise ValueError("n_fine 必须包含各方向至少为 2 的整数划分")
    n_internal_nodes = prod(n - 1 for n in n_fine)
    n_boundary_nodes = prod(n + 1 for n in n_fine) - n_internal_nodes
    n_trace = dim * (2**dim if meta["trace"] == "linear_corner" else n_boundary_nodes)
    n_free = n_trace - meta["n_rigid"]
    if (meta["n_trace"] != n_trace
            or meta["n_rigid"] != dim * (dim + 1) // 2
            or meta["n_i"] != dim * n_internal_nodes
            or meta["n_cells"] != prod(n_fine)
            or meta["n_shape_targets"] != meta["n_i"] * n_free
            or meta["n_stiffness_targets"] != n_free * (n_free + 1) // 2):
        raise ValueError("子结构独立分量维度不一致")
    return {
        "inputs": meta["n_cells"],
        "shape_targets": meta["n_shape_targets"],
        "stiffness_targets": meta["n_stiffness_targets"],
    }
