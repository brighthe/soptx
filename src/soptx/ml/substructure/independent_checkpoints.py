"""独立条目训练权重的校验与恢复."""

from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

from soptx.fem.substructure.independent_targets import IndependentTargetProvider

from soptx.ml.substructure.independent_contract import (
    SCHEMA, activation_names, build_network, provider_metadata_matches,
)


def decoder_metadata_matches(saved, current):
    """比较在线 decoder 与训练权重的局部配置及独立条目契约.

    Parameters
    ----------
    saved, current : dict
        权重保存的元数据与当前 decoder.metadata() 返回的元数据.

    Returns
    -------
    bool
        配置及独立条目编号是否相容, 不修改传入记录.

    Notes
    -----
    材料标度复用 provider_metadata_matches 的旧记录兼容规则.
    cell_size 使用 rtol=1e-12, atol=0 的几何容差; 刚体基及内部取值使用
    元数据中的 roundtrip_tolerance. 其他字段, 包括独立条目编号, 严格比较.
    """
    import numpy as np

    if provider_metadata_matches(saved, current):
        return True
    tolerance = saved.get(
        "roundtrip_tolerance",
        current.get("roundtrip_tolerance", {"rtol": 1e-8, "atol": 1e-10}),
    )
    rtol = float(tolerance["rtol"])
    atol = float(tolerance["atol"])
    saved_normalized = dict(saved)
    current_normalized = dict(current)
    if "cell_size" not in saved or "cell_size" not in current:
        return False
    saved_cell_size = np.asarray(saved["cell_size"], dtype=np.float64)
    current_cell_size = np.asarray(current["cell_size"], dtype=np.float64)
    if (
        saved_cell_size.shape != current_cell_size.shape
        or not np.allclose(
            saved_cell_size, current_cell_size, rtol=1e-12, atol=0.0,
        )
    ):
        return False
    saved_normalized["cell_size"] = current_normalized["cell_size"]
    for name in ("rigid_basis", "rigid_interior"):
        if name not in saved or name not in current:
            return False
        saved_value = np.asarray(saved[name], dtype=np.float64)
        current_value = np.asarray(current[name], dtype=np.float64)
        if (
            saved_value.shape != current_value.shape
            or not np.allclose(saved_value, current_value, rtol=rtol, atol=atol)
        ):
            return False
        saved_normalized[name] = current_normalized[name]
    return provider_metadata_matches(saved_normalized, current_normalized)


def _saved_hidden_layers(architecture, path):
    """从权重结构记录恢复隐藏层宽度与逐层激活类.

    Parameters
    ----------
    architecture : dict
        权重中的 architecture 记录, 须含 hidden_dims 与 activations.
    path : Path
        权重路径, 仅用于报错信息.

    Returns
    -------
    hidden_dims : tuple of int
        各隐藏层宽度.
    activation : tuple of type
        与 hidden_dims 等长的激活类序列.

    Notes
    -----
    激活类名只在 torch.nn 的激活模块中解析, 不执行任意导入.
    """
    hidden_dims = architecture.get("hidden_dims")
    names = architecture.get("activations")
    if (not isinstance(hidden_dims, (list, tuple))
            or not all(isinstance(width, int) and not isinstance(width, bool)
                       for width in hidden_dims)
            or not isinstance(names, (list, tuple))
            or len(names) != len(hidden_dims)):
        raise ValueError(f"权重缺少有效的隐藏层宽度或逐层激活记录: {path}")
    activation = []
    for name in names:
        factory = getattr(nn, name, None) if isinstance(name, str) else None
        if not (isinstance(factory, type) and issubclass(factory, nn.Module)
                and factory.__module__ == "torch.nn.modules.activation"):
            raise ValueError(f"权重记录了无法识别的激活类 {name!r}: {path}")
        activation.append(factory)
    return tuple(hidden_dims), tuple(activation)


def load_independent_provider_metadata(checkpoint_path, *, route=None):
    """从独立条目权重读取可恢复子结构的提供器元数据.

    Parameters
    ----------
    checkpoint_path : str or Path
        已保存的独立条目权重文件.
    route : {"shape", "stiffness"} or None
        可选的预期预测路线. 指定时同时校验权重路线.

    Returns
    -------
    dict
        数据集记录中的提供器元数据副本.
    """
    if route is not None and route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape、stiffness 或 None")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少权重: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"不支持的独立条目 checkpoint 格式: {path}")
    if route is not None and payload.get("route") != route:
        raise ValueError(f"权重的预测路线不匹配: {path}")
    dataset = payload.get("dataset")
    if (not isinstance(dataset, dict) or dataset.get("complete") is not True
            or dataset.get("input_quantity") != "normalized_young_modulus"):
        raise ValueError(f"权重的数据集记录未完成或格式不支持: {path}")
    metadata = dataset.get("provider")
    required = {
        "cell_size", "n_fine", "poisson_ratio", "trace", "spatial_dimension",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError(f"权重缺少可恢复的子结构配置: {path}")
    if metadata["spatial_dimension"] == 2 and "material_hypothesis" not in metadata:
        raise ValueError(f"二维权重缺少 material_hypothesis: {path}")
    return dict(metadata)


def load_independent_network(checkpoint_path, provider_metadata, *, route):
    """加载单条独立分量路线的权重并核对接口契约.

    Parameters
    ----------
    checkpoint_path : str or Path
        当前路线的权重文件, 不要求另一条路线同时存在.
    provider_metadata : dict
        当前子结构配置和独立分量编号.
    route : str
        shape 或 stiffness.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        CPU float64 网络及权重来源记录.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    name = route
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少权重: {path}")
    # 独立条目格式与 artifacts.py 的通用 signature 格式不同.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"不支持的独立条目 checkpoint 格式: {path}")
    dataset = payload.get("dataset", {})
    if (payload.get("route") != name
            or dataset.get("complete") is not True
            or not provider_metadata_matches(dataset.get("provider"), provider_metadata)
            or dataset.get("input_quantity") != "normalized_young_modulus"):
        raise ValueError(f"权重的路线、接口空间或材料/离散配置不匹配: {path}")
    architecture = payload.get("architecture", {})
    if architecture.get("model_class") is None:
        # 根据旧权重的实际布局识别结构; 不猜测原始 Python 类名.
        state = payload.get("model_state")
        if not isinstance(state, dict) or not state:
            raise ValueError(f"旧权重缺少可用于识别网络结构的 model_state: {path}")
        parameter_keys = [key for key in state if key != "restore_order"]
        if ("restore_order" in state and parameter_keys
                and all(key.startswith("nets.") for key in parameter_keys)):
            inferred_class = "SplitOutputNet"
        elif all(key.startswith("net.") for key in state):
            inferred_class = "IndependentOutputNet"
        else:
            raise ValueError(f"无法从旧权重可靠识别网络结构: {path}")
        architecture = dict(architecture, model_class=inferred_class)
    if architecture.get("output_groups") is None:
        state = payload.get("model_state")
        output_dim = provider_metadata[f"n_{name}_targets"]
        if (architecture.get("model_class") not in {"IndependentOutputNet", "DirectStiffnessNet"}
                or architecture.get("num_networks") not in (None, 1)
                or not isinstance(state, dict) or not state
                or not all(key.startswith("net.") for key in state)
                or architecture.get("output_dim") != output_dim):
            raise ValueError(f"缺少输出分组且无法确认单网络结构: {path}")
        # 旧单网络直接按标签顺序输出全部条目, 无分组或排序缓冲区.
        architecture = dict(architecture, num_networks=1,
                            output_groups=(tuple(range(output_dim)),))
    count = architecture.get("num_networks")
    if count is None:
        # 旧权重未登记数量时, 交叉核对输出分组与实际子网络编号.
        saved_groups = architecture.get("output_groups")
        state = payload.get("model_state")
        if not isinstance(saved_groups, (list, tuple)) or not saved_groups or not isinstance(state, dict):
            raise ValueError(f"旧权重缺少可用于恢复网络数量的输出分组或权重: {path}")
        indices = set()
        for key in state:
            if key.startswith("nets."):
                parts = key.split(".", 2)
                if len(parts) != 3 or not parts[1].isdigit():
                    raise ValueError(f"旧权重的子网络编号无效: {path}")
                indices.add(int(parts[1]))
        count = len(saved_groups)
        if indices:
            if indices != set(range(count)):
                raise ValueError(f"旧权重的子网络编号与输出分组数量不一致: {path}")
        elif count != 1 or architecture.get("model_class") not in {
            "DirectStiffnessNet", "IndependentOutputNet",
        }:
            raise ValueError(f"无法从旧权重可靠恢复网络数量: {path}")
        # 后续仍严格校验分组覆盖、矩阵尺寸及全部 state_dict 键.
        architecture = dict(architecture, num_networks=count)
    hidden_dims, activation = _saved_hidden_layers(architecture, path)
    try:
        model = build_network(
            provider_metadata, route=name, num_networks=count,
            hidden_dims=hidden_dims, activation=activation,
            input_normalization=payload.get("input_normalization", "none"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"无法按权重记录重建网络结构: {path}") from error
    output_dim = provider_metadata[f"n_{name}_targets"]
    expected_groups = getattr(model, "output_groups", (tuple(range(output_dim)),))
    groups = tuple(tuple(group) for group in architecture.get("output_groups", ()))
    allowed_classes = {type(model).__name__}
    if count == 1 and name == "stiffness":
        allowed_classes.add("DirectStiffnessNet")
    if count == 1 and name == "shape":
        allowed_classes.add("SplitOutputNet")
    if name == "shape":
        allowed_classes.add("SplitShapeFunctionNet")
    if (architecture.get("input_dim") != provider_metadata["n_cells"]
            or architecture.get("output_dim") != output_dim
            or tuple(model.hidden_dims) != hidden_dims
            or activation_names(model) != list(architecture["activations"])
            or architecture.get("dtype") != "float64"
            or architecture.get("model_class") not in allowed_classes
            or groups != expected_groups):
        raise ValueError(f"权重的网络结构或输出分组不匹配: {path}")
    state = payload["model_state"]
    if count == 1 and architecture.get("model_class") in {
        "SplitOutputNet", "SplitShapeFunctionNet",
    }:
        # 旧单网络形函数模型仅含一个完整输出组, 校验后移除包装层.
        order = state.get("restore_order")
        if (order is None or order.dtype != torch.long
                or not torch.equal(order, torch.arange(output_dim))):
            raise ValueError(f"旧单网络权重的输出索引顺序不匹配: {path}")
        if any(key != "restore_order" and not key.startswith("nets.0.")
               for key in state):
            raise ValueError(f"旧单网络权重包含非预期键: {path}")
        state = {key.removeprefix("nets.0."): value
                 for key, value in state.items() if key != "restore_order"}
    model.load_state_dict(state, strict=True)
    if hasattr(model, "restore_order"):
        expected_order = torch.argsort(torch.tensor([i for group in groups for i in group]))
        if not torch.equal(model.restore_order, expected_order):
            raise ValueError(f"权重的输出索引顺序不匹配: {path}")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError(f"权重包含非有限值: {path}")
    model.eval()
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    source = {
        "path": str(path.resolve()), "sha256": digest.hexdigest(),
        "epoch": payload.get("epoch"),
        "validation_loss": payload.get("validation_loss"),
        "loss": payload.get("loss", "decoded_full_matrix_mse"),
        "loss_settings": payload.get("loss_settings"),
        "architecture": architecture,
        "input_normalization": model.input_normalization,
        "dataset_seed": dataset.get("seed"),
        "dataset_sampling": dataset.get("sampling"),
    }
    return model, source


def load_analysis_provider(checkpoint_dir):
    """从形函数权重恢复整体分析使用的子结构和接口空间.

    Parameters
    ----------
    checkpoint_dir : str or Path
        包含 shape_best.pt 的训练结果目录.

    Returns
    -------
    IndependentTargetProvider
        与形函数权重元数据严格一致的标签提供器.
    """
    path = Path(checkpoint_dir) / "shape_best.pt"
    saved = load_independent_provider_metadata(path, route="shape")
    provider = IndependentTargetProvider(
        cell_size=tuple(saved["cell_size"]),
        n_fine=tuple(saved["n_fine"]),
        nu=saved["poisson_ratio"],
        trace_kind=saved["trace"],
        hypothesis=saved.get("material_hypothesis"),
    )
    if not provider_metadata_matches(saved, provider.metadata()):
        raise ValueError("恢复的子结构配置或独立分量编号与形函数权重不一致")
    return provider


def load_analysis_networks(checkpoint_dir, provider_metadata, *, route="shape"):
    """加载整体求解所需的网络, 刚度路线同时加载形函数用于位移恢复.

    Parameters
    ----------
    checkpoint_dir : str or Path
        包含 shape_best.pt 及所需 stiffness_best.pt 的目录.
    provider_metadata : dict
        当前子结构配置和独立分量编号.
    route : str
        shape 或 stiffness, 默认 shape.

    Returns
    -------
    tuple[dict, dict]
        网络字典与对应的权重来源记录.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    required = ("shape",) if route == "shape" else ("shape", "stiffness")
    networks, sources = {}, {}
    for name in required:
        networks[name], sources[name] = load_independent_network(
            Path(checkpoint_dir) / f"{name}_best.pt", provider_metadata, route=name,
        )
    return networks, sources
