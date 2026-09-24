"""独立条目训练权重的校验与恢复."""

from hashlib import sha256
from pathlib import Path

import torch

from soptx.ml.substructure.independent_training import (
    ACTIVATIONS, HIDDEN_DIMS, SCHEMA, build_network,
)


def load_analysis_networks(checkpoint_dir, provider_metadata, *, route="both"):
    """从最佳权重恢复求解所需的网络并核对接口契约.

    Parameters
    ----------
    checkpoint_dir : str or Path
        含 shape_best.pt 及所需 stiffness_best.pt 的目录.
    provider_metadata : dict
        当前标签提供器的完整元数据.
    route : str
        shape, stiffness 或 both. 刚度路线也加载形函数用于内部恢复.

    Returns
    -------
    tuple[dict, dict]
        CPU float64 网络与包含文件校验值的权重来源记录.
    """
    if route not in ("shape", "stiffness", "both"):
        raise ValueError("route 必须为 shape, stiffness 或 both")
    required = ("shape",) if route == "shape" else ("shape", "stiffness")
    paths = {name: Path(checkpoint_dir) / f"{name}_best.pt" for name in required}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少求解所需权重: {path}")
    networks, sources = {}, {}
    for name, path in paths.items():
        # 独立条目格式与 artifacts.py 的通用 signature 格式不同.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError(f"不支持的独立条目 checkpoint 格式: {path}")
        dataset = payload.get("dataset", {})
        if (payload.get("route") != name
                or dataset.get("complete") is not True
                or dataset.get("provider") != provider_metadata
                or dataset.get("input_quantity") != "normalized_young_modulus"):
            raise ValueError(f"权重的路线、接口空间或材料/离散配置不匹配: {path}")
        architecture = payload.get("architecture", {})
        count = architecture.get("num_networks")
        if count is None:
            raise ValueError(f"权重未登记网络数量: {path}")
        model = build_network(provider_metadata, route=name, num_networks=count)
        output_dim = provider_metadata[f"n_{name}_targets"]
        expected_groups = getattr(model, "output_groups", (tuple(range(output_dim)),))
        groups = tuple(tuple(group) for group in architecture.get("output_groups", ()))
        allowed_classes = {type(model).__name__}
        if name == "shape":
            allowed_classes.add("SplitShapeFunctionNet")
        if (architecture.get("input_dim") != provider_metadata["n_cells"]
                or architecture.get("output_dim") != output_dim
                or tuple(architecture.get("hidden_dims", ())) != HIDDEN_DIMS
                or architecture.get("activations") != [cls.__name__ for cls in ACTIVATIONS]
                or architecture.get("dtype") != "float64"
                or architecture.get("model_class") not in allowed_classes
                or groups != expected_groups):
            raise ValueError(f"权重的网络结构或输出分组不匹配: {path}")
        model.load_state_dict(payload["model_state"], strict=True)
        if hasattr(model, "restore_order"):
            expected_order = torch.argsort(torch.tensor([i for group in groups for i in group]))
            if not torch.equal(model.restore_order, expected_order):
                raise ValueError(f"权重的输出索引顺序不匹配: {path}")
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError(f"权重包含非有限值: {path}")
        model.eval()
        networks[name] = model
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sources[name] = {
            "path": str(path.resolve()), "sha256": digest.hexdigest(),
            "epoch": payload.get("epoch"),
            "validation_loss": payload.get("validation_loss"),
            "architecture": architecture,
        }
    return networks, sources
