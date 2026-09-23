"""独立分量路线的磁盘数据集与补全矩阵监督训练.

Notes
-----
只实现监督训练. 不包含后期一致性损失或结构求解验收.
数据按批次生成, 训练与验证均按小批量读取.
"""

from __future__ import annotations

import json
from math import prod
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .nets import DirectStiffnessNet, SplitOutputNet
from .training import TrainingConfig


HIDDEN_DIMS = (60, 80, 100, 120, 140, 160, 180, 200, 180, 160, 140, 120, 100, 80, 60)
ACTIVATIONS = (
    nn.Tanh, nn.ELU, nn.Tanh, nn.ELU, nn.Tanh,
    nn.ELU, nn.Tanh, nn.ELU, nn.ELU, nn.Tanh,
    nn.ELU, nn.Tanh, nn.ELU, nn.Tanh, nn.ELU,
)
SCHEMA = "independent_entries_v1"


def build_network(provider_metadata, *, route="shape", seed=2026, num_networks=None):
    """根据接口空间元数据构建一条路线的 15 隐藏层模型.

    Parameters
    ----------
    provider_metadata : dict
        标签提供器的元数据, 决定输入与独立输出维度.
    route : str
        单条预测路线, 当前支持 shape 或 stiffness.
    seed : int
        模型初始化种子.
    num_networks : int or None
        该路线的输出拆分网络数量. None 时形函数为 4, 刚度为 1.

    Returns
    -------
    nn.Module
        CPU 上的 float64 模型, 可包含多个输出拆分子网络.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    widths = _check_metadata(provider_metadata)
    if num_networks is not None and (
        isinstance(num_networks, bool) or not isinstance(num_networks, Integral)
        or num_networks <= 0
    ):
        raise ValueError("num_networks 必须为正整数")
    count = num_networks if num_networks is not None else (4 if route == "shape" else 1)
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
    kwargs = dict(
        input_dim=widths["inputs"], output_dim=output_dim,
        hidden_dims=HIDDEN_DIMS, activation=ACTIVATIONS,
    )
    # 保留原单网络刚度模型的权重键格式.
    model = (
        DirectStiffnessNet(**kwargs) if route == "stiffness" and count == 1
        else SplitOutputNet(output_groups=tuple(groups), **kwargs)
    )
    return model.to(dtype=torch.float64)


def _write_json(path, data):
    """写入可读的 JSON 元数据."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_metadata(meta):
    """校验二维/三维两种接口空间的维度关系, 返回数据列宽."""
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


def generate_dataset(
    provider, output_dir, *, n_train=400_000, n_validation=40_000,
    batch_size=64, min_modulus=1e-6, seed=2026,
):
    """分批生成独立随机材料与精确标签, 写入新目录.

    Parameters
    ----------
    provider : callable
        接收 (batch, n_cells) 数组, 返回 shape 与 stiffness 独立分量.
        metadata() 必须描述独立分量编号及有限元配置.
    output_dir : str or Path
        新数据集目录, 已有目录不覆盖.
    n_train, n_validation : int
        互相独立的训练与验证样本数.
    batch_size : int
        每次局部精确计算的样本数.
    min_modulus : float
        归一化杨氏模量的严格正下界, 避免零刚度样本.
    seed : int
        派生独立随机流的主种子.

    Returns
    -------
    Path
        含完成标记的数据集目录.
    """
    if min(n_train, n_validation, batch_size) <= 0:
        raise ValueError("样本数与 batch_size 必须为正")
    if not np.isfinite(min_modulus) or not 0 < min_modulus < 1:
        raise ValueError("min_modulus 必须位于 (0, 1)")
    meta = provider.metadata()
    widths = _check_metadata(meta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA, "complete": False, "provider": meta,
        "seed": seed, "input_quantity": "normalized_young_modulus",
        "sampling": "independent_uniform", "min_modulus": min_modulus,
        "max_modulus": 1.0, "dtype": "float64",
        "counts": {"train": n_train, "validation": n_validation},
    }
    _write_json(output_dir / "manifest.json", manifest)
    streams = np.random.SeedSequence(seed).spawn(2)
    for (split, count), stream in zip(manifest["counts"].items(), streams):
        rng = np.random.default_rng(stream)
        arrays = {
            name: np.lib.format.open_memmap(
                output_dir / f"{split}_{name}.npy", mode="w+",
                dtype=np.float64, shape=(count, width),
            )
            for name, width in widths.items()
        }
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            inputs = rng.uniform(min_modulus, 1.0, size=(stop - start, widths["inputs"]))
            targets = provider(inputs)
            arrays["inputs"][start:stop] = inputs
            for name in ("shape_targets", "stiffness_targets"):
                width = widths[name]
                values = np.asarray(targets[name.removesuffix("_targets")])
                if values.shape != (stop - start, width) or not np.isfinite(values).all():
                    raise ValueError(f"{split} 样本 {start}:{stop} 的 {name} 非法")
                arrays[name][start:stop] = values
            if start == 0 or stop == count or stop % (batch_size * 100) == 0:
                print(f"{split}: {stop}/{count}", flush=True)
        for array in arrays.values():
            array.flush()
        del arrays
    manifest["complete"] = True
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def _epoch(model, codec, inputs, targets, order, batch_size, device, optimizer=None):
    """按样本加权汇总补全矩阵的 MSE."""
    model.train(optimizer is not None)
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x = torch.as_tensor(np.array(inputs[indices]), dtype=torch.float64, device=device)
            y = torch.as_tensor(np.array(targets[indices]), dtype=torch.float64, device=device)
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise FloatingPointError("输入或标签含 NaN/Inf")
            prediction = codec.decode(model(x))
            exact = codec.decode(y)
            loss = (prediction - exact).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("补全矩阵监督损失出现 NaN 或 Inf")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    raise FloatingPointError("梯度缺失或出现 NaN/Inf")
                optimizer.step()
            total += loss.item() * len(indices)
    return total / len(order)


def train_networks(
    dataset_dir, output_dir, *, networks, codecs, provider_metadata,
    route="both", device="cpu", config=None,
):
    """分别训练两条路线, 保存各自验证损失最优的权重.

    Parameters
    ----------
    dataset_dir : str or Path
        generate_dataset 生成的完整数据集.
    output_dir : str or Path
        新训练目录, 禁止覆盖旧结果.
    networks : dict[str, nn.Module]
        由 build_network 分别创建的路线到模型的字典, 训练时原地更新权重.
    codecs : dict
        shape 与 stiffness 对应的可微 decode 对象.
    provider_metadata : dict
        当前解码器对应的提供器元数据, 必须与数据集完全一致.
    route : str
        shape, stiffness 或 both.
    device : str
        显式选择 cpu 或 cuda 设备.
    config : TrainingConfig, optional
        默认 Adam, 500 轮, batch 256, 学习率 1e-3, 提前停止 40 轮.

    Returns
    -------
    dict
        各路线的最佳轮数、验证损失及 checkpoint 路径.

    Notes
    -----
    验证集用于学习率调整和选模, 不等同于独立测试集.
    使用 float64 以减小刚体约束补全与标签存储的精度损失.
    """
    if route not in ("shape", "stiffness", "both"):
        raise ValueError("route 必须为 shape, stiffness 或 both")
    config = config or TrainingConfig(
        epochs=500, batch_size=256, learning_rate=1e-3, seed=2026, patience=40,
    )
    if config.physics_eval_interval or config.select_final_state:
        raise ValueError("独立分量训练只支持验证 MSE 选模")
    if config.learning_rate < 1e-6:
        raise ValueError("初始学习率不能低于调度下限 1e-6")
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    widths = _check_metadata(provider_metadata)
    if (manifest.get("schema") != SCHEMA or manifest.get("complete") is not True
            or manifest.get("provider") != provider_metadata
            or manifest.get("input_quantity") != "normalized_young_modulus"):
        raise ValueError("数据集未完成, 或独立分量编号/材料配置与解码器不一致")
    routes = ("shape", "stiffness") if route == "both" else (route,)
    if set(networks) != set(routes):
        raise ValueError("networks 必须与所选 route 一致")
    data = {}
    for split in ("train", "validation"):
        count = manifest["counts"][split]
        if count <= 0:
            raise ValueError("数据集不能为空")
        for name in ("inputs",) + tuple(f"{route}_targets" for route in routes):
            width = widths[name]
            array = np.load(dataset_dir / f"{split}_{name}.npy", mmap_mode="r")
            if array.shape != (count, width) or array.dtype != np.float64:
                raise ValueError(f"{split}_{name} 的维度或类型不匹配")
            data[split, name] = array
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无可用 CUDA")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "run_config.json", {
        "schema": SCHEMA, "dataset_dir": str(dataset_dir.resolve()),
        "dataset": manifest, "training_config": vars(config),
        "scheduler": {"stale_epochs_to_reduce": 10, "factor": 0.5, "min_lr": 1e-6},
        "route": route, "device": str(device), "dtype": "float64",
        "num_networks": {name: len(model.nets) if isinstance(model, SplitOutputNet) else 1
                         for name, model in networks.items()},
        "loss": "decoded_full_matrix_mse", "consistency_loss": "not_enabled",
    })
    results = {}
    for name in routes:
        rng = np.random.default_rng(config.seed)
        model = networks[name].to(device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=9,  # 第 10 次未改善时降低学习率.
            threshold=0.0, min_lr=1e-6,
        )
        best, stale, history = float("inf"), 0, []
        checkpoint = output_dir / f"{name}_best.pt"
        codec = codecs[name]
        for epoch in range(1, config.epochs + 1):
            training_loss = _epoch(
                model, codec, data["train", "inputs"], data["train", f"{name}_targets"],
                rng.permutation(manifest["counts"]["train"]), config.batch_size, device, optimizer,
            )
            validation_loss = _epoch(
                model, codec, data["validation", "inputs"],
                data["validation", f"{name}_targets"],
                np.arange(manifest["counts"]["validation"]), config.batch_size, device,
            )
            scheduler.step(validation_loss)
            history.append({"epoch": epoch, "training_loss": training_loss,
                            "validation_loss": validation_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"{name} epoch={epoch} train={training_loss:.6e} "
                  f"validation={validation_loss:.6e}", flush=True)
            if validation_loss < best:
                best, stale = validation_loss, 0
                torch.save({
                    "schema": SCHEMA, "route": name,
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch, "validation_loss": best,
                    "dataset": manifest, "training_config": vars(config),
                    "architecture": {
                        "input_dim": widths["inputs"], "output_dim": widths[f"{name}_targets"],
                        "hidden_dims": HIDDEN_DIMS,
                        "activations": [cls.__name__ for cls in ACTIVATIONS],
                        "num_networks": len(model.nets) if isinstance(model, SplitOutputNet) else 1,
                        "output_groups": getattr(model, "output_groups",
                                                 (tuple(range(widths[f"{name}_targets"])),)),
                        "model_class": type(model).__name__,
                        "grouping_origin": "experiment_contiguous_balanced_split",
                        "dtype": "float64",
                    },
                    "loss": "decoded_full_matrix_mse",
                    "consistency_loss": "not_enabled",
                }, checkpoint)
                results[name] = {"best_epoch": epoch, "best_validation_loss": best,
                                 "checkpoint": str(checkpoint)}
            else:
                stale += 1
            _write_json(output_dir / f"{name}_history.json", history)
            if config.patience and stale >= config.patience:
                break
        results[name]["epochs_run"] = len(history)
    _write_json(output_dir / "summary.json", {
        "results": results, "consistency_loss": "not_enabled",
        "independent_test": "not_run",
    })
    return results
