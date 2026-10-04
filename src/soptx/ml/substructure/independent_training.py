"""独立分量路线的补全矩阵监督训练.

Notes
-----
只实现监督训练. 不包含后期一致性损失或结构求解验收.
训练与验证均按小批量读取磁盘数据集.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from .independent_contract import (
    SCHEMA, activation_names, metadata_widths, provider_metadata_matches, write_json,
)
from .nets import IndependentOutputNet, SplitOutputNet
from .training import TrainingConfig, build_optimizer

if TYPE_CHECKING:
    from soptx.fem.substructure.independent_targets import IndependentPredictionDecoder


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


def train_network(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    route: Literal["shape", "stiffness"],
    network: IndependentOutputNet | SplitOutputNet,
    provider: IndependentPredictionDecoder,
    device: str | torch.device = "cpu",
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """训练单条路线的网络, 保存验证损失最优的权重.

    Parameters
    ----------
    dataset_dir : str or Path
        准备的完整训练数据.
    output_dir : str or Path
        新训练目录, 禁止覆盖旧结果.
    route : {"shape", "stiffness"}
        学习目标.
    network : IndependentOutputNet or SplitOutputNet
        由 build_network 按同一 route 创建的模型, 训练时原地更新权重.
    provider : IndependentPredictionDecoder
        生成或读取数据集时使用的接口空间对象.
    device : str or torch.device
        显式选择 cpu 或 cuda 设备.
    config : TrainingConfig, optional
        支持 Adam、AdamW 或 SGD; 默认 Adam, 500 轮, batch 256, 学习率 1e-3, 提前停止 40 轮.

    Returns
    -------
    dict
        最佳轮数 best_epoch、最佳验证损失 best_validation_loss、
        checkpoint 路径及实际训练轮数 epochs_run.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    codec = provider.codecs[route]
    provider_metadata = provider.metadata()
    config = config or TrainingConfig(
        epochs=500, batch_size=256, optimizer_params={"lr": 1e-3}, seed=2026, patience=40,
    )
    if config.physics_eval_interval or config.select_final_state:
        raise ValueError("独立分量训练只支持验证 MSE 选模")
    if config.optimizer_params["lr"] < 1e-6:
        raise ValueError("初始学习率不能低于调度下限 1e-6")
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    widths = metadata_widths(provider_metadata)
    if (manifest.get("schema") != SCHEMA or manifest.get("complete") is not True
            or not provider_metadata_matches(manifest.get("provider"), provider_metadata)
            or manifest.get("input_quantity") != "normalized_young_modulus"):
        raise ValueError("数据集未完成, 或独立分量编号/材料配置与解码器不一致")
    data = {}
    for split in ("train", "validation"):
        count = manifest["counts"][split]
        if count <= 0:
            raise ValueError("数据集不能为空")
        for name in ("inputs", f"{route}_targets"):
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
    write_json(output_dir / "run_config.json", {
        "schema": SCHEMA, "dataset_dir": str(dataset_dir.resolve()),
        "dataset": manifest, "training_config": vars(config),
        "scheduler": {"stale_epochs_to_reduce": 10, "factor": 0.5, "min_lr": 1e-6},
        "route": route, "device": str(device), "dtype": "float64",
        "num_networks": {
            route: len(network.nets) if isinstance(network, SplitOutputNet) else 1,
        },
        "loss": "decoded_full_matrix_mse", "consistency_loss": "not_enabled",
    })
    rng = np.random.default_rng(config.seed)
    model = network.to(device=device, dtype=torch.float64)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=9,  # 第 10 次未改善时降低学习率.
        threshold=0.0, min_lr=1e-6,
    )
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    checkpoint = output_dir / f"{route}_best.pt"
    for epoch in range(1, config.epochs + 1):
        training_loss = _epoch(
            model, codec, data["train", "inputs"], data["train", f"{route}_targets"],
            rng.permutation(manifest["counts"]["train"]), config.batch_size, device, optimizer,
        )
        validation_loss = _epoch(
            model, codec, data["validation", "inputs"],
            data["validation", f"{route}_targets"],
            np.arange(manifest["counts"]["validation"]), config.batch_size, device,
        )
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"{route} epoch={epoch} train={training_loss:.6e} "
              f"validation={validation_loss:.6e}", flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save({
                "schema": SCHEMA, "route": route,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch, "validation_loss": best,
                "dataset": manifest, "training_config": vars(config),
                "architecture": {
                    "input_dim": widths["inputs"], "output_dim": widths[f"{route}_targets"],
                    "hidden_dims": tuple(model.hidden_dims),
                    "activations": activation_names(model),
                    "num_networks": len(model.nets) if isinstance(model, SplitOutputNet) else 1,
                    "output_groups": getattr(model, "output_groups",
                                             (tuple(range(widths[f"{route}_targets"])),)),
                    "model_class": type(model).__name__,
                    "grouping_origin": "experiment_contiguous_balanced_split",
                    "dtype": "float64",
                },
                "loss": "decoded_full_matrix_mse",
                "consistency_loss": "not_enabled",
            }, checkpoint)
            best_epoch = epoch
        else:
            stale += 1
        write_json(output_dir / f"{route}_history.json", history)
        if config.patience and stale >= config.patience:
            break
    # TrainingConfig 保证至少一轮, 且首轮有限损失必然低于 inf, 故 best_epoch >= 1.
    result = {"best_epoch": best_epoch, "best_validation_loss": best,
              "checkpoint": str(checkpoint), "epochs_run": len(history)}
    write_json(output_dir / "summary.json", {
        "results": {route: result}, "consistency_loss": "not_enabled",
        "independent_test": "not_run",
    })
    return result
