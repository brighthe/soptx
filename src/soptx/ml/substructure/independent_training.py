"""独立分量路线的补全矩阵监督训练.

Notes
-----
只实现监督训练. 不包含后期一致性损失或结构求解验收.
训练与验证均按小批量读取磁盘数据集.
"""

from __future__ import annotations

import json
import hashlib
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


def _stiffness_operators(metadata, device):
    """恢复用于单元能量重构的固定算子, 保持原后端设置.

    Parameters
    ----------
    metadata : dict
        样本中的局部离散配置及独立分量契约.
    device : torch.device
        单元算子使用的训练设备.

    Returns
    -------
    dict
        单元刚度, 单元自由度, 内部自由度及边界迹基.
    """
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider

    previous_backend = bm.get_current_backend().backend_name
    try:
        bm.set_backend("numpy")
        reference = IndependentTargetProvider(
            cell_size=tuple(metadata["cell_size"]), n_fine=tuple(metadata["n_fine"]),
            nu=metadata["poisson_ratio"], trace_kind=metadata["trace"],
            hypothesis=metadata.get("material_hypothesis"),
        )
        if not provider_metadata_matches(reference.metadata(), metadata):
            raise ValueError("刚度损失的参考算子与样本契约不匹配")
        proto = reference.prototype
        boundary = np.zeros((proto.n_total_dofs, metadata["n_trace"]))
        boundary[np.asarray(proto.b_dofs)] = np.asarray(reference.trace.matrix)
        return {
            "element_stiffness": torch.as_tensor(np.array(proto.KE_unit), dtype=torch.float64, device=device),
            "cell_dofs": torch.as_tensor(np.array(proto.cell2dof), dtype=torch.long, device=device),
            "internal_dofs": torch.as_tensor(np.array(proto.i_dofs), dtype=torch.long, device=device),
            "boundary_basis": torch.as_tensor(boundary, dtype=torch.float64, device=device),
        }
    finally:
        bm.set_backend(previous_backend)


def _reconstructed_stiffness(shape, modulus, operators):
    """用真实材料比例和单元能量重构接口刚度, 保留形函数梯度.

    Parameters
    ----------
    shape : torch.Tensor
        内部形函数, 形状为 (batch, n_i, n_trace).
    modulus : torch.Tensor
        用于物理装配的单元模量, 形状为 (batch, n_cells).
    operators : dict
        _stiffness_operators 返回的固定单元算子.

    Returns
    -------
    torch.Tensor
        形状为 (batch, n_trace, n_trace) 的重构刚度.
    """
    basis = operators["boundary_basis"].unsqueeze(0).expand(len(shape), -1, -1)
    basis = basis.index_copy(1, operators["internal_dofs"], shape)
    element_basis = basis[:, operators["cell_dofs"]]
    product = torch.einsum("eij,bejr->beir", operators["element_stiffness"], element_basis)
    return torch.einsum("beiq,beir,be->bqr", element_basis, product, modulus)


def _relative_stiffness_loss(prediction, exact, sample_weights=None):
    """计算每个样本相对 Frobenius 刚度误差的平方并平均.

    Parameters
    ----------
    prediction, exact : torch.Tensor
        相同形状的预测与精确刚度矩阵.

    sample_weights : torch.Tensor or None
        按全训练集均值归一化的正权重, None 表示等权.

    Returns
    -------
    torch.Tensor
        可微标量, 不对低模量样本使用绝对误差权重.
    """
    denominator = exact.square().sum(dim=(-2, -1))
    if not bool(torch.isfinite(exact).all() and (denominator > 0).all()):
        raise FloatingPointError("精确刚度须有限且具有非零范数")
    relative = (prediction - exact).square().sum(dim=(-2, -1)) / denominator
    if sample_weights is not None:
        if (sample_weights.shape != relative.shape
                or not bool(torch.isfinite(sample_weights).all() and (sample_weights > 0).all())):
            raise ValueError("样本权重须为与当前批次一致的有限正向量")
        relative = relative * sample_weights
    return relative.mean()


def _epoch(model, codec, inputs, targets, order, batch_size, device, optimizer=None,
           *, stiffness_targets=None, stiffness_codec=None, operators=None,
           stiffness_loss_weight=0., loss_chunk_size=None, return_components=False,
           stiffness_sample_weights=None):
    """按样本汇总监督损失, 小批量累积后执行一次优化器更新.

    Parameters
    ----------
    model, codec : object
        网络及当前路线的补全器.
    inputs, targets : numpy.ndarray
        内存映射输入和当前路线标签.
    order : numpy.ndarray
        本轮样本索引顺序.
    batch_size : int
        每次优化器更新的样本数.
    device : torch.device
        显式训练设备.
    optimizer : torch.optim.Optimizer or None
        None 时只计算验证损失.
    stiffness_targets, stiffness_codec, operators : object or None
        shape 附加刚度监督所需的精确标签, 解码器及固定单元算子.
    stiffness_loss_weight : float
        相对重构刚度损失的非负权重.
    loss_chunk_size : int or None
        每个计算块的样本数, None 使用 batch_size.
    stiffness_sample_weights : numpy.ndarray or None
        全训练集均值归一化的刚度监督权重, 验证时不传入.
    return_components : bool
        是否返回总损失及分量记录; 默认保持标量返回接口.

    Returns
    -------
    float or dict
        按样本数加权的损失, 可包含 shape_mse 与 relative_stiffness_squared.
    """
    model.train(optimizer is not None)
    totals = dict(total=0., matrix_mse=0., relative_stiffness_squared=0.)
    chunk_size = loss_chunk_size or batch_size
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch_indices), chunk_size):
                indices = batch_indices[offset:offset + chunk_size]
                x = torch.as_tensor(np.array(inputs[indices]), dtype=torch.float64, device=device)
                y = torch.as_tensor(np.array(targets[indices]), dtype=torch.float64, device=device)
                if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                    raise FloatingPointError("输入或标签含 NaN/Inf")
                prediction = codec.decode(model(x))
                exact = codec.decode(y)
                mse = (prediction - exact).square().mean()
                stiffness_loss = mse.new_zeros(())
                if stiffness_loss_weight:
                    if not bool((x > 0).all()):
                        raise ValueError("刚度监督要求模量严格为正")
                    labels = torch.as_tensor(np.array(stiffness_targets[indices]), dtype=torch.float64, device=device)
                    scale = x.amax(dim=-1, keepdim=True)
                    predicted_k = _reconstructed_stiffness(prediction, x / scale, operators)
                    exact_k = stiffness_codec.decode(labels) / scale.unsqueeze(-1)
                    weights = (None if stiffness_sample_weights is None else torch.as_tensor(
                        np.array(stiffness_sample_weights[indices]), dtype=torch.float64, device=device))
                    stiffness_loss = _relative_stiffness_loss(predicted_k, exact_k, weights)
                loss = mse + stiffness_loss_weight * stiffness_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("监督损失出现 NaN 或 Inf")
                if optimizer is not None:
                    (loss * (len(indices) / len(batch_indices))).backward()
                for key, value in (("total", loss), ("matrix_mse", mse),
                                   ("relative_stiffness_squared", stiffness_loss)):
                    totals[key] += value.item() * len(indices)
            if optimizer is not None:
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    raise FloatingPointError("梯度缺失或出现 NaN/Inf")
                optimizer.step()
    metrics = {key: value / len(order) for key, value in totals.items()}
    return metrics if return_components else metrics["total"]


def _load_training_stiffness_weights(path, dataset_dir, count):
    """核验训练权重的样本契约与文件摘要, 按全训练集均值归一化.

    Parameters
    ----------
    path, dataset_dir : Path
        权重文件及其对应的完整样本目录.
    count : int
        训练样本数.

    Returns
    -------
    tuple
        归一化权重和可追溯的来源记录.
    """
    path = Path(path).expanduser().resolve()
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256((Path(dataset_dir) / "manifest.json").read_bytes()).hexdigest()
    if (provenance.get("schema") != "training_stiffness_weights_v1"
            or provenance.get("selection_split") != "train"
            or provenance.get("counts") != count
            or provenance.get("dataset_manifest_sha256") != manifest_digest
            or provenance.get("weights_sha256") != digest
            or Path(provenance.get("dataset_dir", "")).resolve() != Path(dataset_dir).resolve()):
        raise ValueError("权重来源与训练集不一致, 或文件摘要不匹配")
    weights = np.load(path, allow_pickle=False)
    if (weights.shape != (count,) or weights.dtype != np.float64
            or not np.isfinite(weights).all() or not (weights > 0).all()):
        raise ValueError("训练刚度权重须为与训练集等长的 float64 有限正向量")
    mean = float(weights.mean())
    return weights / mean, {"path": str(path), "sha256": digest,
                            "normalization": "global_training_mean", "mean": mean,
                            "provenance": provenance, "validation_weighting": "none"}


def train_network(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    route: Literal["shape", "stiffness"],
    network: IndependentOutputNet | SplitOutputNet,
    provider: IndependentPredictionDecoder,
    device: str | torch.device = "cpu",
    config: TrainingConfig | None = None,
    initialization: dict[str, Any] | None = None,
    stiffness_loss_weight: float = 0.,
    loss_chunk_size: int | None = None,
    train_stiffness_weights: str | Path | None = None,
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

    initialization : dict or None
        可选的初始化权重来源记录. 仅记录来源, 本函数始终新建优化器和调度器.

    stiffness_loss_weight : float
        shape 相对重构刚度监督权重, 默认 0 保持原 MSE.
    loss_chunk_size : int or None
        梯度累积的计算块大小, 不改变优化器 batch_size.

    train_stiffness_weights : str or Path or None
        可追溯的训练集刚度权重文件, 不改变验证损失或选模规则.

    Returns
    -------
    dict
        最佳轮数 best_epoch、最佳验证损失 best_validation_loss、
        checkpoint 路径及实际训练轮数 epochs_run.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    if not np.isfinite(stiffness_loss_weight) or stiffness_loss_weight < 0:
        raise ValueError("刚度监督权重须有限且非负")
    if stiffness_loss_weight and route != "shape":
        raise ValueError("重构刚度监督仅支持 shape 路线")
    if loss_chunk_size is not None and (isinstance(loss_chunk_size, bool)
            or not isinstance(loss_chunk_size, int) or loss_chunk_size <= 0):
        raise ValueError("loss_chunk_size 必须为正整数或 None")
    if train_stiffness_weights is not None and not stiffness_loss_weight:
        raise ValueError("训练样本权重要求启用 shape 刚度监督")
    input_normalization = getattr(network, "input_normalization", "none")
    if input_normalization not in ("none", "per_sample_max"):
        raise ValueError("未知的材料输入归一化方式")
    if route != "shape" and input_normalization != "none":
        raise ValueError("最大模量归一化仅支持 shape 路线")
    codec = provider.codecs[route]
    provider_metadata = provider.metadata()
    config = config or TrainingConfig(
        epochs=500, batch_size=256, optimizer_params={"lr": 1e-3}, seed=2026, patience=40,
    )
    if config.physics_eval_interval or config.select_final_state:
        raise ValueError("独立分量训练只支持验证监督损失选模")
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
        names = ("inputs", f"{route}_targets")
        if stiffness_loss_weight:
            names += ("stiffness_targets",)
        for name in names:
            width = widths[name]
            array = np.load(dataset_dir / f"{split}_{name}.npy", mmap_mode="r")
            if array.shape != (count, width) or array.dtype != np.float64:
                raise ValueError(f"{split}_{name} 的维度或类型不匹配")
            data[split, name] = array
    sample_weights, weight_source = (None, None)
    if train_stiffness_weights is not None:
        sample_weights, weight_source = _load_training_stiffness_weights(
            train_stiffness_weights, dataset_dir, manifest["counts"]["train"])
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无可用 CUDA")
    operators = _stiffness_operators(provider_metadata, device) if stiffness_loss_weight else None
    loss_name = ("shape_mse_plus_relative_reconstructed_stiffness"
                 if stiffness_loss_weight else "decoded_full_matrix_mse")
    loss_settings = {"stiffness_loss_weight": stiffness_loss_weight,
                     "loss_chunk_size": loss_chunk_size,
                     "training_stiffness_weights": weight_source,
                     "stiffness_teacher": "exact_dataset_labels" if stiffness_loss_weight else None}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "run_config.json", {
        "schema": SCHEMA, "dataset_dir": str(dataset_dir.resolve()),
        "initialization": initialization,
        "input_normalization": input_normalization,
        "dataset": manifest, "training_config": vars(config),
        "scheduler": {"stale_epochs_to_reduce": 10, "factor": 0.5, "min_lr": 1e-6},
        "route": route, "device": str(device), "dtype": "float64",
        "num_networks": {
            route: len(network.nets) if isinstance(network, SplitOutputNet) else 1,
        },
        "loss": loss_name, "loss_settings": loss_settings, "consistency_loss": "not_enabled",
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
        training_metrics = _epoch(
            model, codec, data["train", "inputs"], data["train", f"{route}_targets"],
            rng.permutation(manifest["counts"]["train"]), config.batch_size, device, optimizer,
            stiffness_targets=data.get(("train", "stiffness_targets")),
            stiffness_codec=provider.codecs["stiffness"], operators=operators,
            stiffness_loss_weight=stiffness_loss_weight, loss_chunk_size=loss_chunk_size,
            return_components=True, stiffness_sample_weights=sample_weights,
        )
        validation_metrics = _epoch(
            model, codec, data["validation", "inputs"],
            data["validation", f"{route}_targets"],
            np.arange(manifest["counts"]["validation"]), config.batch_size, device,
            stiffness_targets=data.get(("validation", "stiffness_targets")),
            stiffness_codec=provider.codecs["stiffness"], operators=operators,
            stiffness_loss_weight=stiffness_loss_weight, loss_chunk_size=loss_chunk_size,
            return_components=True,
        )
        training_loss, validation_loss = training_metrics["total"], validation_metrics["total"]
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "training_components": training_metrics,
                        "validation_components": validation_metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"{route} epoch={epoch} train={training_loss:.6e} "
              f"validation={validation_loss:.6e}", flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save({
                "schema": SCHEMA, "route": route,
                "initialization": initialization,
                "input_normalization": input_normalization,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch, "validation_loss": best,
                "validation_components": validation_metrics,
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
                "loss": loss_name, "loss_settings": loss_settings,
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
        "input_normalization": input_normalization,
        "loss": loss_name, "loss_settings": loss_settings,
        "best_validation_components": history[best_epoch - 1]["validation_components"],
        "independent_test": "not_run",
    })
    return result
