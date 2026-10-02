"""与有限元实现解耦的子结构代理模型训练循环。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainingConfig:
    """监督回归训练配置.

    Parameters
    ----------
    epochs, batch_size : int
        最大训练轮数与批量大小.
    optimizer : str
        adam、adamw 或 sgd, 默认 adam.
    optimizer_params : dict
        优化器参数. 共同支持 lr、weight_decay; sgd 额外支持 momentum.
        缺省值为 lr=1e-3、weight_decay=0.0, SGD 的 momentum=0.0.
    seed : int
        训练随机种子.
    patience : int
        连续未改善的提前停止次数, 0 表示不启用.
    physics_eval_interval : int
        物理指标评估间隔, 0 表示不启用.
    select_final_state : bool
        是否保留最后一轮参数, 默认按验证指标选模.
    """

    epochs: int
    batch_size: int
    optimizer: str = "adam"
    optimizer_params: dict[str, float] = field(default_factory=dict)
    seed: int = 2026
    patience: int = 0
    physics_eval_interval: int = 0
    select_final_state: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs 与 batch_size 必须为正整数.")
        if self.patience < 0 or self.physics_eval_interval < 0:
            raise ValueError("patience 与 physics_eval_interval 不能为负数.")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError("optimizer 必须为 adam, adamw 或 sgd.")
        if not isinstance(self.optimizer_params, Mapping):
            raise ValueError("optimizer_params 必须为参数映射.")
        defaults = {"lr": 1e-3, "weight_decay": 0.0}
        if self.optimizer == "sgd":
            defaults["momentum"] = 0.0
        if any(key not in defaults for key in self.optimizer_params):
            raise ValueError(f"{self.optimizer} 的 optimizer_params 仅支持 {tuple(defaults)}.")
        params = {**defaults, **self.optimizer_params}
        for key, value in params.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value < 0 or (key == "lr" and value == 0)):
                raise ValueError(f"optimizer_params[{key!r}] 必须为有限{'正' if key == 'lr' else '非负'}数.")
        # 复制输入映射, 避免调用者修改原字典影响已登记配置.
        object.__setattr__(self, "optimizer_params", params)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """按统一配置创建优化器.

    Parameters
    ----------
    model : nn.Module
        待训练的网络.
    config : TrainingConfig
        优化器类型、学习率、权重衰减和 SGD 动量配置.

    Returns
    -------
    torch.optim.Optimizer
        绑定网络参数的 Adam、AdamW 或 SGD 优化器.
    """
    optimizer_type = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }[config.optimizer]
    return optimizer_type(model.parameters(), **config.optimizer_params)


@dataclass(frozen=True)
class TrainingResult:
    """训练完成后的可序列化摘要。"""

    epochs_run: int
    best_epoch: int
    best_validation_loss: float
    final_training_loss: float
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    best_selection_score: Optional[float] = None
    physics_history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation", MappingProxyType(dict(self.evaluation)))
        object.__setattr__(
            self,
            "physics_history",
            tuple(MappingProxyType(dict(item)) for item in self.physics_history),
        )


def _as_float_tensor(values: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        raise ValueError(f"{name} 必须是非空二维张量。")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} 含 NaN 或 Inf。")
    return tensor


def train_surrogate(
    model: nn.Module,
    x_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    x_validation: np.ndarray | torch.Tensor,
    y_validation: np.ndarray | torch.Tensor,
    config: TrainingConfig,
    *,
    evaluator: Optional[Callable[[nn.Module], Mapping[str, Any]]] = None,
) -> TrainingResult:
    """训练给定代理网络，并恢复 MSE 或 physics 最优的参数快照。

    本函数只处理已经构造好的 ``(X, Y)``，不导入 FEM，也不负责生成物理标签。
    当 ``physics_eval_interval > 0`` 时，evaluator 必须返回有限的
    selection_score；模型只在固定 physics 检查点参与选模，patience
    也按连续未改善的 physics 检查次数计。否则保持按验证 MSE 选模的旧行为。
    """
    x_train_tensor = _as_float_tensor(x_train, "x_train")
    y_train_tensor = _as_float_tensor(y_train, "y_train")
    x_val_tensor = _as_float_tensor(x_validation, "x_validation")
    y_val_tensor = _as_float_tensor(y_validation, "y_validation")
    if x_train_tensor.shape[0] != y_train_tensor.shape[0]:
        raise ValueError("训练输入与标签的样本数不一致。")
    if x_val_tensor.shape[0] != y_val_tensor.shape[0]:
        raise ValueError("验证输入与标签的样本数不一致。")
    physics_selection = config.physics_eval_interval > 0
    if physics_selection and evaluator is None:
        raise ValueError("physics 选模要求提供 evaluator。")
    if config.select_final_state and physics_selection:
        raise ValueError("fixed-epoch refit 不能同时启用 physics 选模。")
    if config.select_final_state and config.patience:
        raise ValueError("fixed-epoch refit 不能启用 patience。")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=min(config.batch_size, len(x_train_tensor)),
        shuffle=True,
        generator=generator,
    )
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, config)
    best_state = deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_selection_score: Optional[float] = None
    best_evaluation: dict[str, Any] = {}
    best_epoch = 0
    stale_epochs = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []
    physics_history: list[dict[str, Any]] = []

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            weighted_loss = 0.0
            seen = 0
            for inputs, targets in loader:
                optimizer.zero_grad()
                prediction = model(inputs)
                loss = criterion(prediction, targets)
                loss.backward()
                optimizer.step()
                weighted_loss += float(loss.item()) * len(inputs)
                seen += len(inputs)
            training_loss = weighted_loss / seen

            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    criterion(model(x_val_tensor), y_val_tensor).item()
                )
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            if physics_selection:
                should_evaluate = (
                    epoch == 1
                    or epoch % config.physics_eval_interval == 0
                    or epoch == config.epochs
                )
                if should_evaluate:
                    metrics = dict(evaluator(model))
                    if "selection_score" not in metrics:
                        raise ValueError("physics evaluator 必须返回 selection_score。")
                    score = float(metrics["selection_score"])
                    if not np.isfinite(score):
                        raise ValueError("physics evaluator 返回了非有限 selection_score。")
                    record = {"epoch": epoch, "selection_score": score, **metrics}
                    physics_history.append(record)
                    if best_selection_score is None or score < best_selection_score:
                        best_selection_score = score
                        best_validation_loss = validation_loss
                        best_epoch = epoch
                        best_state = deepcopy(model.state_dict())
                        best_evaluation = metrics
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    if config.patience and stale_epochs >= config.patience:
                        break
            else:
                if config.select_final_state:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                    stale_epochs = 0
                elif validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if config.patience and stale_epochs >= config.patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    evaluation = best_evaluation if physics_selection else (
        {} if evaluator is None else dict(evaluator(model))
    )
    return TrainingResult(
        epochs_run=len(training_losses),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        final_training_loss=training_losses[-1],
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        evaluation=evaluation,
        best_selection_score=best_selection_score,
        physics_history=tuple(physics_history),
    )
