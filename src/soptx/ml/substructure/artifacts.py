"""子结构代理模型 checkpoint 的显式签名与兼容性校验。"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn as nn


CHECKPOINT_SCHEMA_VERSION = 1
TARGET_VERSION_SHAPE_FUNCTION = "shape-function-NRperp-v1"


class ArtifactCompatibilityError(RuntimeError):
    """checkpoint 结构或模型签名与当前算例不兼容。"""


@dataclass(frozen=True)
class ModelSignature:
    """决定模型输入、输出和物理语义的不可省略签名。"""

    n_fine: tuple[int, ...]
    input_dim: int
    output_dim: int
    n_interior_dofs: int
    n_reduced: int
    sampler_version: str
    target_version: str = TARGET_VERSION_SHAPE_FUNCTION

    def __post_init__(self) -> None:
        if not self.n_fine or any(value <= 0 for value in self.n_fine):
            raise ValueError("n_fine 必须由正整数组成。")
        dimensions = (
            self.input_dim,
            self.output_dim,
            self.n_interior_dofs,
            self.n_reduced,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("模型签名中的维数必须为正整数。")
        if self.input_dim != int(torch.tensor(self.n_fine).prod().item()):
            raise ValueError("input_dim 必须等于 n_fine 各维度之积。")
        if self.output_dim != self.n_interior_dofs * self.n_reduced:
            raise ValueError("output_dim 必须等于 n_interior_dofs * n_reduced。")


@dataclass(frozen=True)
class ArchitectureSignature:
    """网络结构的登记形式。

    与 ModelSignature 的分工: 后者是调用方给出的物理签名 (输入输出维数、
    采样与标签版本), 前者是网络自身的结构, 由模型实例读出, 调用方不书写,
    以免登记值与实际构造的网络产生漂移。
    """

    hidden_dims: tuple[int, ...]
    activation: str

    @classmethod
    def from_model(cls, model: nn.Module) -> "ArchitectureSignature":
        """从模型实例读取结构。

        参数:
            model: 已构造的网络, 须带 hidden_dims 与 activation_name
                属性 (MLP 及其子类在 __init__ 中登记)。

        返回:
            该模型的架构签名。

        异常:
            TypeError: 模型未登记架构属性时抛出。
        """
        try:
            hidden_dims = tuple(int(value) for value in model.hidden_dims)
            activation = str(model.activation_name)
        except AttributeError as error:
            raise TypeError(
                f"{type(model).__name__} 未登记 hidden_dims/activation_name, "
                "无法写入架构签名; 请改用 MLP 及其子类。"
            ) from error
        return cls(hidden_dims=hidden_dims, activation=activation)


def _check_architecture(
    recorded: Any,
    model: nn.Module,
    source: Path,
) -> None:
    """比对 checkpoint 登记的架构与当前构造的网络。

    参数:
        recorded: checkpoint 中的架构字典; None 表示该文件早于架构登记。
        model: 由 model_factory 构造的网络。
        source: checkpoint 路径, 仅用于消息。

    异常:
        ArtifactCompatibilityError: 登记的架构与当前网络不一致时抛出。

    说明:
        缺少架构登记时只发警告: 旧 checkpoint 的层数与激活无从校验, 但权重
        形状仍由 load_state_dict 把关, 因此不阻断加载。
    """
    if recorded is None:
        warnings.warn(
            f"checkpoint {source} 未登记网络架构: 该文件早于架构签名的引入, "
            "层数与激活无法校验 (激活不含参数, 配错不会触发形状错误); "
            "请自行确认与训练时一致, 或重新训练以补齐登记。",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    if not isinstance(recorded, dict):
        raise ArtifactCompatibilityError("checkpoint 的 architecture 必须是映射。")
    expected = ArchitectureSignature.from_model(model)
    actual = ArchitectureSignature(
        hidden_dims=tuple(int(value) for value in recorded.get("hidden_dims", ())),
        activation=str(recorded.get("activation", "")),
    )
    if actual != expected:
        raise ArtifactCompatibilityError(
            f"网络架构不匹配: artifact={asdict(actual)}, expected={asdict(expected)}。"
        )


ModelT = TypeVar("ModelT", bound=nn.Module)


def _signature_differences(
    actual: Mapping[str, Any],
    expected: ModelSignature,
) -> dict[str, tuple[Any, Any]]:
    expected_dict = asdict(expected)
    return {
        name: (actual.get(name), expected_value)
        for name, expected_value in expected_dict.items()
        if actual.get(name) != expected_value
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    signature: ModelSignature,
    training_summary: Mapping[str, Any],
) -> None:
    """以带 schema 和物理签名的格式原子保存模型。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "signature": asdict(signature),
        "architecture": asdict(ArchitectureSignature.from_model(model)),
        "model_state_dict": model.state_dict(),
        "training_summary": dict(training_summary),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    model_factory: Callable[[], ModelT],
    expected_signature: ModelSignature,
) -> tuple[ModelT, Mapping[str, Any]]:
    """校验 checkpoint 后加载模型；任何签名差异均直接报错。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ArtifactCompatibilityError(
            "检测到无 schema 的旧裸 state_dict；请显式调用 load_legacy_state_dict。"
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"checkpoint schema_version={payload['schema_version']}，"
            f"当前仅支持 {CHECKPOINT_SCHEMA_VERSION}。"
        )
    actual_signature = payload.get("signature")
    if not isinstance(actual_signature, dict):
        raise ArtifactCompatibilityError("checkpoint 缺少有效的模型 signature。")
    differences = _signature_differences(actual_signature, expected_signature)
    if differences:
        details = ", ".join(
            f"{name}: artifact={actual!r}, expected={expected!r}"
            for name, (actual, expected) in differences.items()
        )
        raise ArtifactCompatibilityError(f"模型 signature 不匹配: {details}")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ArtifactCompatibilityError("checkpoint 缺少 model_state_dict。")
    model = model_factory()
    _check_architecture(payload.get("architecture"), model, source)
    model.load_state_dict(state_dict)
    model.eval()
    summary = payload.get("training_summary", {})
    if not isinstance(summary, dict):
        raise ArtifactCompatibilityError("training_summary 必须是映射。")
    return model, summary


def load_legacy_state_dict(
    path: str | Path,
    model: ModelT,
) -> ModelT:
    """显式加载旧裸 state_dict；调用者自行承担缺少物理签名的风险。"""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or "schema_version" in state_dict:
        raise ArtifactCompatibilityError("给定文件不是旧裸 state_dict。")
    model.load_state_dict(state_dict)
    model.eval()
    return model
