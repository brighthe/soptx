# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与校验 (PIML 子结构分析).

本模块只负责把注册表读成结构化的 ``AnalysisCase`` 对象并做静态校验, 不触发任何
计算. 运行调度见 ``run.py``, 产物读取见 ``collect.py``.

注册表按研究任务 (``task``) 组织: 任务决定调用哪个脚本、哪些配置字段合法, 以及
命令行如何组装. 两个 PIML 验证脚本的同义参数命名不一致 (``--n-train`` 与
``--train-samples``, ``--n-eval`` 与 ``--val-samples``), 该差异在本模块的
``AnalysisCase.command`` 中吸收, 注册表内保持统一字段名.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FIGURE_DATA_DIR = EXPERIMENT_DIR / "figure_data"

PANELS = ("a", "b", "c")

#: 研究任务到验证脚本的映射, 路径自仓库根起算.
_SCRIPTS = {
    "shape_function_route":
        "examples/piml_substructure_elasticity/verify_shape_function_route.py",
    "reduced_stiffness_route":
        "examples/piml_substructure_elasticity/verify_stiffness_route.py",
}

#: 计算归其他实验目录所有的任务. 本目录不产出、不重跑, 只把已有产物作为图件
#: 输入引用; 这类工况必须声明 source, 且不组装命令行.
_EXTERNAL_TASKS = ("exact_condensation_equivalence",)

_TASKS = tuple(_SCRIPTS) + _EXTERNAL_TASKS

_TRACES = ("full_trace", "linear_corner")
_BACKENDS = ("numpy", "pytorch")

#: 降阶刚度网络的宽度固定为 ``case_setup.py`` 的模块常量
#: ``_REDUCED_STIFFNESS_HIDDEN_DIMS``, 故无命令行开关.
#: 登记为常量是为了让两条路线的网络容量差异在注册表层面可见.
REDUCED_STIFFNESS_HIDDEN_DIM = 128

#: 训练类字段, 精确缩聚工况禁止出现.
_TRAINING_KEYS = (
    "trace_basis", "n_train", "n_eval", "epochs", "lr", "hidden_dim",
    "seed", "backend",
)


class ConfigError(RuntimeError):
    """``cases.toml`` 结构或取值不合法."""


@dataclass(frozen=True)
class AnalysisCase:
    """统一入口使用的单个分析工况.

    Attributes
    ----------
    id : str
        工况标识, 同时用作产物摘要与命令行选择键.
    task : str
        研究任务, 取值见 ``_SCRIPTS``.
    summary : str
        单行说明, 出现在运行日志中.
    problem : str
        算例类名.
    artifact : str
        产物文件名, 相对本目录 ``outputs/``.
    panels : tuple of str
        该工况供给的图件分格; 空元组表示不进图.
    source : str or None
        产出该工况的实验目录, 自仓库根起算; 非 None 表示本目录不执行该工况.
    dim : int or None
        空间维数, 只用于精确缩聚工况.
    trace_basis : str or None
        接口迹空间, 取值见 ``_TRACES``.
    n_train, n_eval, epochs, hidden_dim, seed : int or None
        训练与留出评估配置.
    lr : float or None
        Adam 学习率.
    backend : str or None
        计算后端, 只有降阶刚度工况支持.
    """

    id: str
    task: str
    summary: str
    problem: str
    artifact: str | None = None
    panels: tuple[str, ...] = field(default_factory=tuple)
    source: str | None = None
    dim: int | None = None
    trace_basis: str | None = None
    n_train: int | None = None
    n_eval: int | None = None
    epochs: int | None = None
    lr: float | None = None
    hidden_dim: int | None = None
    seed: int | None = None
    backend: str | None = None

    @property
    def runnable(self) -> bool:
        """本目录能否执行该工况; 外部来源工况为 False."""
        return self.task in _SCRIPTS

    @property
    def script(self) -> str | None:
        """自仓库根起算的脚本相对路径; 外部来源工况为 None."""
        return _SCRIPTS.get(self.task)

    @property
    def script_path(self) -> Path | None:
        script = self.script
        return REPOSITORY_ROOT / script if script is not None else None

    @property
    def artifact_name(self) -> str:
        """推导或获取产物标准文件名."""
        if self.artifact:
            return self.artifact
        if self.task == "shape_function_route":
            return f"eq17_second_order_{self.trace_basis}.json"
        if self.task == "reduced_stiffness_route":
            return "piml_exact_comparison.json"
        if self.task == "exact_condensation_equivalence":
            return f"lagrange_comparison_{self.dim}d.json"
        return f"{self.id}.json"

    def find_latest_artifact(self, output_root: Path = OUTPUT_DIR) -> Path | None:
        """在 output_root/<case.id>/<timestamp>/ 下寻找最新产物，不存在则回退至根输出目录."""
        case_dir = output_root / self.id
        if case_dir.is_dir():
            subdirs = sorted([d for d in case_dir.iterdir() if d.is_dir()], reverse=True)
            for d in subdirs:
                target = d / self.artifact_name
                if target.is_file():
                    return target
        flat_target = output_root / self.artifact_name
        if flat_target.is_file():
            return flat_target
        return None

    @property
    def artifact_path(self) -> Path:
        latest = self.find_latest_artifact()
        if latest is not None:
            return latest
        return OUTPUT_DIR / self.artifact_name

    @property
    def effective_hidden_dim(self) -> int | None:
        """实际生效的网络宽度, 含不可配置的路线常量."""
        if self.task == "reduced_stiffness_route":
            return REDUCED_STIFFNESS_HIDDEN_DIM
        return self.hidden_dim

    def command(self) -> list[str]:
        """按任务组装子进程命令行.

        Returns
        -------
        list of str
            可直接交给 ``subprocess.run`` 的 argv.

        Raises
        ------
        ConfigError
            该工况声明了 ``source``, 计算不归本目录, 无命令行可组装.
        """
        import sys

        if not self.runnable:
            raise ConfigError(
                f"{self.id}: 计算归 {self.source}, 本目录只引用其产物, 不执行"
            )

        argv = [sys.executable, str(self.script_path)]
        if self.task == "shape_function_route":
            argv += [
                "--trace-basis", str(self.trace_basis),
                "--n-train", str(self.n_train),
                "--n-eval", str(self.n_eval),
                "--epochs", str(self.epochs),
                "--lr", repr(self.lr),
                "--hidden-dim", str(self.hidden_dim),
                "--seed", str(self.seed),
            ]
        elif self.task == "reduced_stiffness_route":
            argv += [
                "--trace-basis", str(self.trace_basis),
                "--train-samples", str(self.n_train),
                "--val-samples", str(self.n_eval),
                "--epochs", str(self.epochs),
                "--lr", repr(self.lr),
                "--seed", str(self.seed),
                "--backend", str(self.backend),
            ]
        else:
            raise ConfigError(f"{self.id}: 未知 task {self.task!r}")
        argv += ["--output-dir", str(OUTPUT_DIR)]
        return argv


def _require(mapping: dict[str, Any], key: str, case_id: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"case {case_id!r} 缺少必填键 {key!r}")
    return mapping[key]


def _positive_int(raw: dict[str, Any], name: str, case_id: str) -> int:
    value = _require(raw, name, case_id)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{case_id}: {name} 必须为正整数")
    return value


def _positive_float(raw: dict[str, Any], name: str, case_id: str) -> float:
    value = _require(raw, name, case_id)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{case_id}: {name} 必须为有限正数")
    return float(value)


def _reject(raw: dict[str, Any], names: tuple[str, ...], case_id: str,
            reason: str) -> None:
    present = [name for name in names if name in raw]
    if present:
        raise ConfigError(
            f"{case_id}: {reason}, 不得声明 {', '.join(sorted(present))}"
        )


def _panels(raw: dict[str, Any], case_id: str) -> tuple[str, ...]:
    value = raw.get("panels", [])
    if not isinstance(value, list):
        raise ConfigError(f"{case_id}: panels 必须为字符串列表")
    for item in value:
        if item not in PANELS:
            raise ConfigError(f"{case_id}: panels 只能取 {PANELS}, 实得 {item!r}")
    if len(value) != len(set(value)):
        raise ConfigError(f"{case_id}: panels 不得重复")
    return tuple(value)


def _parse_case(raw: dict[str, Any]) -> AnalysisCase:
    """把一条 ``[[cases]]`` 解析为工况, 按任务分支校验配置字段."""
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ConfigError("每个工况必须提供非空 id")
    if ".." in case_id or any(char in case_id for char in ("/", "\\")):
        raise ConfigError(f"{case_id}: id 不得包含路径分隔符或相对路径")

    task = _require(raw, "task", case_id)
    if task not in _TASKS:
        raise ConfigError(f"{case_id}: 未知 task {task!r}, 可选 {_TASKS}")
    summary = _require(raw, "summary", case_id)
    if not isinstance(summary, str) or not summary.strip():
        raise ConfigError(f"{case_id}: summary 不能为空")
    artifact = raw.get("artifact")
    if artifact is not None:
        if not isinstance(artifact, str) or "/" in artifact or "\\" in artifact:
            raise ConfigError(f"{case_id}: artifact 必须为 outputs/ 下的文件名")

    common = dict(
        id=case_id,
        task=task,
        summary=summary,
        problem=_require(raw, "problem", case_id),
        artifact=artifact,
        panels=_panels(raw, case_id),
    )

    if task in _EXTERNAL_TASKS:
        _reject(raw, _TRAINING_KEYS, case_id, "外部来源工况不在本目录训练网络")
        source = _require(raw, "source", case_id)
        if not isinstance(source, str) or not source.strip():
            raise ConfigError(
                f"{case_id}: task {task!r} 的计算不归本目录, 必须声明 source"
            )
        if not (REPOSITORY_ROOT / source).is_dir():
            raise ConfigError(f"{case_id}: source 目录不存在 -> {source}")
        dim = _positive_int(raw, "dim", case_id)
        if dim not in (2, 3):
            raise ConfigError(f"{case_id}: dim 必须为 2 或 3")
        return AnalysisCase(**common, source=source, dim=dim)

    _reject(raw, ("source",), case_id, f"task {task!r} 由本目录执行")
    trace_basis = _require(raw, "trace_basis", case_id)
    if trace_basis not in _TRACES:
        raise ConfigError(
            f"{case_id}: trace_basis 只能取 {_TRACES}, 实得 {trace_basis!r}"
        )
    training = dict(
        trace_basis=trace_basis,
        n_train=_positive_int(raw, "n_train", case_id),
        n_eval=_positive_int(raw, "n_eval", case_id),
        epochs=_positive_int(raw, "epochs", case_id),
        lr=_positive_float(raw, "lr", case_id),
        seed=_positive_int(raw, "seed", case_id),
    )

    if task == "shape_function_route":
        _reject(raw, ("backend",), case_id,
                "形函数脚本固定 numpy 后端")
        return AnalysisCase(
            **common,
            **training,
            hidden_dim=_positive_int(raw, "hidden_dim", case_id),
        )

    # reduced_stiffness_route
    if trace_basis != "full_trace":
        raise ConfigError(
            f"{case_id}: verify_stiffness_route.py 的 --trace-basis 只接受 "
            "full_trace, linear_corner 尚无可执行实现"
        )
    _reject(raw, ("hidden_dim",), case_id,
            f"降阶刚度网络宽度固定为 {REDUCED_STIFFNESS_HIDDEN_DIM}, 无命令行开关")
    backend = _require(raw, "backend", case_id)
    if backend not in _BACKENDS:
        raise ConfigError(
            f"{case_id}: backend 只能取 {_BACKENDS}, 实得 {backend!r}"
        )
    return AnalysisCase(**common, **training, backend=backend)


def load_cases(path: Path | None = None) -> tuple[dict[str, Any],
                                                  tuple[AnalysisCase, ...]]:
    """读取全部注册工况并检查 id 唯一性与脚本存在性.

    Returns
    -------
    tuple
        ``[figure]`` 段与工况元组.
    """
    path = path or CASES_FILE
    payload = tomllib.loads(path.read_text(encoding="utf-8"))

    raw_cases = payload.get("cases", [])
    if not raw_cases:
        raise ConfigError(f"{path}: 未注册任何 [[cases]]")

    cases: list[AnalysisCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        case = _parse_case(raw)
        if case.id in seen:
            raise ConfigError(f"case id 重复: {case.id!r}")
        seen.add(case.id)
        if case.runnable and not case.script_path.is_file():
            raise ConfigError(f"{case.id}: script 不存在 -> {case.script_path}")
        cases.append(case)

    return payload.get("figure", {}), tuple(cases)


def select(cases: tuple[AnalysisCase, ...], *, case_id: str | None = None,
           task: str | None = None, panel: str | None = None,
           ) -> tuple[AnalysisCase, ...]:
    """按 id、任务或图件分格筛选工况."""
    selected = cases
    if case_id is not None:
        selected = tuple(c for c in selected if c.id == case_id)
    if task is not None:
        selected = tuple(c for c in selected if c.task == task)
    if panel is not None:
        selected = tuple(c for c in selected if panel in c.panels)
    if not selected:
        available = ", ".join(c.id for c in cases)
        raise ConfigError(
            f"没有匹配的 case (case_id={case_id!r}, task={task!r}, "
            f"panel={panel!r}); 可用 id: {available}"
        )
    return selected


def tasks() -> tuple[str, ...]:
    """已登记的研究任务名."""
    return _TASKS


load = load_cases

