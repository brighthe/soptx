"""子结构 PIML GPU 实验的静态工况注册。"""

from __future__ import annotations

import tomllib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


class ConfigError(RuntimeError):
    """工况注册表不满足实验契约。"""


@dataclass(frozen=True)
class TrainingCase:
    """一个 CPU 或 CUDA 训练工况。"""

    id: str
    pair_id: str
    device: str
    summary: str
    dim: int
    sub_size: tuple[float, ...]
    n_fine: tuple[int, ...]
    samples: int
    validation_fraction: float
    design_min: float
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    seed: int
    cpu_threads: int
    emax: float
    nu: float
    simp_penalty: float
    baseline: str = "exploratory_linear_corner"
    n_eval: int = 200

    def comparison_contract(self) -> dict[str, Any]:
        """返回配对两端必须一致的计算契约。"""
        data = dict(vars(self))
        for name in ("id", "device", "summary"):
            data.pop(name)
        data.update(stage="training", trace_basis="full_trace" if self.baseline == "example_full_trace" else "linear_corner", dtype="float32")
        return data

    def with_overrides(
        self,
        samples: int | None,
        epochs: int | None,
        batch_size: int | None,
    ) -> "TrainingCase":
        """应用同一组命令行工作量覆盖。"""
        result = replace(
            self,
            samples=self.samples if samples is None else samples,
            epochs=self.epochs if epochs is None else epochs,
            batch_size=self.batch_size if batch_size is None else batch_size,
        )
        if result.baseline == "example_full_trace" and batch_size is None:
            result = replace(result, batch_size=result.samples)
        _validate(result)
        return result


def _validate(case: TrainingCase) -> None:
    if case.device not in {"cpu", "cuda"}:
        raise ConfigError(f"device 必须为 cpu 或 cuda: {case.id}")
    if case.dim not in {2, 3}:
        raise ConfigError(f"dim 必须为 2 或 3: {case.id}")
    if len(case.sub_size) != case.dim or len(case.n_fine) != case.dim:
        raise ConfigError(f"sub_size/n_fine 与 dim 不一致: {case.id}")
    if any(value <= 0 for value in (*case.sub_size, *case.n_fine)):
        raise ConfigError(f"sub_size/n_fine 必须为正: {case.id}")
    if case.samples < 2 or case.epochs <= 0 or case.batch_size <= 0:
        raise ConfigError(f"samples/epochs/batch_size 取值无效: {case.id}")
    if not 0.0 < case.validation_fraction < 1.0:
        raise ConfigError(f"validation_fraction 必须位于 (0, 1): {case.id}")
    if not 0.0 < case.design_min < 1.0:
        raise ConfigError(f"design_min 必须位于 (0, 1): {case.id}")
    if case.learning_rate <= 0 or case.hidden_dim <= 0 or case.cpu_threads <= 0:
        raise ConfigError(f"learning_rate/hidden_dim/cpu_threads 必须为正: {case.id}")
    if not all(math.isfinite(value) for value in (
        *case.sub_size, case.validation_fraction, case.design_min,
        case.learning_rate, case.emax, case.nu, case.simp_penalty,
    )):
        raise ConfigError(f"配置包含非有限值: {case.id}")
    if case.emax <= 0 or not -1 < case.nu < 0.5 or case.simp_penalty <= 0:
        raise ConfigError(f"材料参数无效: {case.id}")
    if any(value < 2 for value in case.n_fine):
        raise ConfigError(f"各方向至少需要两个细单元以保留内部自由度: {case.id}")
    if case.baseline == "example_full_trace":
        if case.batch_size != case.samples or case.n_eval <= 0:
            raise ConfigError("full_trace 基线要求全批量训练及非空留出集。")
    elif case.baseline != "exploratory_linear_corner":
        raise ConfigError(f"未知 baseline: {case.baseline}")
    if case.baseline == "exploratory_linear_corner" and case.design_min >= 0.3:
        raise ConfigError(f"混合采样要求 design_min < 0.3: {case.id}")


def load() -> tuple[dict[str, Any], tuple[TrainingCase, ...]]:
    """读取配置；不导入数值库。"""
    try:
        raw = tomllib.loads(CASES_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"无法解析 {CASES_FILE}: {error}") from error
    definitions = {key: value for key, value in raw.items() if key.startswith("training_")}
    cases = []
    for registered in raw.get("cases", ()):
        pair_id = registered["pair_id"]
        values = definitions.get(pair_id)
        if values is None:
            raise ConfigError(f"缺少 pair 配置: {pair_id}")
        if values.get("baseline") == "example_full_trace":
            import importlib.util
            path = EXPERIMENT_DIR.parents[1] / "examples/piml_substructure_elasticity/_common.py"
            spec = importlib.util.spec_from_file_location("_gpu_example_defaults", path)
            defaults = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(defaults)
            values = dict(values, dim=2, sub_size=defaults.SUB_SIZE, n_fine=defaults.N_FINE,
                          samples=defaults.SHAPE_N_TRAIN, n_eval=defaults.SHAPE_N_EVAL,
                          validation_fraction=defaults.SHAPE_N_EVAL / (defaults.SHAPE_N_TRAIN + defaults.SHAPE_N_EVAL),
                          design_min=defaults.DENSITY_RANGE[0], epochs=defaults.SHAPE_EPOCHS,
                          batch_size=defaults.SHAPE_N_TRAIN, learning_rate=defaults.SHAPE_LEARNING_RATE,
                          hidden_dim=defaults.SHAPE_HIDDEN_DIM, seed=defaults.SHAPE_SEED,
                          emax=defaults.E_BASE, nu=defaults.NU, simp_penalty=defaults.SHAPE_SIMP_PENALTY)
        case = TrainingCase(
            id=registered["id"], pair_id=pair_id,
            device=registered["device"], summary=registered["summary"],
            dim=int(values["dim"]),
            sub_size=tuple(float(x) for x in values["sub_size"]),
            n_fine=tuple(int(x) for x in values["n_fine"]),
            samples=int(values["samples"]),
            validation_fraction=float(values["validation_fraction"]),
            design_min=float(values["design_min"]), epochs=int(values["epochs"]),
            batch_size=int(values["batch_size"]),
            learning_rate=float(values["learning_rate"]),
            hidden_dim=int(values["hidden_dim"]), seed=int(values["seed"]),
            cpu_threads=int(values["cpu_threads"]), emax=float(values["emax"]),
            nu=float(values["nu"]), simp_penalty=float(values["simp_penalty"]),
            baseline=values.get("baseline", "exploratory_linear_corner"), n_eval=int(values.get("n_eval", 200)),
        )
        _validate(case)
        cases.append(case)
    if len({case.id for case in cases}) != len(cases):
        raise ConfigError("case id 必须唯一。")
    return dict(raw.get("meta", {})), tuple(cases)


def pair(cases: tuple[TrainingCase, ...], pair_id: str) -> tuple[TrainingCase, TrainingCase]:
    """取得一个配置完全一致的 CPU/CUDA pair。"""
    matches = [case for case in cases if case.pair_id == pair_id]
    selected = {case.device: case for case in matches}
    if len(matches) != 2 or set(selected) != {"cpu", "cuda"}:
        raise ConfigError(f"pair {pair_id} 必须各含一个 cpu/cuda 工况。")
    cpu, cuda = selected["cpu"], selected["cuda"]
    if cpu.comparison_contract() != cuda.comparison_contract():
        raise ConfigError(f"pair {pair_id} 的计算配置不一致。")
    return cpu, cuda
