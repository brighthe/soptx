"""加载并校验精确子结构分析实验的工况注册表."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import tomllib


CURRENT_DIR = Path(__file__).resolve().parent
CASES_FILE = CURRENT_DIR / "cases.toml"
OUTPUT_DIR = CURRENT_DIR / "outputs"

_TASKS = {
    "full_trace_convergence",
    "linear_corner_convergence",
    "density_update_consistency",
    "route_cost",
}
_PROBLEM_DIMENSIONS = {
    "HarmonicPoly2D": 2,
    "HarmonicPoly3D": 3,
    "CantileverCorner2d": 2,
    "FullMBBBeam3d": 3,
}
_SOLVERS = {"scipy", "mumps"}
_ROUTES = {"fa", "full_trace", "linear_corner"}


class ConfigError(ValueError):
    """工况注册表不完整或不一致."""


@dataclass(frozen=True)
class AnalysisCase:
    """统一入口使用的单个分析工况."""

    id: str
    summary: str
    task: str
    problem: str
    dim: int
    solve_method: str
    degree: int | None = None
    levels: int | None = None
    n_sub: tuple[int, ...] | None = None
    n_fine: tuple[int, ...] | None = None
    density: str | None = None
    routes: tuple[str, ...] = ()
    warmup: int | None = None
    repeat: int | None = None
    monitor_interval: float = 0.5
    expected_displacement_sha256: str | None = None
    expected_strain_energy: float | None = None


def _positive_int(raw: dict[str, Any], name: str, case_id: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{case_id}: {name} 必须为正整数")
    return value


def _shape(
    raw: dict[str, Any],
    name: str,
    dim: int,
    case_id: str,
) -> tuple[int, ...]:
    value = raw.get(name)
    if not isinstance(value, list) or len(value) != dim:
        raise ConfigError(f"{case_id}: {name} 必须包含 {dim} 个正整数")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise ConfigError(f"{case_id}: {name} 必须包含 {dim} 个正整数")
    return tuple(value)


def _parse_case(raw: dict[str, Any]) -> AnalysisCase:
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ConfigError("每个工况必须提供非空 id")
    if ".." in case_id or any(char in case_id for char in ("/", "\\")):
        raise ConfigError(f"{case_id}: id 不得包含路径分隔符或相对路径")
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ConfigError(f"{case_id}: summary 不能为空")
    task = raw.get("task")
    if task not in _TASKS:
        raise ConfigError(f"{case_id}: 未知 task {task!r}")
    problem = raw.get("problem")
    if problem not in _PROBLEM_DIMENSIONS:
        raise ConfigError(f"{case_id}: 未知 problem {problem!r}")
    dim = _PROBLEM_DIMENSIONS[problem]
    solve_method = raw.get("solve_method", "mumps")
    if solve_method not in _SOLVERS:
        raise ConfigError(f"{case_id}: solve_method 必须为 scipy 或 mumps")

    common = dict(
        id=case_id,
        summary=summary,
        task=task,
        problem=problem,
        dim=dim,
        solve_method=solve_method,
    )
    if task in ("full_trace_convergence", "linear_corner_convergence"):
        if not problem.startswith("HarmonicPoly"):
            raise ConfigError(f"{case_id}: 收敛工况必须使用 HarmonicPoly2D/3D")
        degree = _positive_int(raw, "degree", case_id)
        if degree != 1:
            raise ConfigError(f"{case_id}: 当前实验固定使用 Q1，degree 必须为 1")
        levels = _positive_int(raw, "levels", case_id)
        if levels < 2:
            raise ConfigError(f"{case_id}: levels 必须不小于 2")
        return AnalysisCase(**common, degree=degree, levels=levels)

    if task == "density_update_consistency":
        if problem not in ("CantileverCorner2d", "FullMBBBeam3d"):
            raise ConfigError(f"{case_id}: 密度更新工况必须使用 CantileverCorner2d/FullMBBBeam3d")
        degree = _positive_int(raw, "degree", case_id)
        if degree != 1:
            raise ConfigError(f"{case_id}: 当前实验固定使用 Q1")
        routes = raw.get("routes")
        if routes not in (["full_trace"], ["linear_corner"]):
            raise ConfigError(f"{case_id}: 密度更新工况须指定一种子结构接口")
        if ("n_sub" in raw) != ("n_fine" in raw):
            raise ConfigError(f"{case_id}: n_sub 与 n_fine 必须同时提供")
        n_sub = _shape(raw, "n_sub", dim, case_id) if "n_sub" in raw else None
        n_fine = _shape(raw, "n_fine", dim, case_id) if "n_fine" in raw else None
        monitor_interval = raw.get("monitor_interval", 0.5)
        if (
            isinstance(monitor_interval, bool)
            or not isinstance(monitor_interval, (int, float))
            or not math.isfinite(monitor_interval)
            or monitor_interval <= 0
        ):
            raise ConfigError(f"{case_id}: monitor_interval 必须为正数")
        return AnalysisCase(
            **common,
            degree=degree,
            routes=tuple(routes),
            n_sub=n_sub,
            n_fine=n_fine,
            monitor_interval=float(monitor_interval),
        )

    if task == "route_cost":
        if problem != "CantileverCorner2d":
            raise ConfigError(
                f"{case_id}: route_cost 必须使用 CantileverCorner2d"
            )
        if solve_method != "mumps":
            raise ConfigError(f"{case_id}: route_cost 必须使用 MUMPS")
        degree = _positive_int(raw, "degree", case_id)
        if degree != 1:
            raise ConfigError(f"{case_id}: 当前实验固定使用 Q1")
        n_sub = _shape(raw, "n_sub", dim, case_id)
        n_fine = _shape(raw, "n_fine", dim, case_id)
        if raw.get("density") != "pattern_a":
            raise ConfigError(f"{case_id}: route_cost 必须使用 pattern_a")
        routes = raw.get("routes")
        if (
            not isinstance(routes, list)
            or len(routes) != 1
            or routes[0] not in _ROUTES
        ):
            raise ConfigError(
                f"{case_id}: route_cost 必须指定一种 fa/full_trace/linear_corner 路径"
            )
        warmup = raw.get("warmup")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ConfigError(f"{case_id}: warmup 必须为非负整数")
        repeat = _positive_int(raw, "repeat", case_id)
        monitor_interval = raw.get("monitor_interval", 0.5)
        if (
            isinstance(monitor_interval, bool)
            or not isinstance(monitor_interval, (int, float))
            or not math.isfinite(monitor_interval)
            or monitor_interval <= 0
        ):
            raise ConfigError(f"{case_id}: monitor_interval 必须为正数")
        expected_hash = raw.get("expected_displacement_sha256")
        if expected_hash is not None and (
            not isinstance(expected_hash, str) or len(expected_hash) != 64
        ):
            raise ConfigError(
                f"{case_id}: expected_displacement_sha256 必须为 64 位字符串"
            )
        expected_energy = raw.get("expected_strain_energy")
        if expected_energy is not None and (
            isinstance(expected_energy, bool)
            or not isinstance(expected_energy, (int, float))
            or not math.isfinite(expected_energy)
            or expected_energy <= 0.0
        ):
            raise ConfigError(
                f"{case_id}: expected_strain_energy 必须为有限正数"
            )
        return AnalysisCase(
            **common,
            degree=degree,
            n_sub=n_sub,
            n_fine=n_fine,
            density="pattern_a",
            routes=tuple(routes),
            warmup=warmup,
            repeat=repeat,
            monitor_interval=float(monitor_interval),
            expected_displacement_sha256=expected_hash,
            expected_strain_energy=(
                None if expected_energy is None else float(expected_energy)
            ),
        )
    raise ConfigError(f"{case_id}: 未支持的 task {task}")


def load() -> tuple[dict[str, Any], tuple[AnalysisCase, ...]]:
    """读取全部注册工况并检查 id 唯一性."""
    if not CASES_FILE.is_file():
        raise ConfigError(f"注册表不存在: {CASES_FILE}")
    try:
        raw = tomllib.loads(CASES_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"解析 {CASES_FILE} 失败: {error}") from error
    cases = tuple(_parse_case(item) for item in raw.get("cases", []))
    if not cases:
        raise ConfigError("cases.toml 至少需要一个工况")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ConfigError("工况 id 不得重复")
    return raw.get("meta", {}), cases
