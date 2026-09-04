# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与校验 (PIML 能力验证).

本模块只负责把注册表读成结构化的 ``Case`` 对象并做静态校验, 不触发任何计算.
运行调度见 ``run.py``, 产物读取见 ``collect.py``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FIGURE_DATA_DIR = EXPERIMENT_DIR / "figure_data"

OUTPUT_MODES = ("dir", "file", "fixed")
PANELS = ("a", "b", "c")


class ConfigError(RuntimeError):
    """``cases.toml`` 结构或取值不合法."""


@dataclass(frozen=True)
class Case:
    id: str
    panel: str
    role: str
    summary: str
    script: str
    args: tuple[str, ...]
    output_mode: str
    artifact: str
    extra: dict[str, Any]

    @property
    def script_path(self) -> Path:
        return REPOSITORY_ROOT / self.script

    @property
    def artifact_path(self) -> Path:
        if self.output_mode == "fixed":
            return REPOSITORY_ROOT / self.artifact
        return OUTPUT_DIR / self.artifact

    def command(self) -> list[str]:
        import sys
        argv = [sys.executable, str(self.script_path), *self.args]
        if self.output_mode == "dir":
            argv += ["--output-dir", str(OUTPUT_DIR)]
        elif self.output_mode == "file":
            argv += ["--output", str(self.artifact_path)]
        elif self.output_mode != "fixed":
            raise ConfigError(
                f"{self.id}: output_mode 只能取 {OUTPUT_MODES}, 实得 {self.output_mode!r}"
            )
        return argv


def _require(mapping: dict[str, Any], key: str, case_id: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"case {case_id!r} 缺少必填键 {key!r}")
    return mapping[key]


def load_cases(path: Path | None = None) -> tuple[dict[str, Any], tuple[Case, ...]]:
    path = path or CASES_FILE
    payload = tomllib.loads(path.read_text(encoding="utf-8"))

    figure = payload.get("figure", {})
    raw_cases = payload.get("cases", [])
    if not raw_cases:
        raise ConfigError(f"{path}: 未注册任何 [[cases]]")

    known = {"id", "panel", "role", "summary", "script", "args",
             "output_mode", "artifact"}
    cases: list[Case] = []
    seen: set[str] = set()
    for raw in raw_cases:
        case_id = raw.get("id", "<无 id>")
        if case_id in seen:
            raise ConfigError(f"case id 重复: {case_id!r}")
        seen.add(case_id)

        panel = _require(raw, "panel", case_id)
        if panel not in PANELS:
            raise ConfigError(f"{case_id}: panel 只能取 {PANELS}, 实得 {panel!r}")
        output_mode = _require(raw, "output_mode", case_id)
        if output_mode not in OUTPUT_MODES:
            raise ConfigError(
                f"{case_id}: output_mode 只能取 {OUTPUT_MODES}, 实得 {output_mode!r}"
            )

        case = Case(
            id=case_id,
            panel=panel,
            role=_require(raw, "role", case_id),
            summary=raw.get("summary", ""),
            script=_require(raw, "script", case_id),
            args=tuple(str(a) for a in _require(raw, "args", case_id)),
            output_mode=output_mode,
            artifact=_require(raw, "artifact", case_id),
            extra={k: v for k, v in raw.items() if k not in known},
        )
        if not case.script_path.is_file():
            raise ConfigError(f"{case_id}: script 不存在 -> {case.script_path}")
        cases.append(case)

    return figure, tuple(cases)


def select(cases: tuple[Case, ...], *, case_id: str | None = None,
           panel: str | None = None) -> tuple[Case, ...]:
    selected = cases
    if case_id is not None:
        selected = tuple(c for c in selected if c.id == case_id)
    if panel is not None:
        if panel == "c":
            selected = tuple(c for c in selected if c.id == "b-piml-exact")
        else:
            selected = tuple(c for c in selected if c.panel == panel)
    if not selected:
        available = ", ".join(c.id for c in cases)
        raise ConfigError(
            f"没有匹配的 case (case_id={case_id!r}, panel={panel!r}); 可用 id: {available}"
        )
    return selected
