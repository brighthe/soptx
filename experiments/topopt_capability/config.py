# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与校验 (TopOpt 平台能力验证)."""

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

PANELS = ("a", "b")


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
    outdir: str
    artifact: str
    collect_as: str | None = None

    @property
    def script_path(self) -> Path:
        return REPOSITORY_ROOT / self.script

    @property
    def outdir_path(self) -> Path:
        """该用例运行时的 ``--outdir``."""
        return OUTPUT_DIR / self.outdir

    @property
    def artifact_path(self) -> Path:
        """该用例直接写出的产物路径."""
        return self.outdir_path / self.artifact

    @property
    def collected_path(self) -> Path:
        """收编进快照基目录后的产物路径; 无 ``collect_as`` 时即原路径."""
        if self.collect_as is None:
            return self.artifact_path
        return OUTPUT_DIR / self.collect_as


def load() -> tuple[dict[str, Any], tuple[Case, ...]]:
    if not CASES_FILE.is_file():
        raise ConfigError(f"注册表不存在: {CASES_FILE}")
    try:
        raw = tomllib.loads(CASES_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"解析 {CASES_FILE} 失败: {error}") from error

    fig = raw.get("figure", {})
    if "base" not in fig:
        raise ConfigError("[figure] 缺少 base: 快照产物的汇总目录名")
    raw_cases = raw.get("cases", [])
    cases: list[Case] = []
    for rc in raw_cases:
        c = Case(
            id=rc["id"],
            panel=rc["panel"],
            role=rc["role"],
            summary=rc["summary"],
            script=rc["script"],
            args=tuple(rc.get("args", [])),
            outdir=rc["outdir"],
            artifact=rc["artifact"],
            collect_as=rc.get("collect_as"),
        )
        cases.append(c)
    return fig, tuple(cases)
