# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与配置解析 (PIML 子结构拓扑优化实验)."""

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


class ConfigError(RuntimeError):
    """``cases.toml`` 结构或取值不合法."""


@dataclass(frozen=True)
class TopOptCase:
    id: str
    dim: int
    solver_mode: str  # "piml_route_a", "piml_route_b", "fea_baseline"
    domain: tuple[float, ...]
    n_sub: tuple[int, ...]
    n_fine: tuple[int, ...]
    volfrac: float
    filter_radius: float
    filter_type: str  # "sensitivity", "density"
    max_iter: int
    tol_change: float
    simp_penalty: float
    emin: float
    emax: float
    nu: float
    p_load: float
    seed: int
    summary: str


def load() -> tuple[dict[str, Any], tuple[TopOptCase, ...]]:
    """加载 cases.toml 并解析工况."""
    if not CASES_FILE.is_file():
        raise ConfigError(f"注册表不存在: {CASES_FILE}")
    try:
        raw = tomllib.loads(CASES_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"解析 {CASES_FILE} 失败: {error}") from error

    meta = raw.get("meta", {})
    raw_cases = raw.get("cases", [])
    cases: list[TopOptCase] = []
    for rc in raw_cases:
        domain_val = tuple(float(x) for x in rc.get("domain", [0.0, 12.0, 0.0, 2.0]))
        dim = 3 if len(domain_val) == 6 else 2
        c = TopOptCase(
            id=rc["id"],
            dim=dim,
            solver_mode=rc.get("solver_mode", "piml_route_a"),
            domain=domain_val,
            n_sub=tuple(int(x) for x in rc.get("n_sub", [12, 2])),
            n_fine=tuple(int(x) for x in rc.get("n_fine", [5, 5])),
            volfrac=float(rc.get("volfrac", 0.5)),
            filter_radius=float(rc.get("filter_radius", 0.3)),
            filter_type=rc.get("filter_type", "sensitivity"),
            max_iter=int(rc.get("max_iter", 100)),
            tol_change=float(rc.get("tol_change", 0.001)),
            simp_penalty=float(rc.get("simp_penalty", 3.0)),
            emin=float(rc.get("emin", 1e-6)),
            emax=float(rc.get("emax", 1.0)),
            nu=float(rc.get("nu", 0.3)),
            p_load=float(rc.get("p_load", -1.0)),
            seed=int(rc.get("seed", 2026)),
            summary=rc.get("summary", ""),
        )
        cases.append(c)
    return meta, tuple(cases)
