# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与校验 (子结构缩聚变密度拓扑优化实验)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

FA_EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "topopt_simp_fa"

SUPPORTED_TRACES = ("full_trace", "linear_corner")
SUPPORTED_REDUCTIONS = ("exact_schur",)
SUPPORTED_ROLES = (
    "fa-equivalence-implementation-gate",
    "exact-reference-for-piml",
)
SUPPORTED_FILTERS = ("sensitivity", "density")


class ConfigError(RuntimeError):
    """``cases.toml`` 结构或取值不合法."""


@dataclass(frozen=True)
class TopOptCase:
    id: str
    dim: int
    trace: str          # "full_trace" | "linear_corner"
    reduction: str      # "exact_schur"
    role: str
    domain: tuple[float, ...]
    n_sub: tuple[int, ...]
    n_fine: tuple[int, ...]
    integration_order: int
    chunk_size: int
    volfrac: float
    filter_radius: float
    filter_type: str
    max_iter: int
    tol_change: float
    simp_penalty: float
    emin: float
    emax: float
    nu: float
    p_load: float
    seed: int
    fa_reference: str
    summary: str

    @property
    def output_dir(self) -> Path:
        return OUTPUT_DIR / self.id


def _require(raw: dict[str, Any], key: str, case_id: str) -> Any:
    """取必填字段; 本目录不给求解相关参数留缺省."""
    if key not in raw:
        raise ConfigError(f"工况 {case_id} 缺少必填字段 {key}.")
    return raw[key]


def _parse_case(raw: dict[str, Any]) -> TopOptCase:
    case_id = raw.get("id")
    if not case_id:
        raise ConfigError("每条 [[cases]] 必须给出非空 id.")

    domain = tuple(float(x) for x in _require(raw, "domain", case_id))
    if len(domain) not in (4, 6):
        raise ConfigError(f"工况 {case_id} 的 domain 长度必须为 4 (2D) 或 6 (3D).")
    dim = 3 if len(domain) == 6 else 2

    n_sub = tuple(int(x) for x in _require(raw, "n_sub", case_id))
    n_fine = tuple(int(x) for x in _require(raw, "n_fine", case_id))
    if len(n_sub) != dim or len(n_fine) != dim:
        raise ConfigError(
            f"工况 {case_id} 的 n_sub/n_fine 长度必须等于维数 {dim}."
        )

    trace = str(_require(raw, "trace", case_id))
    if trace not in SUPPORTED_TRACES:
        raise ConfigError(
            f"工况 {case_id} 的 trace={trace!r} 不受支持; "
            f"可选值为 {SUPPORTED_TRACES}."
        )

    reduction = str(raw.get("reduction", "exact_schur"))
    if reduction not in SUPPORTED_REDUCTIONS:
        raise ConfigError(
            f"工况 {case_id} 的 reduction={reduction!r} 不受支持; "
            f"本目录只跑精确缩聚, 可选值为 {SUPPORTED_REDUCTIONS}. "
            "PIML 近似路径由 experiments/piml_substructure_topopt 维护."
        )

    role = str(_require(raw, "role", case_id))
    if role not in SUPPORTED_ROLES:
        raise ConfigError(
            f"工况 {case_id} 的 role={role!r} 不受支持; "
            f"可选值为 {SUPPORTED_ROLES}."
        )

    filter_type = str(_require(raw, "filter_type", case_id))
    if filter_type not in SUPPORTED_FILTERS:
        raise ConfigError(
            f"工况 {case_id} 的 filter_type={filter_type!r} 尚未实现; "
            f"可选值为 {SUPPORTED_FILTERS}."
        )

    volfrac = float(_require(raw, "volfrac", case_id))
    if not 0.0 < volfrac <= 1.0:
        raise ConfigError(f"工况 {case_id} 的 volfrac 必须位于 (0, 1].")

    chunk_size = int(_require(raw, "chunk_size", case_id))
    if chunk_size <= 0:
        raise ConfigError(f"工况 {case_id} 的 chunk_size 必须为正整数.")

    integration_order = int(_require(raw, "integration_order", case_id))
    if integration_order <= 0:
        raise ConfigError(f"工况 {case_id} 的 integration_order 必须为正整数.")

    return TopOptCase(
        id=str(case_id),
        dim=dim,
        trace=trace,
        reduction=reduction,
        role=role,
        domain=domain,
        n_sub=n_sub,
        n_fine=n_fine,
        integration_order=integration_order,
        chunk_size=chunk_size,
        volfrac=volfrac,
        filter_radius=float(_require(raw, "filter_radius", case_id)),
        filter_type=filter_type,
        max_iter=int(_require(raw, "max_iter", case_id)),
        tol_change=float(_require(raw, "tol_change", case_id)),
        simp_penalty=float(_require(raw, "simp_penalty", case_id)),
        emin=float(_require(raw, "emin", case_id)),
        emax=float(_require(raw, "emax", case_id)),
        nu=float(_require(raw, "nu", case_id)),
        p_load=float(_require(raw, "p_load", case_id)),
        seed=int(_require(raw, "seed", case_id)),
        fa_reference=str(raw.get("fa_reference", "")),
        summary=str(raw.get("summary", "")),
    )


def load() -> tuple[dict[str, Any], tuple[TopOptCase, ...]]:
    """加载 cases.toml 并解析全部工况."""
    if not CASES_FILE.is_file():
        raise ConfigError(f"注册表不存在: {CASES_FILE}")
    try:
        raw = tomllib.loads(CASES_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"解析 {CASES_FILE} 失败: {error}") from error

    cases = tuple(_parse_case(rc) for rc in raw.get("cases", []))
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ConfigError(f"工况 id 重复: {case.id}")
        seen.add(case.id)
    return raw.get("meta", {}), cases


def get_case(case_id: str) -> TopOptCase:
    """按 id 取单个工况."""
    _, cases = load()
    for case in cases:
        if case.id == case_id:
            return case
    raise ConfigError(f"未找到工况: {case_id}")
