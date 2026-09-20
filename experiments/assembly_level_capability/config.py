# -*- coding: utf-8 -*-
"""cases.toml 的加载、校验与模型转换.

本模块只负责 cases.toml 的数据结构反序列化与静态门禁检查,
不依赖 fealpy, 不执行任何数值计算。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

PANELS: Tuple[str, ...] = ("bandwidth", "matvec", "solve")
SCHEMES: Tuple[str, ...] = ("stored-b", "ea", "fa", "pa", "shared-ke")
PSEUDO_SCHEMES: Tuple[str, ...] = ("all", "memcpy")

# 全部工况统一的单线程口径: 子进程与直接启动的 worker 都注入这组环境变量。
THREAD_ENV: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


class ConfigError(ValueError):
    """cases.toml 配置校验失败时抛出."""


@dataclass(frozen=True)
class Case:
    """单个工况的强类型声明."""

    id: str
    panel: str
    summary: str
    script: str
    args: Tuple[str, ...]
    output_mode: str
    artifact: str
    artifact_path: Path
    role: str = ""
    mesh_type: str = "HexahedronMesh"
    grid: str = "48^3"
    problem: str = "LinearElasticity3D_UnitCube"
    scheme: str = "ea"
    device: str = "cpu"
    n: int = 48
    repeats: int = 20

    def to_command(self, repo_root: Path, overrides: Optional[Dict[str, Any]] = None) -> List[str]:
        """将工况配置还原为可执行的子进程命令行列表."""
        ov = overrides or {}
        cmd = [sys.executable, str(repo_root / self.script)]
        cmd.extend(self.args)

        for key, flag in (("n", "--n"), ("scheme", "--scheme"), ("repeats", "--repeats")):
            if key in ov and ov[key] is not None:
                if flag in cmd:
                    idx = cmd.index(flag)
                    cmd[idx + 1] = str(ov[key])
                else:
                    cmd.extend([flag, str(ov[key])])
        return cmd

    def subprocess_env(self) -> Dict[str, str]:
        """子进程环境: 继承当前环境并强制单线程 BLAS/OpenMP."""
        env = dict(os.environ)
        env.update(THREAD_ENV)
        return env


def load_cases(path: Optional[Path] = None) -> Tuple[Dict[str, Any], Tuple[Case, ...]]:
    """解析并校验 cases.toml 文件."""
    cases_path = path or DEFAULT_CASES_PATH
    if not cases_path.is_file():
        raise ConfigError(f"未找到 cases.toml: {cases_path}")

    with cases_path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except Exception as err:
            raise ConfigError(f"TOML 语法解析失败: {err}") from err

    figure = data.get("figure", {})
    raw_cases = data.get("cases", [])
    if not raw_cases:
        raise ConfigError("cases.toml 中未定义任何 [[cases]] 工况")

    cases: List[Case] = []
    known_ids = set()

    for idx, c in enumerate(raw_cases):
        case_id = c.get("id")
        if not case_id:
            raise ConfigError(f"第 {idx + 1} 个工况缺少必填项 'id'")
        if case_id in known_ids:
            raise ConfigError(f"工况 id 重复: '{case_id}'")
        known_ids.add(case_id)

        panel = c.get("panel", "")
        if panel not in PANELS:
            raise ConfigError(f"工况 '{case_id}' 的 panel '{panel}' 非法, 必须为 {PANELS} 之一")

        scheme = c.get("scheme", "")
        if scheme not in SCHEMES + PSEUDO_SCHEMES:
            raise ConfigError(
                f"工况 '{case_id}' 的 scheme '{scheme}' 非法, 必须为 {SCHEMES + PSEUDO_SCHEMES} 之一"
            )
        if panel == "bandwidth" and scheme != "memcpy":
            raise ConfigError(f"工况 '{case_id}': bandwidth 面板的 scheme 必须为 'memcpy'")
        if panel in ("matvec", "solve") and scheme not in SCHEMES:
            raise ConfigError(f"工况 '{case_id}': {panel} 面板的 scheme 必须为 {SCHEMES} 之一")

        if "artifact" in c:
            raise ConfigError(
                f"工况 '{case_id}' 不应声明 'artifact': 产物名一律由 id 推导为 '{case_id}.json'"
            )
        artifact_name = f"{case_id}.json"
        artifact_path = OUTPUT_DIR / artifact_name

        case_obj = Case(
            id=case_id,
            panel=panel,
            summary=c.get("summary", ""),
            script=c.get("script", "experiments/assembly_level_capability/run.py"),
            args=tuple(c.get("args", [])),
            output_mode=c.get("output_mode", "file"),
            artifact=artifact_name,
            artifact_path=artifact_path,
            role=c.get("role", ""),
            mesh_type=c.get("mesh_type", "HexahedronMesh"),
            grid=c.get("grid", "48^3"),
            problem=c.get("problem", "LinearElasticity3D_UnitCube"),
            scheme=scheme,
            device=c.get("device", "cpu"),
            n=int(c.get("n", 48)),
            repeats=int(c.get("repeats", 20)),
        )
        cases.append(case_obj)

    return figure, tuple(cases)


def select(
    cases: Tuple[Case, ...],
    case_ids: Optional[Sequence[str]] = None,
    panel: Optional[str] = None,
) -> Tuple[Case, ...]:
    """根据过滤条件筛选工况子集."""
    selected = list(cases)
    if panel:
        selected = [c for c in selected if c.panel == panel]
    if case_ids:
        target_set = set(case_ids)
        unknown = target_set - {c.id for c in cases}
        if unknown:
            raise ConfigError(f"未注册的 case id: {sorted(unknown)}")
        selected = [c for c in selected if c.id in target_set]
    return tuple(selected)


if __name__ == "__main__":
    fig, all_cases = load_cases()
    print(f"成功加载 figure: {fig.get('id')} ({len(all_cases)} cases)")
    for c in all_cases:
        print(f"  - [{c.panel}] {c.id} -> {c.artifact}")
