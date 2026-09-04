# -*- coding: utf-8 -*-
"""cases.toml 的加载、校验与模型转换.

本模块只负责 cases.toml 的数据结构反序列化与静态门禁检查,
不依赖 fealpy, 不执行任何数值计算。
"""

from __future__ import annotations

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

PANELS: Tuple[str, ...] = ("cache", "matvec", "solve")
METHODS: Tuple[str, ...] = ("fast", "standard", "voigt")


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
    mesh_type: str = "TetrahedronMesh"
    grid: str = "32^3"
    problem: str = "DivergenceFreePolynomialElasticity3D"
    method: str = "fast"
    methods: Tuple[str, ...] = ("fast", "standard", "voigt")
    device: str = "cpu"
    n: int = 32

    def to_command(self, repo_root: Path, overrides: Optional[Dict[str, Any]] = None) -> List[str]:
        """将工况配置还原为可执行的子进程命令行列表."""
        ov = overrides or {}
        cmd = [sys.executable, str(repo_root / self.script)]
        cmd.extend(self.args)

        # 动态参数覆盖
        if "n" in ov and ov["n"] is not None:
            if "--n" in cmd:
                idx = cmd.index("--n")
                cmd[idx + 1] = str(ov["n"])
            else:
                cmd.extend(["--n", str(ov["n"])])

        if "method" in ov and ov["method"] is not None:
            if "--method" in cmd:
                idx = cmd.index("--method")
                cmd[idx + 1] = str(ov["method"])
            else:
                cmd.extend(["--method", str(ov["method"])])

        if "device" in ov and ov["device"] is not None:
            if "--device" in cmd:
                idx = cmd.index("--device")
                cmd[idx + 1] = str(ov["device"])
            else:
                cmd.extend(["--device", str(ov["device"])])

        return cmd


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

        artifact_name = c.get("artifact", f"{case_id}.json")
        artifact_path = OUTPUT_DIR / artifact_name

        methods_raw = c.get("methods", [c.get("method", "fast")])
        if isinstance(methods_raw, list):
            methods_tuple = tuple(methods_raw)
        else:
            methods_tuple = (str(methods_raw),)

        case_obj = Case(
            id=case_id,
            panel=panel,
            summary=c.get("summary", ""),
            script=c.get("script", "experiments/ea_assembly_capability/run.py"),
            args=tuple(c.get("args", [])),
            output_mode=c.get("output_mode", "file"),
            artifact=artifact_name,
            artifact_path=artifact_path,
            role=c.get("role", ""),
            mesh_type=c.get("mesh_type", "TetrahedronMesh"),
            grid=c.get("grid", "32^3"),
            problem=c.get("problem", "DivergenceFreePolynomialElasticity3D"),
            method=c.get("method", "fast"),
            methods=methods_tuple,
            device=c.get("device", "cpu"),
            n=int(c.get("n", 32)),
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
        selected = [c for c in selected if c.id in target_set]
    return tuple(selected)


if __name__ == "__main__":
    fig, all_cases = load_cases()
    print(f"成功加载 figure: {fig.get('id')} ({len(all_cases)} cases)")
    for c in all_cases:
        print(f"  - [{c.panel}] {c.id} -> {c.artifact}")
