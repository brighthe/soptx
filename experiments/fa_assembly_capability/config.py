# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与校验.

本模块只负责把注册表读成结构化的 ``Case`` 对象并做静态校验, 不触发任何计算.
运行调度见 ``run.py``, 报表呈现见 ``compare.py``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 本目录与仓库根。experiments/<name>/config.py 上溯两级即仓库根。
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

# 产物重定向方式, 与 ``cases.toml`` 的 ``output_mode`` 取值一一对应.
OUTPUT_MODES = ("dir", "file", "fixed")

# 装配测量的阶段分格.
PANELS = ("stage1", "stage2", "full")


class ConfigError(RuntimeError):
    """``cases.toml`` 结构或取值不合法."""


@dataclass(frozen=True)
class Case:
    """一个数据点的运行规格.

    属性:
        id: 唯一标识, 用于 ``run.py --cases``.
        panel: 所属图面分格, 取 ``PANELS`` 之一.
        role: 读取角色, 由 ``collect.py`` 按此分派提取逻辑.
        summary: 一句话说明, 用于 ``--list`` 输出.
        script: 仓库根起算的脚本相对路径.
        args: 传给该脚本的固定参数, 不依赖脚本缺省值.
        output_mode: 产物重定向方式, 取 ``OUTPUT_MODES`` 之一.
        artifact: ``fixed`` 时为仓库根起算的相对路径, 否则为 ``outputs/`` 下的文件名.
        launcher: 脚本前置启动器, 如 ``["mpiexec", "-n", "16"]``; 空元组表示直接跑.
        env: 附加环境变量, 与当前环境合并后传给子进程, 如 ``{"OMP_NUM_THREADS": "1"}``.
        extra: 其余键值 (如 ``level``, ``resolution``), 原样保留供 ``collect.py`` 使用.
    """

    id: str
    panel: str
    role: str
    summary: str
    script: str
    args: tuple[str, ...]
    output_mode: str
    artifact: str
    launcher: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    extra: dict[str, Any]

    @property
    def script_path(self) -> Path:
        """脚本的绝对路径."""
        return REPOSITORY_ROOT / self.script

    @property
    def artifact_path(self) -> Path:
        """产物的绝对路径.

        ``fixed`` 模式下产物落在上游脚本自己的目录里, 因此从仓库根解析;
        其余模式一律落在本目录 ``outputs/`` 下.
        """
        if self.output_mode == "fixed":
            return REPOSITORY_ROOT / self.artifact
        return OUTPUT_DIR / self.artifact

    def command(self, overrides: dict[str, Any] | None = None) -> list[str]:
        """构造完整的子进程 argv, 含产物重定向参数与动态覆盖参数.

        参数:
            overrides: 可选的参数覆盖映射 (如 `{'n': 48, 'chunk': 1024}`).

        返回:
            argv: 以 ``sys.executable`` 开头的参数列表.
        """
        import sys

        args_list = list(self.args)
        if overrides:
            for flag_name, val in overrides.items():
                if val is None:
                    continue
                flag = f"--{flag_name}"
                if flag in args_list:
                    idx = args_list.index(flag)
                    if idx + 1 < len(args_list):
                        args_list[idx + 1] = str(val)
                else:
                    args_list.extend([flag, str(val)])

        argv = [*self.launcher, sys.executable, str(self.script_path), *args_list]
        if self.output_mode == "dir":
            argv += ["--output-dir", str(OUTPUT_DIR)]
        elif self.output_mode == "file":
            argv += ["--output", str(self.artifact_path)]
        elif self.output_mode != "fixed":
            raise ConfigError(
                f"{self.id}: output_mode 只能取 {OUTPUT_MODES}, 实得 {self.output_mode!r}"
            )
        return argv

    def subprocess_env(self) -> dict[str, str] | None:
        """返回子进程环境变量; 未声明 ``env`` 时返回 ``None`` 表示继承当前环境."""
        import os

        if not self.env:
            return None
        merged = dict(os.environ)
        merged.update(dict(self.env))
        return merged


def _require(mapping: dict[str, Any], key: str, case_id: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"case {case_id!r} 缺少必填键 {key!r}")
    return mapping[key]


def load_cases(path: Path | None = None) -> tuple[dict[str, Any], tuple[Case, ...]]:
    """读取并校验 ``cases.toml``.

    参数:
        path: 注册表路径, ``None`` 时用本目录的 ``cases.toml``.

    返回:
        figure: ``[figure]`` 段, 图面元信息.
        cases: 按文件出现顺序排列的 ``Case`` 元组.

    异常:
        ConfigError: 缺少必填键、``id`` 重复、``panel``/``output_mode`` 取值非法,
            或 ``script`` 指向的文件不存在.
    """
    path = path or CASES_FILE
    payload = tomllib.loads(path.read_text(encoding="utf-8"))

    figure = payload.get("figure", {})
    raw_cases = payload.get("cases", [])
    if not raw_cases:
        raise ConfigError(f"{path}: 未注册任何 [[cases]]")

    known = {"id", "panel", "role", "summary", "script", "args",
             "output_mode", "artifact", "launcher", "env"}
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
            launcher=tuple(str(a) for a in raw.get("launcher", ())),
            env=tuple((str(k), str(v)) for k, v in raw.get("env", {}).items()),
            extra={k: v for k, v in raw.items() if k not in known},
        )
        if not case.script_path.is_file():
            raise ConfigError(f"{case_id}: script 不存在 -> {case.script_path}")
        cases.append(case)

    return figure, tuple(cases)


def select(
    cases: tuple[Case, ...],
    *,
    case_id: str | None = None,
    case_ids: list[str] | tuple[str, ...] | str | None = None,
    panel: str | None = None,
) -> tuple[Case, ...]:
    """按 ``id`` 或 ``panel`` 过滤 case.

    参数:
        cases: 全部 case.
        case_id: 单个 case id (兼容旧代码).
        case_ids: 匹配的一个或多个 case id.
        panel: 图面分格; ``None`` 表示不按分格过滤.

    返回:
        selected: 过滤后的 case 元组, 保持原顺序.
    """
    selected = cases
    target_ids: set[str] = set()
    if case_id is not None:
        target_ids.add(case_id)
    if case_ids is not None:
        if isinstance(case_ids, str):
            target_ids.add(case_ids)
        else:
            target_ids.update(case_ids)

    if target_ids:
        selected = tuple(c for c in selected if c.id in target_ids)
    if panel is not None:
        selected = tuple(c for c in selected if c.panel == panel)
    if not selected:
        raise ConfigError(
            f"没有匹配的 case (target_ids={sorted(target_ids)}, panel={panel!r}); "
            f"可用 id: {', '.join(c.id for c in cases)}"
        )
    return selected
