# -*- coding: utf-8 -*-
"""运行溯源的采集 (Hu--Zhang 拓扑优化投稿实验).

记录 Git revision、dirty 状态、时间戳、主机环境与关键依赖版本; 与
``experiments/topopt_capability/provenance.py`` 保持同一口径, 只去掉本实验用不到的
深度学习框架版本. 快照里的数字能否被引用, 取决于 ``reproducible()``: 只有在
干净工作区上跑出的产物才允许写进论文.

数值代码 (含自 FEALPy 移植的 backend / mesh / functionspace 等) 全部位于本仓库,
本仓库的 Git revision 与 dirty 状态即可钉住产生结果的全部源码. dirty 只看代码与配置,
不计入库的产物目录 ``results/``: 逐组运行时前一组刚写下的产物不应把后一组判为不可复现;
再分析读入的产物由各自记录的文件摘要 (``file_digest``) 钉住.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR, REPOSITORY_ROOT

# git status 的 pathspec: 全仓库, 但排除产物目录 results/
_DIRTY_PATHSPEC = (".", f":(exclude){OUTPUT_DIR.relative_to(REPOSITORY_ROOT).as_posix()}")


def _git(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str | None:
    if not cwd.is_dir():
        return None
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def file_digest(path: Path) -> dict[str, Any]:
    """对产物文件取 sha256 摘要; 文件缺失时如实记录 None, 不静默跳过."""
    try:
        relative = str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        relative = str(path)
    if not path.is_file():
        return {"path": relative, "sha256": None, "bytes": None}
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def collect() -> dict[str, Any]:
    """采集当前运行环境的溯源记录; ``git_dirty`` 不计产物目录 ``results/``."""
    status = _git("status", "--porcelain", "--", *_DIRTY_PATHSPEC)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
    }


def run_stamp() -> dict[str, Any]:
    """一次运行的溯源戳记: 本仓库的 revision/dirty + 是否可复现.

    落盘时逐次盖上, 而不是事后汇总时统一盖: 同一个 results/ 目录常横跨多次运行与
    多个代码版本, 事后补的戳记会把全部产物错标成最后一次的版本.
    """
    record = collect()
    return {
        "generated_at_utc": record.get("generated_at_utc"),
        "git_revision": record.get("git_revision"),
        "git_dirty": record.get("git_dirty"),
        "reproducible": reproducible(record),
    }


def reproducible(record: dict[str, Any]) -> bool:
    """判定该快照是否可作为论文证据.

    要求本仓库取得到 revision 且工作区 (产物目录 ``results/`` 除外) 干净.
    """
    return bool(record.get("git_revision")) and record.get("git_dirty") is False
