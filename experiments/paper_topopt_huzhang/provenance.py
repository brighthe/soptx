# -*- coding: utf-8 -*-
"""运行溯源的采集 (Hu--Zhang 拓扑优化投稿实验).

记录 Git revision、dirty 状态、时间戳、主机环境与关键依赖版本; 与
``experiments/topopt_capability/provenance.py`` 保持同一口径, 只去掉本实验用不到的
深度学习框架版本. 快照里的数字能否被引用, 取决于 ``reproducible()``: 只有在
干净工作区上跑出的产物才允许写进论文.

fealpy 通常以 editable 方式装自一个可变的工作副本, 此时 ``importlib.metadata``
给出的版本号钉不住任何代码 (副本改了版本号也不动). 因此这里额外记录 fealpy 源码
所在的 Git revision 与 dirty 状态, 并把它纳入 ``reproducible()`` 的判定。
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import REPOSITORY_ROOT


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


def _source_root(name: str) -> Path | None:
    """返回已导入包的源码目录; 包未安装或无文件路径时返回 None."""
    from importlib.util import find_spec

    try:
        spec = find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _checkout_record(name: str) -> dict[str, Any]:
    """采集某个依赖包所在 Git 工作副本的 revision 与 dirty 状态."""
    root = _source_root(name)
    if root is None:
        return {"path": None, "git_revision": None, "git_dirty": None}
    status = _git("status", "--porcelain", cwd=root)
    return {
        "path": str(root),
        "git_revision": _git("rev-parse", "HEAD", cwd=root),
        "git_dirty": None if status is None else bool(status),
    }


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
    """采集当前运行环境的溯源记录."""
    status = _git("status", "--porcelain")
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
        "fealpy": _package_version("fealpy"),
        "fealpy_checkout": _checkout_record("fealpy"),
    }


def run_stamp() -> dict[str, Any]:
    """一次运行的溯源戳记: 本仓库与 fealpy 副本的 revision/dirty + 是否可复现.

    落盘时逐次盖上, 而不是事后汇总时统一盖: 同一个 outputs/ 目录常横跨多次运行与
    多个代码版本, 事后补的戳记会把全部产物错标成最后一次的版本.
    """
    record = collect()
    checkout = record.get("fealpy_checkout") or {}
    return {
        "generated_at_utc": record.get("generated_at_utc"),
        "git_revision": record.get("git_revision"),
        "git_dirty": record.get("git_dirty"),
        "fealpy_git_revision": checkout.get("git_revision"),
        "fealpy_git_dirty": checkout.get("git_dirty"),
        "reproducible": reproducible(record),
    }


def reproducible(record: dict[str, Any]) -> bool:
    """判定该快照是否可作为论文证据.

    要求本仓库与 fealpy 工作副本都取得到 revision 且都干净: 只钉住本仓库是不够的,
    数值结果同样取决于 fealpy 副本里那些未提交的改动.
    """
    if not record.get("git_revision") or record.get("git_dirty") is not False:
        return False
    checkout = record.get("fealpy_checkout") or {}
    return bool(checkout.get("git_revision")) and checkout.get("git_dirty") is False
