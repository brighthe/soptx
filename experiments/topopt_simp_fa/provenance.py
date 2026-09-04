# -*- coding: utf-8 -*-
"""运行环境与输入文件溯源 (FA 密度法拓扑优化实验)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from config import REPOSITORY_ROOT


def _git(*arguments: str) -> str | None:
    """读取 Git 状态字段, 失败时返回 ``None``."""
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=REPOSITORY_ROOT,
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_version(name: str) -> str | None:
    """读取已安装包版本."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def file_digest(path: Path) -> dict[str, Any]:
    """计算输入文件的 SHA-256 摘要."""
    relative = str(path.relative_to(REPOSITORY_ROOT))
    if not path.is_file():
        return {"path": relative, "exists": False}
    payload = path.read_bytes()
    return {
        "path": relative,
        "exists": True,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def capture(inputs: tuple[Path, ...]) -> dict[str, Any]:
    """采集本次运行的版本、环境和输入摘要."""
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "fealpy")
        },
        "inputs": [file_digest(path) for path in inputs],
    }
