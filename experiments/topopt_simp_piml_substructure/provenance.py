# -*- coding: utf-8 -*-
"""运行溯源的采集 (PIML 子结构拓扑优化实验)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import REPOSITORY_ROOT


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
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
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _cuda_device_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def file_digest(path: Path) -> dict[str, Any]:
    try:
        relative = str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        relative = str(path)
    if not path.is_file():
        return {"path": relative, "exists": False}
    content = path.read_bytes()
    return {
        "path": relative,
        "exists": True,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def capture() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    dirty = None
    status = _git("status", "--porcelain")
    if status is not None:
        dirty = bool(status.strip())

    packages = {
        "fealpy": _package_version("fealpy"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": head,
            "dirty": dirty,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "cuda": {
            "available": _cuda_device_name() is not None,
            "device_name": _cuda_device_name(),
        },
    }
