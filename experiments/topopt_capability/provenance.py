# -*- coding: utf-8 -*-
"""运行溯源的采集 (PIML 能力验证).

记录 Git revision、dirty 状态、运行时间戳、主机环境、Python/PyTorch/JAX/FEALPy 版本
以及 GPU 设备型号 (如存在)。
"""

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
        return {"path": relative, "sha256": None, "bytes": None}
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def collect() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    revision = _git("rev-parse", "HEAD")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": revision,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "jax": _package_version("jax"),
        "numpy": _package_version("numpy"),
        "fealpy": _package_version("fealpy"),
        "cuda_device": _cuda_device_name(),
    }


def reproducible(record: dict[str, Any]) -> bool:
    return bool(record.get("git_revision")) and record.get("git_dirty") is False
