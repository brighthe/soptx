# -*- coding: utf-8 -*-
"""运行溯源的采集 (子结构缩聚变密度拓扑优化实验).

与 experiments/piml_substructure_topopt/provenance.py 同构, 但本目录不依赖
torch, 因此不采集 CUDA 信息。
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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


def capture(inputs: Iterable[Path] = ()) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    dirty = None
    status = _git("status", "--porcelain")
    if status is not None:
        dirty = bool(status.strip())

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
        "packages": {
            "fealpy": _package_version("fealpy"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
        },
        "inputs": [file_digest(Path(path)) for path in inputs],
    }
