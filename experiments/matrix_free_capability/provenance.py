# -*- coding: utf-8 -*-
"""运行溯源的采集.

上游脚本自己不写溯源: ``validate.py`` 不产出 ``environment`` 块,
``benchmark_cpu_ea.py`` 连时间戳都不写。因此"这批数字出自哪个 revision"
无法从产物本身判定, 只能由本模块在采集侧补上。

溯源写进 ``figure_data/fig2_data.json``, 与数字同一次落盘、同一次提交,
使快照可被机器校验, 而不是只能靠重跑核对。
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
    """在仓库根执行一条 ``git`` 命令并返回去尾空白的 stdout.

    参数:
        arguments: ``git`` 子命令及其参数.

    返回:
        output: 命令 stdout; 命令失败或 ``git`` 不可用时为 ``None``.
    """
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


def _memory_total_bytes() -> int | None:
    """读取本机物理内存总量.

    峰值内存的绝对值绑定本机上限, 记下来才能解释"为什么 47 GiB 是那条红线".

    返回:
        total: 物理内存字节数; 非 Linux 或读取失败时为 ``None``.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def file_digest(path: Path) -> dict[str, Any]:
    """给一个产物文件计算内容指纹与大小.

    参数:
        path: 产物路径.

    返回:
        record: 含 ``path`` (仓库根相对)、``sha256`` 与 ``bytes`` 的字典;
            文件不存在时 ``sha256`` 与 ``bytes`` 为 ``None``.
    """
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
    """采集本次落盘的运行溯源.

    返回:
        record: 溯源字典. ``git_dirty`` 为 ``True`` 表示工作区有未提交改动,
            此时这批数字**不可复现**, 只能作为开发证据;
            ``git_dirty`` 为 ``None`` 表示 ``git`` 不可用, 同样不可复现.
    """
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
        "numpy": _package_version("numpy"),
        "fealpy": _package_version("fealpy"),
        "memory_total_bytes": _memory_total_bytes(),
    }


def reproducible(record: dict[str, Any]) -> bool:
    """判断该溯源是否对应一次可复现的运行.

    参数:
        record: ``collect()`` 的返回值.

    返回:
        flag: 有 ``git_revision`` 且 ``git_dirty`` 明确为 ``False`` 时为 ``True``.
    """
    return bool(record.get("git_revision")) and record.get("git_dirty") is False
