# -*- coding: utf-8 -*-
"""``cases.toml`` 的加载与校验 (pa_assembly_capability).

本模块是 ``experiments/_common/casefile.py`` 的目录级薄封装: 提供本目录的路径常量与
``PANELS``, 并把 ``Case`` / ``ConfigError`` / ``select`` 原样再导出, 供 ``run.py`` / ``compare.py`` 使用.
"""

from __future__ import annotations

import sys
from pathlib import Path

# 本目录与仓库根。experiments/<name>/config.py 上溯两级即仓库根。
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

if str(EXPERIMENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR.parent))

from _common import casefile as _casefile  # noqa: E402
from _common.casefile import OUTPUT_MODES, Case, ConfigError, select  # noqa: E402,F401

# PA 数据点的分格: 积分点几何缓存 / 算子乘 / Jacobi-PCG 求解 / 连续测量, 另加单核硬件基线 (memcpy 带宽 + dgemm 算力).
PANELS = ("cache", "matvec", "solve", "baseline", "continuous")


def load_cases(path: Path | None = None) -> tuple[dict, tuple[Case, ...]]:
    """读取并校验本目录的 ``cases.toml``; 参数与返回见 ``_common.casefile.load_cases``."""
    return _casefile.load_cases(
        path or CASES_FILE,
        panels=PANELS,
        repository_root=REPOSITORY_ROOT,
        output_dir=OUTPUT_DIR,
    )


__all__ = [
    "EXPERIMENT_DIR", "REPOSITORY_ROOT", "CASES_FILE", "OUTPUT_DIR",
    "OUTPUT_MODES", "PANELS", "Case", "ConfigError", "load_cases", "select",
]


if __name__ == "__main__":
    _figure, _cases = load_cases()
    print(f"figure: {_figure.get('id')} | {len(_cases)} cases")
    for _c in _cases:
        print(f"  {_c.id:<16} panel={_c.panel:<7} n={_c.extra.get('n')} -> {_c.artifact_path.name}")
