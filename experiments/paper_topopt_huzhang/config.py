"""Hu--Zhang 拓扑优化论文实验的路径常量与源码路径注入.

算例参数写在各 run 脚本 (``run_fixed_fixed.py`` / ``run_bearing.py`` /
``run_cantilever_stress.py``) 顶部的常量里; 本模块只提供共用的目录常量.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# --- 目录常量 --------------------------------------------------------------
# 实验内所有脚本的路径都由此派生, 不再各自硬编码 /home/... 或 /mnt/c/... 绝对路径.
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
# 运行产物、再分析结果与成图统一写入入库的论文证据目录 results/; 逐步帧 vtu/ 由
# .gitignore 排除, 其余入库.
OUTPUT_DIR = EXPERIMENT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
# ParaView 查看副本 (Windows 本地盘): 只供查看的大文件写到这里, 目录结构与 results/ 一一对应;
# 各 run 脚本另有同值常量 VIEW_ROOT. 该盘不可用时调用方退回 results/ (由 .gitignore 排除)
VIEW_DIR = Path("/mnt/c/workspace/soptx-results/paper_topopt_huzhang")

# 论文插图目录属于另一个仓库 (dut-postdoc), 默认取 WSL 下的挂载路径; 可用环境变量
# HUZHANG_PAPER_FIGDIR 覆盖, 目录不存在时同步步骤自动跳过而非报错.
PAPER_FIGURE_DIR = Path(
    os.environ.get("HUZHANG_PAPER_FIGDIR", "/mnt/c/workspace/dut-postdoc/papers/huzhang-topopt/figures")
)


def bootstrap_source_path() -> None:
    """把 soptx 源码目录与本实验目录加入 ``sys.path``, 供脚本按文件路径直接运行."""
    for path in (SOURCE_DIR, EXPERIMENT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
