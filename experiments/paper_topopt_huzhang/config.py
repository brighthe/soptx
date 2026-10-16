"""Hu--Zhang 拓扑优化论文实验共用的路径常量.

算例参数写在各 run 脚本 (``run_fixed_fixed.py`` / ``run_bearing.py`` /
``run_cantilever_stress.py``) 顶部的常量里; 本模块只提供共用的目录常量.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- 目录常量 --------------------------------------------------------------
# 实验内所有脚本的路径都由此派生, 不再各自硬编码 /home/... 或 /mnt/c/... 绝对路径.
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
# 入库的论文证据目录 results/: run 脚本的运行产物先写 outputs/, 确认后经 sync_results.py 固化至此;
# 再分析结果与成图直接写入.
OUTPUT_DIR = EXPERIMENT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
# ParaView 查看副本 (Windows 本地盘): run 脚本的逐步帧留在运行目录, 需要时手动迁移到这里, 目录结构
# 与 outputs/ 一一对应; 再分析的大文件直接写这里, 该盘不可用时调用方退回 results/ (由 .gitignore 排除)
VIEW_DIR = Path("/mnt/c/workspace/soptx-results/paper_topopt_huzhang")

# 论文插图目录属于另一个仓库 (dut-postdoc), 默认取 WSL 下的挂载路径; 可用环境变量
# HUZHANG_PAPER_FIGDIR 覆盖, 目录不存在时同步步骤自动跳过而非报错.
PAPER_FIGURE_DIR = Path(
    os.environ.get("HUZHANG_PAPER_FIGDIR", "/mnt/c/workspace/dut-postdoc/papers/huzhang-topopt/figures")
)
