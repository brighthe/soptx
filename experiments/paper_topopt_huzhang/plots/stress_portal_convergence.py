# -*- coding: utf-8 -*-
"""三次胡张离散下门式框架应力约束优化的收敛历史与主应力分布 (论文图 5.13).

产物 case ``stress-portal-convergence``: 与图 5.9 (stress_cubic_convergence) 同版式的一行:
- (a) 收敛历史: 只读运行目录下的 history.json 与 summary.json. 右轴画全部受约束单元上的
  最大约束值 ``max_relative_violation`` (本算例无被动实体区, 即全部单元), 与论文 4.6.2 节的
  判据集合同口径; 最新运行按全部单元验收 (portal_frame.md).
- (b) 主应力空间内的单元应力分布: 取 postprocess/stress_fields/fig_data_portal-huzhang-k3.npz,
  由 ``plot.py export --run portal-huzhang-k3`` 冻结重分析导出; 散点画全部受约束单元的
  归一化表观主应力, 与图 5.9 及 PolyStress 图 8 同口径.

绘图面板函数取自 stress_cubic_convergence, 尺寸、配色与字号常量同图 5.9. 输出 PDF 矢量与
600 dpi PNG.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from ._base import paper_rcparams, require_run_dir, save_figure
from .stress_cubic_convergence import DPI, load_fields, plot_convergence, plot_yield_surface

paper_rcparams()

# 版心宽 150 mm; 一行: 左收敛历史 (双纵轴), 右主应力平面 (正方形 + 色标).
FIG_SIZE_IN = (5.9, 2.55)
VOLUME_YLIM = (0.20, 0.55)

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "portal-frame-2d-stress"
REQUIRED_RUNS = (
    "analyzer-huzhang__order-3__p1-interpolation__all-cells__lambda-max-10000.0",
    "postprocess/stress_fields/fig_data_portal-huzhang-k3.npz",
)


def load_history(base, folder: str):
    """读体积分数、全部受约束单元上的最大约束值与应力容差.

    Parameters
    ----------
    base : Path
        算例产物根目录.
    folder : str
        运行目录名.

    Returns
    -------
    tuple
        ``(volfrac, g_max, delta_g)``, 前两者为逐 MMA 更新的一维数组.

    Raises
    ------
    RuntimeError
        运行设有被动实体区: 此时 ``max_relative_violation`` 不等于全部受约束单元口径.
    """
    run_dir = require_run_dir(base, folder)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("pad_cells"):
        raise RuntimeError(f"{folder}: 设有被动实体区, 本图只用于无被动区的运行.")
    scalars = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))["scalar_histories"]
    return (
        np.asarray(scalars["volfrac"], dtype=float),
        np.asarray(scalars["max_relative_violation"], dtype=float),
        float(summary["relative_stress_tolerance"]),
    )


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE
    volfrac, g_max, tolerance = load_history(base, REQUIRED_RUNS[0])
    fields = load_fields(base, REQUIRED_RUNS[1])

    # MMA 更新次数由正文给出, 标题不写.
    fig = plt.figure(figsize=FIG_SIZE_IN)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.55,
                          left=0.10, right=0.93, top=0.90, bottom=0.17)
    history_axes = fig.add_subplot(gs[0, 0])
    plot_convergence(history_axes, volfrac, g_max, tolerance,
                     "(a) Optimization history", volume_ylim=VOLUME_YLIM)
    history_axes.set_xlabel("MMA updates")
    plot_yield_surface(fig.add_subplot(gs[0, 1]), fields,
                       "(b) Principal stress distribution", compact_apparent=True)

    save_figure(fig, "stress_portal_hz_k3_convergence", formats=("pdf", "png"), dpi=DPI)


if __name__ == "__main__":
    main()
