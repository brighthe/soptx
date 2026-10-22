# -*- coding: utf-8 -*-
"""门式框架的非结构化三角形网格与两处局部放大 (论文图 5.11(b)).

产物 case ``portal-mesh``: 网格取自 Hu--Zhang k=3 运行的 density_final.vtu (与优化同一张网格,
由 portal_frame_mesh.build_mesh 在左半域剖分后镜像得到). 三栏: (a) 全域网格, 方框标出放大区;
(b) 凹角顶点附近, 网格贴合两条斜边; (c) 顶边载荷区间附近, 区间端点 (红点) 落在网格节点上.
示意图 (几何尺寸、载荷与支座, 图 5.11(a)) 是论文侧 TikZ, 不在本目录.

网格线是矢量, 输出 PDF 与 600 dpi PNG.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

import config
from ._base import load_density, paper_rcparams, save_figure

paper_rcparams()

FIG_SIZE_IN = (5.9, 1.75)
TITLE_PT = 9.0
DPI = 600
EDGE_COLOR = "#333333"
BOX_COLOR = "#d62728"

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "portal-frame-2d-stress"
REQUIRED_RUNS = ("analyzer-huzhang__order-3",)


def main() -> None:
    run_dir = config.OUTPUT_DIR / SOURCE_CASE / REQUIRED_RUNS[0]
    points, cells, _ = load_density(run_dir / "density_final.vtu")
    model = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["model"]
    x_mid, h_c, height = model["L"] / 2.0, model["corner_height"], model["H"]
    half_load = model["load_width"] / 2.0
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)

    # 放大区: 凹角顶点与顶边载荷区间, 均以 x = L/2 为中心
    windows = (
        ("(b) Re-entrant corner", (x_mid - 0.8, x_mid + 0.8), (h_c - 0.45, h_c + 0.55)),
        ("(c) Loaded segment", (x_mid - 0.8, x_mid + 0.8), (height - 0.6, height + 0.05)),
    )

    fig = plt.figure(figsize=FIG_SIZE_IN, layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 1.0])
    ax_full = fig.add_subplot(gs[0, 0])
    ax_full.triplot(triangulation, color=EDGE_COLOR, lw=0.08)
    for _, (x0, x1), (y0, y1) in windows:
        ax_full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor=BOX_COLOR, lw=0.8))
    ax_full.set_title("(a) Full mesh", fontsize=TITLE_PT, pad=3)

    for column, (title, (x0, x1), (y0, y1)) in enumerate(windows, start=1):
        axes = fig.add_subplot(gs[0, column])
        axes.triplot(triangulation, color=EDGE_COLOR, lw=0.35)
        axes.set_xlim(x0, x1)
        axes.set_ylim(y0, y1)
        axes.set_title(title, fontsize=TITLE_PT, pad=3)
        if column == 2:
            axes.plot([x_mid - half_load, x_mid + half_load], [height, height], "o",
                      color=BOX_COLOR, ms=2.5)
    for axes in fig.axes:
        axes.set_aspect("equal")
        axes.axis("off")

    save_figure(fig, "portal_mesh", formats=("pdf", "png"), dpi=DPI)


if __name__ == "__main__":
    main()
