# -*- coding: utf-8 -*-
"""悬臂梁应力约束拓扑与应力分布对比 (论文图 5.7).

产物 case ``stress-topologies``: 依赖见 REQUIRED_RUNS —— 不是运行目录而是 postprocess/
下的 npz, 由 ``compare.py export`` 从 density_final.vtu 冻结求解导出。

输出 png/pdf/eps 三种格式至 papers/figures 与本地 outputs/figures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

import config
from ._base import chinese_font, save_figure

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "postprocess/fig_data_lfem-k2.npz",
    "postprocess/fig_data_huzhang-k2.npz",
)


def main() -> None:
    # 1. 加载 SOPTX 计算数据
    data_dir = config.OUTPUT_DIR / SOURCE_CASE / "postprocess"
    lf_data = np.load(data_dir / "fig_data_lfem-k2.npz")
    hz_data = np.load(data_dir / "fig_data_huzhang-k2.npz")

    node = lf_data["node"]
    cell = lf_data["cell"]
    triang = Triangulation(node[:, 0], node[:, 1], cell)

    lf_rho = lf_data["rho"]
    lf_vm = lf_data["vm"]
    lf_vol = float(lf_data["vol"])

    hz_rho = hz_data["rho"]
    hz_vm = hz_data["vm"]
    hz_vol = float(hz_data["vol"])

    # 2. 中文字体设置
    ZH = chinese_font()

    # 3. 绘制 2x2 四宫格
    fig = plt.figure(figsize=(12.5, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.15, hspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_cb1 = fig.add_subplot(gs[0, 2])

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_cb2 = fig.add_subplot(gs[1, 2])

    # (a) LFEM 拓扑
    ax_a.tripcolor(triang, facecolors=1.0 - lf_rho, cmap="gray", vmin=0, vmax=1, edgecolors="none")
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.set_title(f"(a) 标准位移法 ($k = 2$) 最终拓扑构型 ($V^* = {lf_vol * 100:.2f}\\%$) ", fontsize=11, fontproperties=ZH, pad=8)

    # (b) LFEM 应力
    im_b = ax_b.tripcolor(triang, facecolors=lf_vm, cmap="jet", vmin=0, vmax=1, edgecolors="none")
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.set_title("(b) 标准位移法 ($k = 2$) 归一化 von Mises 应力", fontsize=11, fontproperties=ZH, pad=8)

    # (c) HZMFEM 拓扑
    ax_c.tripcolor(triang, facecolors=1.0 - hz_rho, cmap="gray", vmin=0, vmax=1, edgecolors="none")
    ax_c.set_aspect("equal")
    ax_c.axis("off")
    ax_c.set_title(f"(c) 胡张混合法 ($k = 2$) 最终拓扑构型 ($V^* = {hz_vol * 100:.2f}\\%$) ", fontsize=11, fontproperties=ZH, pad=8)

    # (d) HZMFEM 应力
    im_d = ax_d.tripcolor(triang, facecolors=hz_vm, cmap="jet", vmin=0, vmax=1, edgecolors="none")
    ax_d.set_aspect("equal")
    ax_d.axis("off")
    ax_d.set_title("(d) 胡张混合法 ($k = 2$) 归一化 von Mises 应力", fontsize=11, fontproperties=ZH, pad=8)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cb1 = ColorbarBase(ax_cb1, cmap=cm.jet, norm=norm, orientation="vertical")
    cb1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb1.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontproperties=ZH, fontsize=9)

    cb2 = ColorbarBase(ax_cb2, cmap=cm.jet, norm=norm, orientation="vertical")
    cb2.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb2.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontproperties=ZH, fontsize=9)

    # 4. 输出 png/pdf/eps 三种格式
    save_figure(fig, "stress_topologies", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
