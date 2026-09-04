# -*- coding: utf-8 -*-
"""悬臂梁应力约束高阶 k=3/4 拓扑与应力分布 (论文图 5.9).

产物 case ``stress-highorder-topologies``: 依赖见 REQUIRED_RUNS —— 不是运行目录而是
postprocess/ 下的 npz, 由 ``compare.py export`` 从 density_final.vtu 冻结求解导出。

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
    "postprocess/fig_data_huzhang-k3.npz",
    "postprocess/fig_data_huzhang-k4.npz",
)


def main() -> None:
    # 1. 加载 SOPTX 计算数据
    data_dir = config.OUTPUT_DIR / SOURCE_CASE / "postprocess"
    k3_data = np.load(data_dir / "fig_data_huzhang-k3.npz")
    k4_data = np.load(data_dir / "fig_data_huzhang-k4.npz")

    node = k3_data["node"]
    cell = k3_data["cell"]
    triang = Triangulation(node[:, 0], node[:, 1], cell)

    k3_rho = k3_data["rho"]
    k3_vm = k3_data["vm"]
    k3_vol = float(k3_data["vol"])

    k4_rho = k4_data["rho"]
    k4_vm = k4_data["vm"]
    k4_vol = float(k4_data["vol"])

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

    # (a) k=3 拓扑
    ax_a.tripcolor(triang, facecolors=1.0 - k3_rho, cmap="gray", vmin=0, vmax=1, edgecolors="none")
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.set_title(f"(a) 胡张混合法 ($k = 3$) 最终拓扑构型 ($V^* = {k3_vol * 100:.2f}\\%$) ", fontsize=11, fontproperties=ZH, pad=8)

    # (b) k=3 应力
    ax_b.tripcolor(triang, facecolors=k3_vm, cmap="jet", vmin=0, vmax=1, edgecolors="none")
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.set_title("(b) 胡张混合法 ($k = 3$) 归一化 von Mises 应力", fontsize=11, fontproperties=ZH, pad=8)

    # (c) k=4 拓扑
    ax_c.tripcolor(triang, facecolors=1.0 - k4_rho, cmap="gray", vmin=0, vmax=1, edgecolors="none")
    ax_c.set_aspect("equal")
    ax_c.axis("off")
    ax_c.set_title(f"(c) 胡张混合法 ($k = 4$) 最终拓扑构型 ($V^* = {k4_vol * 100:.2f}\\%$) ", fontsize=11, fontproperties=ZH, pad=8)

    # (d) k=4 应力
    ax_d.tripcolor(triang, facecolors=k4_vm, cmap="jet", vmin=0, vmax=1, edgecolors="none")
    ax_d.set_aspect("equal")
    ax_d.axis("off")
    ax_d.set_title("(d) 胡张混合法 ($k = 4$) 归一化 von Mises 应力", fontsize=11, fontproperties=ZH, pad=8)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cb1 = ColorbarBase(ax_cb1, cmap=cm.jet, norm=norm, orientation="vertical")
    cb1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb1.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontproperties=ZH, fontsize=9)

    cb2 = ColorbarBase(ax_cb2, cmap=cm.jet, norm=norm, orientation="vertical")
    cb2.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb2.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontproperties=ZH, fontsize=9)

    # 4. 输出 png/pdf/eps 三种格式
    save_figure(fig, "stress_highorder_topologies", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
