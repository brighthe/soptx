# -*- coding: utf-8 -*-
"""任意阶胡张离散 (k=2, 4) 下悬臂梁应力约束拓扑与应力分布 (论文图 5.10).

产物 case ``stress-hz-orders-topologies``: 与图 5.8 (stress_cubic_topologies) 同版式、
同数据口径, 只是把两行换成 k=2 (跳量稳定化格式, k <= d) 与 k=4 (原生格式), 与图 5.8
的 k=3 合起来覆盖稳定化低阶到原生高阶. 数据来自 postprocess/discretization_probe/ 下
两份 fields.npz, 由 ``plot.py discretization-probe --design <运行目录>`` 产出; 密度与
表观 von Mises 应力比取各自离散的自读数, 被动实体区照常着色, 体积分数含该区.

与 stress_cubic_topologies 同理只出 600 dpi 的 PNG (至 papers/huzhang-topopt/figures 与本地
results/figures), 尺寸、字号与字体常量也从该模块取, 两图版式保持一致.

2026-09-28 起按版心尺寸出图: 图宽取 CICP 版心 150 mm (5.9 in), 论文里以 ``width=\\textwidth``
原尺寸嵌入, 面板标题字号即纸面字号; 字体走 ``paper_rcparams`` 的 Palatino 口径, 标题与轴名
改英文 (投稿稿用). 面板标题只写编号、方法、阶次与物理量, 体积分数等数值由正文给出.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

import config
from ._base import paper_rcparams, save_figure
from .stress_cubic_topologies import DPI, FIG_SIZE_IN, TITLE_PT, load_design

paper_rcparams()

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "postprocess/discretization_probe/huzhang-2-pad-solid__fields.npz",
    "postprocess/discretization_probe/huzhang-4-pad-solid__fields.npz",
)


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE

    k2_tri, k2_rho, k2_vm, k2_vol = load_design(base, "huzhang-2-pad-solid", "huzhang-2")
    k4_tri, k4_rho, k4_vm, k4_vol = load_design(base, "huzhang-4-pad-solid", "huzhang-4")

    fig = plt.figure(figsize=FIG_SIZE_IN, layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06, hspace=0.12)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_cb1 = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_cb2 = fig.add_subplot(gs[1, 2])

    # 体积分数由正文给出, 标题不写; k2_vol / k4_vol 仍由 load_design 返回供核对.
    panels = (
        (ax_a, k2_tri, 1.0 - k2_rho, "gray", "(a) HZMFEM, $k = 2$: final topology"),
        (ax_b, k2_tri, k2_vm, "jet", "(b) HZMFEM, $k = 2$: apparent von Mises stress ratio"),
        (ax_c, k4_tri, 1.0 - k4_rho, "gray", "(c) HZMFEM, $k = 4$: final topology"),
        (ax_d, k4_tri, k4_vm, "jet", "(d) HZMFEM, $k = 4$: apparent von Mises stress ratio"),
    )
    for axes, triangulation, values, cmap, title in panels:
        axes.tripcolor(triangulation, facecolors=values, cmap=cmap,
                       vmin=0, vmax=1, edgecolors="none")
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_title(title, fontsize=TITLE_PT, pad=3)

    norm = Normalize(vmin=0.0, vmax=1.0)
    for axes in (ax_cb1, ax_cb2):
        colorbar = ColorbarBase(axes, cmap=cm.jet, norm=norm, orientation="vertical")
        colorbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        colorbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    save_figure(fig, "stress_hz_orders_topologies", dpi=DPI)


if __name__ == "__main__":
    main()
