# -*- coding: utf-8 -*-
"""三次胡张离散下门式框架应力约束拓扑与应力分布 (论文图 5.12).

产物 case ``stress-portal-topology``: 依赖 postprocess/discretization_probe/ 下的
huzhang-3 fields.npz, 由 ``plot.py discretization-probe --case portal-frame-2d-stress``
产出; 缺失时 plot.py 会提示运行该探针. 密度与表观 von Mises 应力比取 Hu--Zhang k=3 的
自读数, 与优化中约束所用的读数同口径; 本算例无被动实体区, 全部单元受约束.

版式沿用图 5.8 (stress_cubic_topologies): 一行两栏 (构型 / 应力比) + 色标, 尺寸、字号与
字体常量从该模块取; 两栏都描出设计域轮廓 (含底部三角缺口), 使空洞区与域外可分.
只出 600 dpi 的 PNG (密度与应力比都是分片常数场, 矢量输出会显出三角形接缝).
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

import config
from ._base import paper_rcparams, save_figure
from .stress_cubic_topologies import DPI, TITLE_PT, PROBE_DIR, load_design

paper_rcparams()

# 版心宽 150 mm; 一行两栏 12 x 6 面板 (每栏约 2.75 x 1.4 in) + 9 pt 标题.
FIG_SIZE_IN = (5.9, 1.7)
OUTLINE_COLOR = "#9a9a9a"

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "portal-frame-2d-stress"
REQUIRED_RUNS = (
    "postprocess/discretization_probe/huzhang-3__fields.npz",
)


def domain_outline(base, tag: str) -> np.ndarray:
    """由冻结运行的 config.json 复原设计域七边形顶点 (闭合).

    Parameters
    ----------
    base : Path
        算例产物根目录.
    tag : str
        探针产物前缀, 如 ``huzhang-3``.

    Returns
    -------
    ndarray, shape (8, 2)
        逆时针顶点, 首尾相同.
    """
    probe = json.loads((base / PROBE_DIR / f"{tag}__probe.json").read_text(encoding="utf-8"))
    model = json.loads((config.OUTPUT_DIR / probe["design_run_dir"] / "config.json")
                       .read_text(encoding="utf-8"))["model"]
    length, height = model["L"], model["H"]
    b, h_c = model["support_width"], model["corner_height"]
    vertices = [(0.0, 0.0), (b, 0.0), (length / 2.0, h_c), (length - b, 0.0),
                (length, 0.0), (length, height), (0.0, height), (0.0, 0.0)]
    return np.asarray(vertices)


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE
    triangulation, rho, vm, _ = load_design(base, "huzhang-3", "huzhang-3")
    outline = domain_outline(base, "huzhang-3")

    fig = plt.figure(figsize=FIG_SIZE_IN, layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    # 体积分数由正文给出, 标题不写.
    panels = (
        (ax_a, 1.0 - rho, "gray", "(a) Physical density"),
        (ax_b, vm, "jet", "(b) Apparent stress ratio"),
    )
    for axes, values, cmap, title in panels:
        axes.tripcolor(triangulation, facecolors=values, cmap=cmap,
                       vmin=0, vmax=1, edgecolors="none")
        axes.plot(outline[:, 0], outline[:, 1], color=OUTLINE_COLOR, lw=0.5)
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_title(title, fontsize=TITLE_PT, pad=3)

    colorbar = ColorbarBase(ax_cb, cmap=cm.jet, norm=Normalize(vmin=0.0, vmax=1.0),
                            orientation="vertical")
    colorbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colorbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    save_figure(fig, "stress_portal_hz_k3_topology", dpi=DPI)


if __name__ == "__main__":
    main()
