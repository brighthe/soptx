# -*- coding: utf-8 -*-
"""三次离散下悬臂梁应力约束拓扑与应力分布 (论文图 5.8).

产物 case ``stress-cubic-topologies``: 依赖 postprocess/discretization_probe/ 下的
两份 fields.npz, 由 ``plot.py discretization-probe`` 产出 (不是 ``plot.py
export``, 故 REQUIRED_RUNS 缺失时 run_case 打印的 run.py 命令不适用, 正确的补数据
命令见本文件末尾注释).

与图 5.7 的区别: 图 5.7 是 k=2, 本图是 k=3, 对应 §5.2.3 的主对比; 两者同为 pad 1.5 mm,
本图的运行另按判据集合 (rho >= 0.5, 区外) 验收. 密度与表观 von Mises 应力比都取各自
离散的自读数, 即 huzhang-3 构型读 huzhang-3、lfem-3 构型读 lfem-3, 与优化过程中约束
所用的读数同口径; 被动实体区照常着色 (其密度为 1, 应力比为自读值), 体积分数含该区.

仅输出 PNG 至 papers/huzhang-topopt/figures 与本地 results/figures: 密度与应力比都是
分片常数场, 矢量输出会显出三角形接缝, 故只出 600 dpi 的 PNG.

2026-09-28 起按版心尺寸出图: 图宽取 CICP 版心 150 mm (5.9 in), 论文里以 ``width=\\textwidth``
原尺寸嵌入, 面板标题字号即纸面字号; 字体走 ``paper_rcparams`` 的 Palatino 口径, 标题与轴名
改英文 (投稿稿用). 面板标题只写编号、方法、阶次与物理量, 体积分数等数值由正文给出.
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
from ._base import load_probe_mesh, paper_rcparams, save_figure

paper_rcparams()

# 版心宽 150 mm; 高度 = 2 行 80x40 面板 (每行约 1.3 in) + 2 行 9 pt 标题 + 行距.
FIG_SIZE_IN = (5.9, 3.15)
TITLE_PT = 9.0
DPI = 600

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "postprocess/discretization_probe/lfem-3-pad-solid__fields.npz",
    "postprocess/discretization_probe/huzhang-3-pad-solid__fields.npz",
)

PROBE_DIR = "postprocess/discretization_probe"


def load_design(base, tag: str, label: str):
    """读一个冻结构型的网格、密度与自读表观应力比.

    Parameters
    ----------
    base : Path
        算例产物根目录.
    tag : str
        探针产物前缀, 如 ``lfem-3-nopad``.
    label : str
        自读离散标签, 如 ``lfem-3``; 决定取哪一列 ``vm_app__``.

    Returns
    -------
    tuple
        ``(triangulation, rho, vm, volume_fraction)``.
    """
    directory = base / PROBE_DIR
    fields = np.load(directory / f"{tag}__fields.npz")
    # npz 只有单元量, 网格拓扑取自被冻结运行的 density_final.vtu; 单元序由探针读同一
    # 文件保证一致, 这里用密度逐单元核对一次, 不一致即报错而非画出错位的图.
    points, cells, density_vtu = load_probe_mesh(directory, tag)
    rho = fields["density"]
    if not np.allclose(rho, density_vtu):
        raise RuntimeError(f"{tag}: npz 与 vtu 的密度不一致, 单元序可能错位.")

    triangulation = Triangulation(points[:, 0], points[:, 1], cells)
    return triangulation, rho, fields[f"vm_app__{label}"], float(rho.mean())


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE

    lf_tri, lf_rho, lf_vm, lf_vol = load_design(base, "lfem-3-pad-solid", "lfem-3")
    hz_tri, hz_rho, hz_vm, hz_vol = load_design(base, "huzhang-3-pad-solid", "huzhang-3")

    # 版式与图 5.7 一致: 两行 (位移法 / 混合法) x 两栏 (构型 / 应力) + 逐行色标
    fig = plt.figure(figsize=FIG_SIZE_IN, layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06, hspace=0.12)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_cb1 = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_cb2 = fig.add_subplot(gs[1, 2])

    # 体积分数由正文给出, 标题不写; lf_vol / hz_vol 仍由 load_design 返回供核对.
    panels = (
        (ax_a, lf_tri, 1.0 - lf_rho, "gray", "(a) LFEM, $p = 3$: final topology"),
        (ax_b, lf_tri, lf_vm, "jet", "(b) LFEM, $p = 3$: apparent von Mises stress ratio"),
        (ax_c, hz_tri, 1.0 - hz_rho, "gray", "(c) HZMFEM, $k = 3$: final topology"),
        (ax_d, hz_tri, hz_vm, "jet", "(d) HZMFEM, $k = 3$: apparent von Mises stress ratio"),
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

    save_figure(fig, "stress_cubic_topologies", dpi=DPI)


if __name__ == "__main__":
    main()
