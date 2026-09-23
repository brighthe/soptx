# -*- coding: utf-8 -*-
"""三次离散下悬臂梁应力约束拓扑与应力分布 (论文图 5.8).

产物 case ``stress-cubic-topologies``: 依赖 postprocess/discretization_probe/ 下的
两份 fields.npz, 由 ``compare.py discretization-probe`` 产出 (不是 ``compare.py
export``, 故 REQUIRED_RUNS 缺失时 run_case 打印的 run.py 命令不适用, 正确的补数据
命令见本文件末尾注释)。

与图 5.7 的区别: 图 5.7 是 k=2, 本图是 k=3, 对应 §5.2.3 的主对比; 两者同为 pad 1.5 mm,
本图的运行另按判据集合 (rho >= 0.5, 区外) 验收. 密度与表观 von Mises 应力比都取各自
离散的自读数, 即 huzhang-3 构型读 huzhang-3、lfem-3 构型读 lfem-3, 与优化过程中约束
所用的读数同口径; 被动实体区照常着色 (其密度为 1, 应力比为自读值), 体积分数含该区.

仅输出 PNG 至 papers/huzhang-topopt/figures 与本地 outputs/figures.
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
from ._base import chinese_font, load_density, save_figure

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
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
    # 网格拓扑只存在 vtu 里, npz 只有单元量; 两者的单元序由探针同一次导出保证一致,
    # 这里用密度逐单元核对一次, 不一致即报错而非画出错位的图.
    points, cells, density_vtu = load_density(directory / f"{tag}__{label}.vtu")
    rho = fields["density"]
    if not np.allclose(rho, density_vtu):
        raise RuntimeError(f"{tag}: npz 与 vtu 的密度不一致, 单元序可能错位.")

    triangulation = Triangulation(points[:, 0], points[:, 1], cells)
    return triangulation, rho, fields[f"vm_app__{label}"], float(rho.mean())


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE

    lf_tri, lf_rho, lf_vm, lf_vol = load_design(base, "lfem-3-pad-solid", "lfem-3")
    hz_tri, hz_rho, hz_vm, hz_vol = load_design(base, "huzhang-3-pad-solid", "huzhang-3")

    ZH = chinese_font()

    # 版式与图 5.7 一致: 两行 (位移法 / 混合法) x 两栏 (构型 / 应力) + 逐行色标
    fig = plt.figure(figsize=(12.5, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.15, hspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_cb1 = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_cb2 = fig.add_subplot(gs[1, 2])

    panels = (
        (ax_a, lf_tri, 1.0 - lf_rho, "gray",
         f"(a) 标准位移法 ($p = 3$) 最终拓扑构型 ($f_V = {lf_vol * 100:.2f}\\%$) "),
        (ax_b, lf_tri, lf_vm, "jet",
         "(b) 标准位移法 ($p = 3$) 表观 von Mises 应力比"),
        (ax_c, hz_tri, 1.0 - hz_rho, "gray",
         f"(c) 胡张混合法 ($k = 3$) 最终拓扑构型 ($f_V = {hz_vol * 100:.2f}\\%$) "),
        (ax_d, hz_tri, hz_vm, "jet",
         "(d) 胡张混合法 ($k = 3$) 表观 von Mises 应力比"),
    )
    for axes, triangulation, values, cmap, title in panels:
        axes.tripcolor(triangulation, facecolors=values, cmap=cmap,
                       vmin=0, vmax=1, edgecolors="none")
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_title(title, fontsize=11, fontproperties=ZH, pad=8)

    norm = Normalize(vmin=0.0, vmax=1.0)
    for axes in (ax_cb1, ax_cb2):
        colorbar = ColorbarBase(axes, cmap=cm.jet, norm=norm, orientation="vertical")
        colorbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        colorbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                fontproperties=ZH, fontsize=9)

    save_figure(fig, "stress_cubic_topologies")


if __name__ == "__main__":
    main()
