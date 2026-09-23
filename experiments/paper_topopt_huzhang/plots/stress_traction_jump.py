# -*- coding: utf-8 -*-
"""冻结构型实体带上逐单元绝对牵引跳量 A_e 的空间分布与经验累积分布 (论文图 5.11).

产物 case ``stress-traction-jump``: 依赖 postprocess/discretization_probe/ 下 LFEM p=2..4 与
HZMFEM k=2..4 六份 fields.npz, 由 ``compare.py discretization-probe`` 产出, 不重解方程。

- (a) LFEM p=2 构型上实体带 (physical density > SOLID_THRESHOLD 且不在被动实体区)
  逐单元 ``A_e = max_{F in dT_e, F 为内边} ||[[sigma n]]||_rms(F) / sigma_bar``
  (npz 的 ``absjump__lfem-2``) 的空间分布, 底图为灰度构型, 超出色标上界者截断.
- (b) HZMFEM k=2 构型上同一量, 与 (a) 共用色标; H(div) 协调使其整幅为零.
- (c) 六个构型各自在优化所用离散下 A_e 的经验累积分布, 对数横轴断开: 左段 1e-18..1e-15
  为 HZMFEM k=2..4 (虚线), 右段 1e-5..1 为 LFEM p=2..4 (实线), 同阶同色; 竖线为 delta_g
  与 4 delta_g.

每个构型的跳量与余量都取其优化所用离散的自读数: 跳量是该离散读数的不唯一性, 余量是该
离散读数距约束边界的距离, 二者才可比. 实体带内 m_E ~ 1 故 eta ~ 1, ``-g_e`` 即
1 - sigma_vm / sigma_bar.

输出 png/pdf/eps 三种格式至 papers/huzhang-topopt/figures 与本地 outputs/figures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

import config
from ._base import academic_rcparams, chinese_font, load_density, save_figure

academic_rcparams()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "postprocess/discretization_probe/lfem-2-pad-solid__fields.npz",
    "postprocess/discretization_probe/lfem-3-pad-solid__fields.npz",
    "postprocess/discretization_probe/lfem-4-pad-solid__fields.npz",
    "postprocess/discretization_probe/huzhang-2-pad-solid__fields.npz",
    "postprocess/discretization_probe/huzhang-3-pad-solid__fields.npz",
    "postprocess/discretization_probe/huzhang-4-pad-solid__fields.npz",
)

PROBE_DIR = "postprocess/discretization_probe"
SOLID_THRESHOLD = 0.9
# 停止准则容差, 与 discretization_probe.DELTA_G / cases.toml 的 stress_tolerance 同值.
DELTA_G = 5.0e-3
ORDERS = (2, 3, 4)
ORDER_COLORS = ("#5a5a5a", "#1f77b4", "#ff7f0e")
COLOR_DELTA = "#d62728"
MAP_LABELS = ("lfem-2", "huzhang-2")
MAP_VMAX = 0.1
CDF_XLIM_HZ = (1e-18, 1e-15)
CDF_XLIM_LFEM = (1e-5, 1e0)
# 断口两侧不标端点刻度, 免得 1e-15 与 1e-5 的标签相撞.
CDF_XTICKS_HZ = (1e-18, 1e-17, 1e-16)
CDF_XTICKS_LFEM = (1e-4, 1e-3, 1e-2, 1e-1, 1e0)
LEGEND_STYLE = dict(fontsize=9, framealpha=0.95, edgecolor="#cccccc")


def load_design(base, label: str):
    """读标签为 label 的构型的密度、实体带掩码、自读 A_e 与余量 -g_e; 地图面板另读网格."""
    tag = f"{label}-pad-solid"
    directory = base / PROBE_DIR
    fields = np.load(directory / f"{tag}__fields.npz")
    key = f"absjump__{label}"
    if key not in fields.files:
        raise RuntimeError(
            f"{tag}: npz 缺绝对跳量 {key}, 是旧口径探针; 请重新运行 compare.py discretization-probe."
        )
    rho = fields["density"]
    design = {
        "label": label,
        "rho": rho,
        "solid": (rho > SOLID_THRESHOLD) & ~fields["pad_mask"].astype(bool),
        "jump": fields[key],
        "headroom": -fields[f"g__{label}"],
    }
    if label in MAP_LABELS:
        points, cells, density_vtu = load_density(directory / f"{tag}__{label}.vtu")
        if not np.allclose(rho, density_vtu):
            raise RuntimeError(f"{tag}: npz 与 vtu 的密度不一致, 单元序可能错位.")
        design["triangulation"] = Triangulation(points[:, 0], points[:, 1], cells)
    return design


def plot_map(fig, axes, design, title, font):
    """灰度构型底图 + 实体带上 A_e 着色; 色标由 add_shared_colorbar 另放."""
    tri = design["triangulation"]
    axes.tripcolor(tri, facecolors=1.0 - design["rho"], cmap="gray",
                   vmin=0, vmax=1, edgecolors="none")
    solid = design["solid"]
    values = np.clip(design["jump"], 0.0, MAP_VMAX)
    masked = Triangulation(tri.x, tri.y, tri.triangles[solid])
    axes.tripcolor(masked, facecolors=values[solid], cmap="viridis",
                   vmin=0.0, vmax=MAP_VMAX, edgecolors="none")
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title(title, fontsize=11, fontproperties=font, pad=8)


def add_shared_colorbar(fig, ax_top, ax_bottom, font):
    """两张地图共用一根色标: 纵向覆盖上图地图顶到下图地图底 (地图纵横比 2:1)."""
    fig.canvas.draw()
    width_in, height_in = fig.get_size_inches()

    def map_span(axes):
        box = axes.get_position()
        h = box.width * width_in / 2.0 / height_in
        y0 = box.y0 + (box.height - h) / 2.0
        return box.x1, y0, y0 + h

    x1, _, y_top = map_span(ax_top)
    _, y_bot, _ = map_span(ax_bottom)
    cax = fig.add_axes([x1 + 0.006, y_bot, 0.011, y_top - y_bot])
    colorbar = ColorbarBase(cax, cmap=cm.viridis, norm=Normalize(0.0, MAP_VMAX),
                            orientation="vertical")
    ticks = [0.0, 0.025, 0.05, 0.075, 0.10]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(["0", "0.025", "0.05", "0.075", r"$\geq 0.10$"],
                            fontproperties=font, fontsize=9)
    cax.set_title(r"$A_e$", fontsize=11, pad=6)


def ecdf(values):
    order = np.sort(values)
    return order, np.arange(1, order.size + 1) / order.size


def plot_cdf(ax_hz, ax_lfem, lfem_designs, hz_designs, font):
    """(c) 断轴经验累积分布: 左段 HZMFEM k=2..4 (虚线), 右段 LFEM p=2..4 (实线), 共用纵轴."""
    handles = []
    for lfem, hz, color in zip(lfem_designs, hz_designs, ORDER_COLORS):
        x, y = ecdf(lfem["jump"][lfem["solid"]])
        line_l, = ax_lfem.plot(x, y, color=color, lw=1.7, ls="-",
                               label=f"LFEM $p = {lfem['label'][-1]}$")
        x, y = ecdf(hz["jump"][hz["solid"]])
        line_h, = ax_hz.plot(x, y, color=color, lw=1.7, ls="--",
                             label=f"HZMFEM $k = {hz['label'][-1]}$")
        handles.extend([line_l, line_h])
    line = ax_lfem.axvline(DELTA_G, color=COLOR_DELTA, lw=1.3, ls="--",
                           label=r"$\delta_g = 0.005$")
    handles.append(line)
    line = ax_lfem.axvline(4.0 * DELTA_G, color=COLOR_DELTA, lw=1.3, ls=":",
                           label=r"$4\delta_g$")
    handles.append(line)

    for axes, xlim, xticks in ((ax_hz, CDF_XLIM_HZ, CDF_XTICKS_HZ),
                               (ax_lfem, CDF_XLIM_LFEM, CDF_XTICKS_LFEM)):
        axes.set_xscale("log")
        axes.set_xlim(*xlim)
        axes.set_xticks(xticks)
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes.set_ylim(0.0, 1.0)
        axes.tick_params(labelsize=9)
        axes.grid(True, ls=":", alpha=0.5)
    ax_hz.set_ylabel("实体单元累积占比", fontsize=11, fontproperties=font)
    ax_lfem.tick_params(labelleft=False, left=False)

    # 断轴: 去掉相邻脊线, 在断口两端画斜杠.
    ax_hz.spines["right"].set_visible(False)
    ax_lfem.spines["left"].set_visible(False)
    mark = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=10, linestyle="none",
                color="k", mec="k", mew=1, clip_on=False)
    ax_hz.plot([1, 1], [0, 1], transform=ax_hz.transAxes, **mark)
    ax_lfem.plot([0, 0], [0, 1], transform=ax_lfem.transAxes, **mark)
    return handles


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE
    lfem = [load_design(base, f"lfem-{order}") for order in ORDERS]
    huzhang = [load_design(base, f"huzhang-{order}") for order in ORDERS]
    ZH = chinese_font()

    fig = plt.figure(figsize=(12.5, 7.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.28, hspace=0.30,
                          left=0.03, right=0.98, top=0.93, bottom=0.19)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    plot_map(fig, ax_a, lfem[0],
             f"(a) LFEM $p = 2$ 构型实体带上 $A_e$ 的空间分布（{int(lfem[0]['solid'].sum())} 个单元）", ZH)
    plot_map(fig, ax_b, huzhang[0],
             f"(b) HZMFEM $k = 2$ 构型实体带上 $A_e$ 的空间分布（{int(huzhang[0]['solid'].sum())} 个单元）", ZH)
    add_shared_colorbar(fig, ax_a, ax_b, ZH)

    gs_c = gs[:, 1].subgridspec(1, 2, width_ratios=[3, 5], wspace=0.05)
    ax_hz = fig.add_subplot(gs_c[0, 0])
    ax_lfem = fig.add_subplot(gs_c[0, 1], sharey=ax_hz)
    handles = plot_cdf(ax_hz, ax_lfem, lfem, huzhang, ZH)
    # (c) 的标题与横轴标签跨两段居中.
    fig.canvas.draw()
    box_l, box_r = ax_hz.get_position(), ax_lfem.get_position()
    x_mid = 0.5 * (box_l.x0 + box_r.x1)
    fig.text(x_mid, box_r.y1 + 0.012,
             "(c) LFEM $p = 2$–$4$ 与 HZMFEM $k = 2$–$4$ 构型上 $A_e$ 的经验累积分布",
             ha="center", va="bottom", fontsize=11, fontproperties=ZH)
    fig.text(x_mid, box_r.y0 - 0.045, r"$A_e$（以 $\bar\sigma$ 无量纲化；横轴断开）",
             ha="center", va="top", fontsize=11, fontproperties=ZH)
    fig.legend(handles=handles, loc="lower center", ncol=4, prop=ZH,
               bbox_to_anchor=(x_mid, 0.01), **LEGEND_STYLE)

    save_figure(fig, "stress_traction_jump", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
